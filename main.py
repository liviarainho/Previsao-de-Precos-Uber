import streamlit as st
import pandas as pd
import googlemaps
import folium
from streamlit_folium import st_folium
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import datetime

# Função para treinar o modelo
def train_model():
    # Carregar o arquivo CSV
    file_path = 'Projeto-Integrado-Negocios/Previsão-de-Corridas-do-Uber/uber_peru_2010.csv'
    df = pd.read_csv(file_path, delimiter=';')

    # Remover entradas onde 'end_state' é diferente de 'drop off'
    df = df[df['end_state'] == 'drop off']

    # Transformar colunas 'start_at', 'end_at' e 'arrived_at' para apenas a hora
    df['start_at'] = pd.to_datetime(df['start_at'], dayfirst=True).dt.hour
    df['end_at'] = pd.to_datetime(df['end_at'], dayfirst=True).dt.hour
    df['arrived_at'] = pd.to_datetime(df['arrived_at'], dayfirst=True).dt.hour

    df = df[df['duration'] > 10]
    df = df[df['distance'] > 50]
    df = df[df['price'] > 0]

    columns = ['price', 'icon', 'start_type', 'start_at', 'end_at', 'arrived_at', 'distance', 'duration', 'source']
    features = ['icon', 'start_type', 'start_at', 'end_at', 'arrived_at', 'distance', 'duration', 'source']
    df = df[columns]
    # Remover entradas com valores NaN
    df = df.dropna()
    
    # Selecionar colunas para a regressão
    X = df[features]
    y = df['price']

    # Dividir os dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Criar o pré-processador com OneHotEncoder para colunas categóricas
    categorical_features = ['icon', 'start_type', 'source']
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(drop='first'), categorical_features)],
        remainder='passthrough')

    # Criar e treinar o pipeline com o pré-processador e o modelo Gradient Boosting
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', GradientBoostingRegressor())
    ])
    model.fit(X_train, y_train)
    
    return model

# Função para obter informações do Google Maps API
def get_trip_info(api_key, origin, destination):
    gmaps = googlemaps.Client(key=api_key)
    now = datetime.datetime.now()
    directions = gmaps.directions(origin, destination, mode="driving", departure_time=now)
    
    if directions:
        leg = directions[0]['legs'][0]
        distance = leg['distance']['value']  # em metros
        duration = leg['duration']['value']  # em segundos
        end_time = (now + datetime.timedelta(seconds=duration)).hour
        return distance, duration, end_time
    return None, None, None

# Configurações do Streamlit
st.title('Previsão de Preços de Viagens - Uber Peru 2010')
st.sidebar.header('Parâmetros da Viagem')

# Entradas do usuário
icon = st.sidebar.selectbox('Tipo de Veículo', ['executive', 'easy', 'group'])
start_type = st.sidebar.selectbox('Tipo de Início', ['asap', 'reserved'])
source = st.sidebar.selectbox('Fonte', ['iPhone', 'Android', 'iPad', 'web'])
start_time = st.sidebar.slider('Hora de Início', 0, 23, 12)

# Google Maps API Key
api_key = ''

# Função para inicializar um mapa com um evento de clique
def create_click_map(lat, lon, marker_color, marker_popup):
    m = folium.Map(location=[lat, lon], zoom_start=12)
    click_marker = folium.ClickForMarker(popup=marker_popup)
    m.add_child(click_marker)
    return m, click_marker

# Coordenadas iniciais do mapa
map_center = [-23.538027401700948, -46.6711430618589]  # Centro especificado

# Mapa interativo para seleção do ponto de origem
st.header('Selecione o Ponto de Origem no Mapa')
origin_map, origin_click_marker = create_click_map(map_center[0], map_center[1], 'blue', 'Origem')
origin_map_data = st_folium(origin_map, width=700, height=500)

# Mapa interativo para seleção do ponto de destino
st.header('Selecione o Ponto de Destino no Mapa')
destination_map, destination_click_marker = create_click_map(map_center[0], map_center[1], 'red', 'Destino')
destination_map_data = st_folium(destination_map, width=700, height=500)

# Função para obter coordenadas dos cliques no mapa
def get_click_coords(map_data):
    if 'last_clicked' in map_data and map_data['last_clicked'] is not None:
        return map_data['last_clicked']['lat'], map_data['last_clicked']['lng']
    return None, None

# Obter coordenadas dos cliques
start_coords = get_click_coords(origin_map_data)
end_coords = get_click_coords(destination_map_data)

# Mostrar coordenadas selecionadas
if start_coords:
    st.write(f"Coordenadas de Origem: {start_coords}")
else:
    st.write("Por favor, selecione o ponto de origem no mapa.")

if end_coords:
    st.write(f"Coordenadas de Destino: {end_coords}")
else:
    st.write("Por favor, selecione o ponto de destino no mapa.")

# Carregar o modelo treinado
model = train_model()

# Fazer a previsão
if st.button('Prever Preço'):
    if api_key and start_coords and end_coords:
        origin = f"{start_coords[0]},{start_coords[1]}"
        destination = f"{end_coords[0]},{end_coords[1]}"
        distance, duration, end_time = get_trip_info(api_key, origin, destination)
        if distance and duration and end_time:
            new_trip = pd.DataFrame({
                'icon': [icon],
                'start_type': [start_type],
                'start_at': [start_time],
                'end_at': [end_time],
                'arrived_at': [end_time],
                'distance': [distance],
                'duration': [duration],
                'source': [source]
            })
            predicted_price = model.predict(new_trip)[0] / 100  # Ajustando o valor para milhares
            predicted_price_brl = predicted_price * 1.42
            st.write(f"Preço Previsto: R$ {predicted_price_brl:,.2f}".replace('.', ','))
        else:
            st.error('Falha ao obter informações da viagem.')
    else:
        st.error('Por favor, preencha todos os campos e selecione os pontos no mapa.')
