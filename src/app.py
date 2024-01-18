import streamlit as st
import pandas as pd
import joblib 
from sklearn.preprocessing import StandardScaler

model = joblib.load("model/svm_model.pkl")
df = pd.read_csv('data/coches.csv')

st.title('COCHES:')


st.header('Ingresar Datos:')
efficiency = st.number_input("Litros por 100 kilómetros:", value=15.70)
displacement = st.number_input("Centímetros cúbicos:", value=5030.81)
horsepower = st.number_input("Caballos de potencia:", value=130.0)
weight = st.number_input("Peso en kg:", value=1589.386368)
acceleration = st.number_input("Aceleración en segundos (0 a 60 mph):", value=12.0)


origin_mapping = {'EEUU': 1, 'Europa': 2, 'Japón': 3}
origin = st.selectbox("Origen:", list(origin_mapping.keys()), index=0)

origin_number = origin_mapping[origin]

year = st.number_input("Año del modelo:", value=70)



if st.button("Buscar"):
    features = [[efficiency, displacement, horsepower, weight, acceleration, origin_number, year]]
    car_name = model.predict(features)
    st.success(f'El coche más similar en la base de datos es: {car_name[0]}') 
if st.checkbox("Ver datos"):
    st.write(df)


    

