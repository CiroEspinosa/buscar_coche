import joblib 
import streamlit as st
import pandas as pd

model = joblib.load("model/svm_model.pkl")
df = pd.read_csv('data/coches.csv')

st.title('COCHES:')


st.header('Ingresar Datos:')

efficiency = st.number_input("Litros por 100 kilómetros:", value=15.70)
displacement = st.number_input("Centímetros cúbicos:", value=5030.81)
horsepower = st.number_input("Caballos de potencia:", value=130.0)
weight = st.number_input("Peso en kg:", value=1589.386368)
acceleration = st.number_input("Aceleración en segundos (0 a 60 mph):", value=12.0)

if st.button("Buscar"):
    features = [[efficiency, displacement, horsepower, weight, acceleration]]
    car_name = model.predict(features)
    st.success(f'El coche más similar en la base de datos es: {car_name[0]}') 
if st.checkbox("Ver datos"):
    st.write(df)


    

