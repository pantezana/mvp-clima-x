import streamlit as st
import tweepy

st.set_page_config(page_title="MVP Clima en X", layout="wide")
st.title("📊 MVP – Clima del Tema en X")

bearer_token = st.secrets["X_BEARER_TOKEN"]

client = tweepy.Client(bearer_token=bearer_token)

st.success("Conectado a X correctamente ✅")

query = st.text_input("Palabras clave / hashtags")
time_range = st.selectbox(
    "Rango temporal",
    ["24 horas", "7 días", "30 días"]
)

if st.button("Consultar"):
    st.write("Consulta enviada:", query, time_range)
