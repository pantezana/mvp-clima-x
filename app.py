import streamlit as st
import tweepy

st.set_page_config(page_title="MVP Clima en X", layout="wide")
st.title("📊 MVP – Clima del Tema en X")

bearer_token = st.secrets["X_BEARER_TOKEN"]

client = tweepy.Client(bearer_token=bearer_token)

st.success("Conectado a X correctamente ✅")

