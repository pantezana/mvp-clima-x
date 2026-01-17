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

from datetime import datetime, timedelta
import pandas as pd

def get_start_time(option):
    if option == "24 horas":
        return datetime.utcnow() - timedelta(hours=24)
    if option == "7 días":
        return datetime.utcnow() - timedelta(days=7)
    return datetime.utcnow() - timedelta(days=30)

if st.button("Buscar en X"):
    start_time = get_start_time(time_range).isoformat("T") + "Z"

    tweets = client.search_recent_tweets(
        query=query,
        start_time=start_time,
        max_results=50,
        tweet_fields=["created_at","public_metrics"],
        user_fields=["location","description"],
        expansions="author_id"
    )

    if tweets.data:
        data = []
        for t in tweets.data:
            data.append({
                "texto": t.text,
                "fecha": t.created_at,
                "likes": t.public_metrics["like_count"]
            })

        df = pd.DataFrame(data)
        st.dataframe(df)
    else:
        st.warning("No se encontraron resultados")
