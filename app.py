import streamlit as st
import tweepy
import re
import pandas as pd
from datetime import datetime, timedelta

st.set_page_config(page_title="MVP Clima en X", layout="wide")
st.title("📊 MVP – Clima del Tema en X")

bearer_token = st.secrets["X_BEARER_TOKEN"]
client = tweepy.Client(bearer_token=bearer_token)

st.success("Conectado a X correctamente ✅")

# ─────────────────────────────
# Inputs
# ─────────────────────────────
query = st.text_input("Palabras clave / hashtags")
time_range = st.selectbox("Rango temporal", ["24 horas", "7 días", "30 días"])

# Lista simple (MVP) de departamentos/ciudades clave para inferir ubicación
PERU_PLACES = [
    "Amazonas","Áncash","Apurímac","Arequipa","Ayacucho","Cajamarca","Callao","Cusco",
    "Huancavelica","Huánuco","Ica","Junín","La Libertad","Lambayeque","Lima","Loreto",
    "Madre de Dios","Moquegua","Pasco","Piura","Puno","San Martín","Tacna","Tumbes","Ucayali",
    "Trujillo","Chiclayo","Huancayo","Iquitos","Tarapoto","Pucallpa","Juliaca"
]

def get_start_time(option):
    if option == "24 horas":
        return datetime.utcnow() - timedelta(hours=24)
    if option == "7 días":
        return datetime.utcnow() - timedelta(days=7)
    return datetime.utcnow() - timedelta(days=30)

def infer_peru_location(profile_location: str, profile_desc: str):
    loc = (profile_location or "").strip()
    desc = (profile_desc or "").strip()

    haystack = f"{loc} {desc}".lower()

    peru_signals = ["perú", "peru", "🇵🇪", "lima", "cusco", "arequipa", "piura", "callao"]
    mentions_peru = any(s in haystack for s in peru_signals)

    for place in PERU_PLACES:
        if re.search(rf"\b{re.escape(place.lower())}\b", haystack):
            if loc and place.lower() in loc.lower():
                return place, "Media", "Perfil (location)"
            return place, "Baja", "Bio/Descripción"

    if loc and ("perú" in loc.lower() or "peru" in loc.lower() or "🇵🇪" in loc):
        return "Perú (sin región)", "Baja", "Perfil (location)"

    if not loc and not desc:
        return "No disponible", "N/A", "Sin datos"

    if mentions_peru:
        return "Perú (no identificada)", "Baja", "Señales en perfil/bio"

    return "No inferible", "N/A", "Sin señales claras"

# ─────────────────────────────
# Acción
# ─────────────────────────────
if st.button("Buscar en X"):
    if not query:
        st.warning("Ingresa una palabra clave")
        st.stop()

    start_time = get_start_time(time_range).isoformat("T") + "Z"

    response = client.search_recent_tweets(
        query=query,
        start_time=start_time,
        max_results=50,
        tweet_fields=["created_at", "public_metrics", "author_id"],
        expansions=["author_id"],
        user_fields=["username", "name", "location", "description"]
    )

    if not response.data:
        st.warning("No se encontraron resultados")
        st.stop()

    # Mapa author_id -> user
    users_by_id = {}
    if response.includes and "users" in response.includes:
        users_by_id = {u.id: u for u in response.includes["users"]}

    data = []
    for t in response.data:
        u = users_by_id.get(t.author_id)

        username = getattr(u, "username", None) if u else None
        name = getattr(u, "name", None) if u else None
        profile_location = getattr(u, "location", None) if u else None
        profile_desc = getattr(u, "description", None) if u else None

        ubicacion, confianza, fuente = infer_peru_location(profile_location, profile_desc)
        tweet_url = f"https://x.com/{username}/status/{t.id}" if username else ""

        data.append({
            "Autor": f"@{username}" if username else (name or "Desconocido"),
            "URL": tweet_url,
            "Texto": t.text,
            "Fecha": t.created_at,
            "Likes": t.public_metrics.get("like_count", 0),
            "Retweets": t.public_metrics.get("retweet_count", 0),
            "Ubicación inferida": ubicacion,
            "Confianza": confianza,
            "Fuente ubic.": fuente
        })

    df = pd.DataFrame(data)

    st.subheader("Resultados encontrados")
    st.caption("Nota: la ubicación NO es exacta; es una inferencia basada en 'location' del perfil y/o bio. Úsala solo como aproximación.")

    # ── Filtro por ubicación inferida
    ubicaciones = ["(Todas)"] + sorted(df["Ubicación inferida"].dropna().unique().tolist())
    filtro_ubic = st.selectbox("Filtrar por ubicación inferida", ubicaciones)

    df_view = df.copy()
    if filtro_ubic != "(Todas)":
        df_view = df_view[df_view["Ubicación inferida"] == filtro_ubic]

    # ── URL clicable
    def make_link(url):
        if not url:
            return ""
        return f'<a href="{url}" target="_blank">Abrir</a>'

    df_view["Link"] = df_view["URL"].apply(make_link)

    cols_to_show = [
        "Autor", "Link", "Texto", "Fecha",
        "Likes", "Retweets",
        "Ubicación inferida", "Confianza", "Fuente ubic."
    ]

    st.markdown(
        df_view[cols_to_show].to_html(escape=False, index=False),
        unsafe_allow_html=True
    )

    st.download_button(
        "Descargar resultados (CSV)",
        data=df_view.to_csv(index=False).encode("utf-8"),
        file_name="resultados_x.csv",
        mime="text/csv"
    )
