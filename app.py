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

import re
import pandas as pd
from datetime import datetime, timedelta

# Lista simple (MVP) de departamentos/ciudades clave para inferir ubicación
PERU_PLACES = [
    "Amazonas","Áncash","Apurímac","Arequipa","Ayacucho","Cajamarca","Callao","Cusco",
    "Huancavelica","Huánuco","Ica","Junín","La Libertad","Lambayeque","Lima","Loreto",
    "Madre de Dios","Moquegua","Pasco","Piura","Puno","San Martín","Tacna","Tumbes","Ucayali",
    # Ciudades muy usadas en perfil
    "Trujillo","Chiclayo","Huancayo","Iquitos","Tarapoto","Pucallpa","Juliaca"
]

def get_start_time(option):
    if option == "24 horas":
        return datetime.utcnow() - timedelta(hours=24)
    if option == "7 días":
        return datetime.utcnow() - timedelta(days=7)
    return datetime.utcnow() - timedelta(days=30)

def infer_peru_location(profile_location: str, profile_desc: str):
    """
    Inferencia ética y simple:
    - Usa 'location' del perfil (si existe)
    - Busca menciones a lugares de Perú
    - Devuelve (ubicacion_inferida, confianza, fuente)
    """
    loc = (profile_location or "").strip()
    desc = (profile_desc or "").strip()

    # Normalizamos texto para comparar
    haystack = f"{loc} {desc}".lower()

    # Señales de Perú
    peru_signals = ["perú", "peru", "🇵🇪", "lima", "cusco", "arequipa", "piura", "callao"]
    mentions_peru = any(s in haystack for s in peru_signals)

    # Buscar match exacto (case-insensitive) de lista
    for place in PERU_PLACES:
        if re.search(rf"\b{re.escape(place.lower())}\b", haystack):
            # Confianza:
            # - Media si viene del campo location del perfil
            # - Baja si viene solo de la bio/description
            if loc and place.lower() in loc.lower():
                return place, "Media", "Perfil (location)"
            return place, "Baja", "Bio/Descripción"

    # Si solo dice "Perú" sin región
    if loc and ("perú" in loc.lower() or "peru" in loc.lower() or "🇵🇪" in loc):
        return "Perú (sin región)", "Baja", "Perfil (location)"

    # Sin datos
    if not loc and not desc:
        return "No disponible", "N/A", "Sin datos"

    # Algo hay, pero no identificamos región
    if mentions_peru:
        return "Perú (no identificada)", "Baja", "Señales en perfil/bio"
    return "No inferible", "N/A", "Sin señales claras"

if st.button("Buscar en X"):
    if not query:
        st.warning("Ingresa una palabra clave")
    else:
        start_time = get_start_time(time_range).isoformat("T") + "Z"

        # Pedimos también info del autor vía expansions
        response = client.search_recent_tweets(
            query=query,
            start_time=start_time,
            max_results=50,
            tweet_fields=["created_at", "public_metrics", "author_id"],
            expansions=["author_id"],
            user_fields=["username", "name", "location", "description"]
        )

        if response.data:
            # Mapa author_id -> objeto user
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

                # Link público al post (siempre que tengamos username)
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
        
        # ── 1) Filtro por ubicación inferida
        ubicaciones = ["(Todas)"] + sorted([u for u in df["Ubicación inferida"].dropna().unique().tolist()])
        
        col_f1, col_f2 = st.columns([1, 2])
        with col_f1:
            filtro_ubic = st.selectbox("Filtrar por ubicación inferida", ubicaciones)
        with col_f2:
            st.write("")  # espacio visual
        
        df_view = df.copy()
        if filtro_ubic != "(Todas)":
            df_view = df_view[df_view["Ubicación inferida"] == filtro_ubic]
        
        # ── 2) URL clicable: creamos una columna con link HTML
        def make_link(url):
            if not url:
                return ""
            return f'<a href="{url}" target="_blank">Abrir</a>'
        
        df_view = df_view.copy()
        df_view["Link"] = df_view["URL"].apply(make_link)
        
        # Elegimos columnas a mostrar (ocultamos URL cruda)
        cols_to_show = [
            "Autor", "Link", "Texto", "Fecha",
            "Likes", "Retweets",
            "Ubicación inferida", "Confianza", "Fuente ubic."
        ]
        
        # Render tabla HTML con links clicables
        st.markdown(
            df_view[cols_to_show].to_html(escape=False, index=False),
            unsafe_allow_html=True
        )
        
        # Extra: botón para descargar CSV (útil)
        st.download_button(
            "Descargar resultados (CSV)",
            data=df_view.to_csv(index=False).encode("utf-8"),
            file_name="resultados_x.csv",
            mime="text/csv"
        )
        else:
            st.warning("No se encontraron resultados")

