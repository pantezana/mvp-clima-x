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

query = st.text_input("Palabras clave / hashtags")

time_range = st.selectbox(
    "Rango temporal",
    ["24 horas", "7 días", "30 días"]
)


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

            # Mostrar tabla
            st.dataframe(df, use_container_width=True)

           # ─────────────────────────────
# ANÁLISIS GERENCIAL
# ─────────────────────────────

st.markdown("## 🧠 Resumen Ejecutivo Automático")

# --- Preparación de texto
textos = df["Texto"].str.lower()

# Stopwords básicas en español (MVP)
stopwords = set([
    "de","la","que","el","en","y","a","los","del","se","las","por","un","para","con",
    "no","una","su","al","lo","como","más","pero","sus","le","ya","o","este","sí",
    "porque","esta","entre","cuando","muy","sin","sobre"
])

def limpiar_texto(texto):
    palabras = re.findall(r"\b[a-záéíóúñ]+\b", texto)
    return [p for p in palabras if p not in stopwords and len(p) > 3]

# --- Narrativas dominantes
todas_palabras = []
for t in textos:
    todas_palabras.extend(limpiar_texto(t))

top_palabras = pd.Series(todas_palabras).value_counts().head(10)

# --- Sentimiento simple (léxico)
positivas = set([
    # Aprobación directa
    "bueno","bien","positivo","excelente","correcto","adecuado","acertado","justo",
    
    # Progreso / avance
    "avance","avanzar","mejora","mejorar","progreso","logro","logrado","resultado",
    
    # Confianza / esperanza
    "confianza","esperanza","optimismo","tranquilidad","seguridad","estabilidad",
    
    # Gestión / política pública
    "cumple","cumplió","eficiente","efectivo","funciona","solución","resuelve",
    
    # Legitimidad / respaldo
    "apoyo","respaldo","legítimo","necesario","importante","prioritario",
    
    # Éxito / impacto
    "exitoso","beneficio","beneficioso","impacto","positivo","histórico"
])

negativas = set([
    # Rechazo directo
    "malo","mal","negativo","pésimo","terrible","inaceptable","vergonzoso",
    
    # Crisis / conflicto
    "crisis","conflicto","caos","problema","grave","colapso","fracaso",
    
    # Desconfianza / enojo
    "indignación","enojo","rabia","molestia","hartazgo","descontento",
    
    # Gestión deficiente
    "ineficiente","incapaz","incompetente","error","fallo","improvisación",
    
    # Corrupción / legitimidad
    "corrupción","corrupto","ilegal","irregular","fraude","impunidad",
    
    # Miedo / riesgo
    "peligro","amenaza","riesgo","inseguridad","violencia","abuso",
    
    # Protesta / rechazo social
    "rechazo","repudio","protesta","denuncia","escándalo"
])


def calcular_sentimiento(texto):
    palabras = limpiar_texto(texto)
    pos = sum(1 for p in palabras if p in positivas)
    neg = sum(1 for p in palabras if p in negativas)
    if pos > neg:
        return "Positivo"
    if neg > pos:
        return "Negativo"
    return "Neutral"

df["Sentimiento"] = df["Texto"].apply(calcular_sentimiento)

# --- Métricas de temperatura
total = len(df)
pct_pos = round((df["Sentimiento"] == "Positivo").mean() * 100, 1)
pct_neg = round((df["Sentimiento"] == "Negativo").mean() * 100, 1)
pct_neu = round((df["Sentimiento"] == "Neutral").mean() * 100, 1)

if pct_neg > 40:
    temperatura = "🔴 Riesgo reputacional"
elif pct_pos > 60:
    temperatura = "🟢 Clima favorable"
else:
    temperatura = "🟡 Clima mixto / neutro"

# --- Mostrar resumen ejecutivo
st.markdown("### 📌 Principales hallazgos")

st.markdown(f"""
- **Volumen analizado:** {total} publicaciones  
- **Temperatura del tema:** {temperatura}  
- **Distribución de sentimiento:**  
  - Positivo: {pct_pos}%  
  - Neutral: {pct_neu}%  
  - Negativo: {pct_neg}%  
- **Narrativas dominantes:** {', '.join(top_palabras.index.tolist())}
""")

# --- Riesgos y oportunidades
st.markdown("### ⚠️ Riesgos identificados")
if pct_neg > 30:
    st.markdown("- Presencia relevante de mensajes negativos que podrían escalar si aumenta el volumen.")
else:
    st.markdown("- No se identifican riesgos reputacionales significativos en el periodo analizado.")

st.markdown("### 🌱 Oportunidades")
if pct_pos > pct_neg:
    st.markdown("- Predominan mensajes favorables que pueden reforzarse con información clara y oportuna.")
else:
    st.markdown("- Existe oportunidad de clarificar información y reducir ambigüedad en la conversación.")

st.markdown("### 👀 Qué monitorear mañana")
st.markdown("""
- Evolución del volumen de publicaciones.
- Aparición de nuevos términos o hashtags.
- Cambios en la proporción de sentimiento negativo.
- Mayor actividad desde regiones específicas.
""")

st.markdown("### ⚖️ Advertencia metodológica")
st.caption(
    "Este análisis se basa en publicaciones públicas de X, con inferencia aproximada de ubicación "
    "y análisis automático de texto. No representa la opinión de la totalidad de la población "
    "y debe interpretarse como una señal temprana, no como medición estadística."
)

