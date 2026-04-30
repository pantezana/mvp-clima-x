import streamlit as st
import tweepy
import re
import pandas as pd
import requests
import time
import plotly.express as px
import json
import io
from reportlab.lib.pagesizes import A4, landscape
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image, Table, TableStyle, LongTable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from datetime import datetime, timedelta

# ─────────────────────────────
# 🔎 Chequeo técnico: Plotly → PNG (Kaleido)
# Solo diagnóstico (no afecta lógica)
# ─────────────────────────────
#try:
#    import kaleido  # requerido por plotly.to_image()
#    st.success("✅ Kaleido OK: exportación PNG habilitada (PDF con gráficos funcionará)")
#except Exception as e:
#    st.error(f"❌ Kaleido NO disponible: {type(e).__name__} — el PDF NO podrá incluir gráficos")

st.set_page_config(page_title="MVP Clima en X", layout="wide")
st.title("🖥️ Clima de Opinión del Tema en X (Twiter)")

bearer_token = st.secrets["X_BEARER_TOKEN"]
client = tweepy.Client(bearer_token=bearer_token)

# ─────────────────────────────
# Panel de control lateral
# ─────────────────────────────
with st.sidebar:
    st.header("⚙️ Parámetros")

    query = st.text_input("Palabras clave / hashtags")

    time_range = st.selectbox(
        "Rango temporal",
        ["24 horas", "48 horas", "72 horas", "7 días"]
    )

    limite_opcion = st.selectbox(
        "Límite de publicaciones (cuota X)",
        ["50", "100", "200", "500", "1000", "Sin límite (hasta donde llegue X)"],
        index=2
    )

    max_posts = None if "Sin límite" in limite_opcion else int(limite_opcion)

    st.markdown("### Tipo de contenido")
    c1, c2 = st.columns(2)
    with c1:
        incluir_originales = st.checkbox("Posts originales", value=True)
        incluir_quotes = st.checkbox("RT con cita (quote)", value=True)
    with c2:
        incluir_retweets = st.checkbox("RT puros", value=True)
        incluir_replies = st.checkbox("Replies (comentarios)", value=False)

    if not (incluir_originales or incluir_retweets or incluir_quotes):
        st.warning("Selecciona al menos un tipo: Originales, RT puros o Quotes.")

    MODELOS_SENTIMIENTO = {
        "BETO (ES) – recomendado": "finiteautomata/beto-sentiment-analysis",
        "Robertuito (ES) – social": "pysentimiento/robertuito-sentiment-analysis",
        "Twitter-RoBERTa (X) – actual": "cardiffnlp/twitter-roberta-base-sentiment-latest",
    }

    modelo_nombre = st.selectbox(
        "Modelo de sentimiento (IA)",
        list(MODELOS_SENTIMIENTO.keys()),
        index=0
    )

modelo_hf_id = MODELOS_SENTIMIENTO[modelo_nombre]
HF_MODEL_URL = f"https://router.huggingface.co/hf-inference/models/{modelo_hf_id}"
debug_gemini = False

# Regla simple de validación (neófito-friendly)
if not (incluir_originales or incluir_retweets or incluir_quotes):
    st.warning("Selecciona al menos un tipo de contenido (Originales, RT puros o Quotes).")

# Nota de uso (educativa)

def build_x_query(base_query: str, incluir_originales: bool, incluir_retweets: bool, incluir_quotes: bool) -> str:
    """
    Construye query para X aplicando filtros SOLO cuando el usuario eligió 1 solo tipo.
    Si el usuario eligió combinaciones, NO filtra (lo resolveremos con llamadas separadas).
    """
    q = (base_query or "").strip()
    if not q:
        return ""

    # Siempre agrupamos el término base
    base = f"({q})"

    seleccionados = sum([incluir_originales, incluir_retweets, incluir_quotes])

    # ✅ Si SOLO eligió 1 tipo, filtramos en el query para ahorrar cuota
    if seleccionados == 1:
        if incluir_retweets:
            # RT puros: retweet sí, quote no
            return f"{base} is:retweet -is:quote"
        if incluir_quotes:
            # Quotes
            return f"{base} is:quote"
        if incluir_originales:
            # Originales: NO retweet, NO quote
            # (Si quieres permitir replies, esto está OK. Si quieres excluir replies también, agrega: -is:reply)
            return f"{base} -is:retweet -is:quote"

    # ✅ Si eligió 2 o 3 tipos, devolvemos SOLO el base y resolvemos con múltiples llamadas
    return base


# Query final (por ahora igual al base; se usa en la llamada)
query_final = build_x_query(query, incluir_originales, incluir_retweets, incluir_quotes)

# Guardamos selección en session_state (por si luego cacheamos)
st.session_state["incl_originales"] = incluir_originales
st.session_state["incl_retweets"] = incluir_retweets
st.session_state["incl_quotes"] = incluir_quotes
st.session_state["query_final"] = query_final
st.session_state["incl_replies"] = incluir_replies

# ─────────────────────────────
# Parámetros MVP de Replies (control cuota)
# ─────────────────────────────
if time_range in ["7 días"]:
    TOP_TWEETS_CONV_REPLIES = 10
    TOP_TWEETS_AMP_REPLIES  = 10
else:
    TOP_TWEETS_CONV_REPLIES = 20
    TOP_TWEETS_AMP_REPLIES  = 20
MAX_REPLIES_POR_TWEET   = 50     # máximo replies por tweet objetivo (control de cuota)
MIN_REPLIES_ALERTA      = 20     # mínimo replies para considerar temperatura/alertas como “señal razonable”
W_REPLIES = 5                    # Score = Interacción + (W_REPLIES * Replies)


# ─────────────────────────────
# Persistencia de resultados (evita que se borre al cambiar selects)
# ─────────────────────────────
if "HAS_RESULTS" not in st.session_state:
    st.session_state["HAS_RESULTS"] = False

def _save_results(**kwargs):
    for k, v in kwargs.items():
        st.session_state[k] = v
    st.session_state["HAS_RESULTS"] = True

def _clear_results():
    # Borra solo lo necesario (no toca tus inputs)
    keys_to_drop = [
        "HAS_RESULTS",
        "DF_CONV_RANK", "DF_AMP_RANK",
        "DF_REPLIES", "DF_REPLIES_CONV_AGG", "DF_REPLIES_AMP_AGG",
        "COLS_TOP_AMP", "COLS_CONV",
    ]
    for k in keys_to_drop:
        if k in st.session_state:
            del st.session_state[k]
    st.session_state["HAS_RESULTS"] = False

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
    if option == "48 horas":
        return datetime.utcnow() - timedelta(hours=48)
    if option == "72 horas":
        return datetime.utcnow() - timedelta(hours=72)
    if option == "7 días":
        return datetime.utcnow() - timedelta(hours=168)
    return datetime.utcnow() - timedelta(hours=168)

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

# ─────────────────────────────
# Sentimiento con Hugging Face (CardiffNLP Twitter-RoBERTa)
# ─────────────────────────────

def sentimiento_hf(texto: str):
    """
    Devuelve: (sentimiento, score)
    - sentimiento: Positivo / Neutral / Negativo o None
    - score: confianza 0..1 o None
    """
    HF_TOKEN = st.secrets.get("HF_TOKEN", "")
    if not HF_TOKEN:
        return None, None

    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {"inputs": texto[:512]}

    try:
        r = requests.post(HF_MODEL_URL, headers=headers, json=payload, timeout=25)
        if r.status_code != 200:
            return None, None

        data = r.json()

        # A veces viene [[...]]
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
            data = data[0]

        if not isinstance(data, list) or len(data) == 0:
            return None, None

        best = max(data, key=lambda x: x.get("score", 0))
        label = best.get("label", "")
        score = best.get("score", 0)

        mapping = {
            "positive": "Positivo",
            "neutral": "Neutral",
            "negative": "Negativo",
            "pos": "Positivo",
            "neu": "Neutral",
            "neg": "Negativo",
            "LABEL_2": "Positivo",
            "LABEL_1": "Neutral",
            "LABEL_0": "Negativo",
        }

        sentimiento = mapping.get(label.lower(), mapping.get(label, None))
        if sentimiento is None:
            return None, None

        return sentimiento, round(float(score), 3)

    except Exception:
        return None, None

def resumen_ejecutivo_gemini(payload: dict, debug: bool = False):
    """
    Retorna: (texto_resumen, status_str)
    texto_resumen: Markdown con 3 secciones:
      **Narrativa:** ...
      **Riesgos:** ...
      **Oportunidades:** ...
    """
    api_key = st.secrets.get("GEMINI_API_KEY", "")
    if not api_key:
        return None, "Gemini: falta GEMINI_API_KEY en secrets"

    model = "gemini-2.5-flash"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}

    insumos_json = json.dumps(payload, ensure_ascii=False)

    prompt = f"""
Actúa como un analista senior de clima social y especialista en interpretación de conversaciones públicas en X (Twitter).
Tu objetivo es ayudar a un tomador de decisiones con un resumen claro, profesional y NO propagandístico.

CONTEXTO:
Los INSUMOS provienen de publicaciones públicas en X sobre una temática (query) durante un rango temporal (time_range).
Incluyen volumen, distribución de sentimiento estimada, términos dominantes, ejemplos de posts con más interacción y (si está activado) métricas de replies: cantidad, % negativo y temperatura por conversación y amplificación.

IMPORTANTE:
- No inventes datos.
- Si la muestra es chica o la evidencia es insuficiente, dilo explícitamente.
- No repitas números literalmente si no aporta.
- No uses viñetas.

INSUMOS (JSON):
{insumos_json}

SALIDA OBLIGATORIA:
Escribe EXACTAMENTE 3 párrafos, cada uno iniciando con estas etiquetas (tal cual):

Narrativa: <explica en un párrafo la narrativa predominante usando top_terminos y ejemplos>
Riesgos: <explica en un párrafo los riesgos detectados (reputacional, amplificación, confusión, etc.)>
Oportunidades: <explica en un párrafo oportunidades accionables (aclaración, vocería, contenido informativo, monitoreo)>

Nada más. No agregues saludos ni conclusiones.
""".strip()

    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.2
        }
    }

    try:
        r = requests.post(url, headers=headers, json=body, timeout=35)

        if r.status_code != 200:
            return None, f"Gemini ERROR {r.status_code}: {r.text[:200]}"

        data = r.json()

        # ✅ OJO: Gemini puede devolver varias parts -> concatenamos todas
        parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
        text = "\n".join([p.get("text", "") for p in parts if p.get("text")]).strip()

        finish = data.get("candidates", [{}])[0].get("finishReason", "")

        if not text:
            return None, "Gemini: respuesta vacía"

        # Parseo estilo Apps Script: buscamos las 3 secciones
        m = re.search(
            r"(?is)^\s*Narrativa:\s*(.+?)\s*^\s*Riesgos:\s*(.+?)\s*^\s*Oportunidades:\s*(.+?)\s*$",
            text,
            re.MULTILINE
        )

        if not m:
            # Si no calza perfecto, igual devolvemos el texto para no perderlo
            return text, "Gemini OK (sin parseo exacto; mostrando texto tal cual)"

        narrativa = m.group(1).strip()
        riesgos = m.group(2).strip()
        oportunidades = m.group(3).strip()

        salida = (
            f"**Narrativa:** {narrativa}\n\n"
            f"**Riesgos:** {riesgos}\n\n"
            f"**Oportunidades:** {oportunidades}"
        )

        # Si llega truncado, lo reportamos (pero igual lo mostramos)
        if finish == "MAX_TOKENS":
            return salida, "Gemini OK (pero truncado por MAX_TOKENS)"

        return salida, "Gemini OK (3 secciones)"

    except Exception as e:
        return None, f"Gemini: excepción ({type(e).__name__})"

if "last_search_ts" not in st.session_state:
    st.session_state["last_search_ts"] = 0

def fetch_tweets_paginado(
    client,
    query,
    start_time,
    max_posts=None,
    tweet_fields=None,
    expansions=None,
    user_fields=None
):
    tweets_all = []
    users_by_id = {}
    next_token = None

    tweet_fields = tweet_fields or ["created_at", "public_metrics", "author_id"]
    expansions = expansions or ["author_id"]
    user_fields = user_fields or ["username", "name", "location", "description"]

    page_size = 100

    while True:
        if max_posts is not None and len(tweets_all) >= max_posts:
            break

        req_size = page_size
        if max_posts is not None:
            req_size = min(page_size, max_posts - len(tweets_all))
            req_size = max(10, req_size)

        resp = client.search_recent_tweets(
            query=query,
            start_time=start_time,
            max_results=req_size,
            tweet_fields=tweet_fields,
            expansions=expansions,
            user_fields=user_fields,
            next_token=next_token
        )

        if not resp or not resp.data:
            break

        tweets_all.extend(resp.data)

        if resp.includes and "users" in resp.includes:
            for u in resp.includes["users"]:
                users_by_id[u.id] = u

        meta = getattr(resp, "meta", {}) or {}
        next_token = meta.get("next_token")
        if not next_token:
            break

    return tweets_all, users_by_id

def fetch_originals_by_ids(client, original_ids: list[str]):
    """
    Devuelve un dict:
      original_id -> {"autor": "@username", "url": "...", "texto": "...", "conversation_id": "..."}
    """
    original_ids = [str(x) for x in original_ids if x]
    if not original_ids:
        return {}

    ids_chunk = original_ids[:100]

    resp = client.get_tweets(
        ids=ids_chunk,
        tweet_fields=["created_at", "public_metrics", "author_id", "conversation_id"],
        expansions=["author_id"],
        user_fields=["username", "name"]
    )

    users_by_id = {}
    if resp and resp.includes and "users" in resp.includes:
        for u in resp.includes["users"]:
            users_by_id[str(u.id)] = u

    out = {}
    if resp and resp.data:
        for tw in resp.data:
            uid = str(getattr(tw, "author_id", ""))
            u = users_by_id.get(uid)
            username = getattr(u, "username", None) if u else None
            autor = f"@{username}" if username else "Desconocido"
            oid = str(getattr(tw, "id", ""))

            out[oid] = {
                "autor": autor,
                "url": f"https://x.com/{username}/status/{oid}" if username else f"https://x.com/i/web/status/{oid}",
                "texto": getattr(tw, "text", "") or "",
                "conversation_id": str(getattr(tw, "conversation_id", "")) if getattr(tw, "conversation_id", None) else None
            }
    return out

def sentimiento_dominante_conservador(series_sent: pd.Series) -> str:
    """
    Devuelve el sentimiento dominante con desempate conservador:
    Negativo > Neutral > Positivo
    """
    if series_sent is None or len(series_sent) == 0:
        return "N/A"

    s = series_sent.dropna()
    if s.empty:
        return "N/A"

    counts = s.value_counts().to_dict()
    neg = counts.get("Negativo", 0)
    neu = counts.get("Neutral", 0)
    pos = counts.get("Positivo", 0)

    m = max(neg, neu, pos)
    # desempate conservador
    if neg == m and m > 0:
        return "Negativo"
    if neu == m and m > 0:
        return "Neutral"
    if pos == m and m > 0:
        return "Positivo"
    return "N/A"

def calc_temperatura_con_min(pct_neg: float, pct_pos: float, n: int, min_n: int = 20):
    """
    Temperatura con umbral mínimo de muestra:
    - si n < min_n -> "⚪ Insuficiente"
    """
    if n < min_n:
        return "⚪ Insuficiente"
    if pct_neg >= 40:
        return "🔴 Riesgo reputacional"
    if pct_pos >= 60 and pct_neg < 25:
        return "🟢 Clima favorable"
    return "🟡 Mixto / neutro"

def fetch_replies_for_conversation_id(client, conversation_id: str, start_time: str, max_replies: int = 50):
    if conversation_id is None:
        return []
    conversation_id = str(conversation_id).strip()
    if not conversation_id.isdigit():
        return []

    query = f"conversation_id:{conversation_id} is:reply"

    replies_all = []
    next_token = None
    page_size = 100

    while True:
        if max_replies is not None and len(replies_all) >= max_replies:
            break

        req_size = min(page_size, max_replies - len(replies_all)) if max_replies is not None else page_size
        req_size = max(10, req_size)

        try:
            resp = client.search_recent_tweets(
                query=query,
                start_time=start_time,
                max_results=req_size,
                tweet_fields=["created_at", "public_metrics", "author_id", "conversation_id", "lang"],
                expansions=["author_id"],
                user_fields=["username", "name", "location", "description"],
                next_token=next_token
            )

        except tweepy.errors.TooManyRequests as e:
            # ⏳ Esperar hasta reset (si viene en headers)
            reset_ts = 0
            try:
                reset_ts = int(e.response.headers.get("x-rate-limit-reset", "0"))
            except Exception:
                reset_ts = 0

            if reset_ts:
                wait_sec = max(5, reset_ts - int(time.time()) + 2)
                time.sleep(wait_sec)
                continue

            # fallback si no hay header
            time.sleep(20)
            continue

        except tweepy.errors.BadRequest:
            return []
        except Exception:
            return []

        if not resp or not resp.data:
            break

        replies_all.extend(resp.data)

        meta = getattr(resp, "meta", {}) or {}
        next_token = meta.get("next_token")
        if not next_token:
            break

    return replies_all

# --- Preparación de texto
            
# Stopwords básicas en español (MVP)
STOPWORDS = set([
    "de","la","que","el","en","y","a","los","del","se","las","por","un","para","con",
    "no","una","su","al","lo","como","más","pero","sus","le","ya","o","este","sí",
    "porque","esta","entre","cuando","muy","sin","sobre","https","http","tco","www"
])
            
def limpiar_texto(texto):
    # ✅ blindaje: si viene NaN/None/float, lo convertimos a ""
    if texto is None:
        texto = ""
    try:
        # pandas NaN es float y no tiene lower()
        if isinstance(texto, float) and pd.isna(texto):
            texto = ""
    except Exception:
        pass

    if not isinstance(texto, str):
        texto = str(texto)

    # ✅ quitar URLs (http/https) y t.co
    texto = re.sub(r"https?://\S+", " ", texto, flags=re.IGNORECASE)
    texto = re.sub(r"\bt\.co/\S+", " ", texto, flags=re.IGNORECASE)
    
    palabras = re.findall(r"\b[a-záéíóúñ]+\b", texto.lower())
    return [p for p in palabras if p not in STOPWORDS and len(p) > 3]


# --- Sentimiento simple (léxico)
POSITIVAS = set([
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
"exitoso","beneficio","beneficioso","impacto","histórico"
])
            
NEGATIVAS = set([
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
    pos = sum(1 for p in palabras if p in POSITIVAS)
    neg = sum(1 for p in palabras if p in NEGATIVAS)
    if pos > neg:
        return "Positivo"
    if neg > pos:
        return "Negativo"
    return "Neutral"

def fetch_por_tipo(client, base_query, start_time, max_posts, tweet_fields_req, expansions_req, user_fields_req,
                   incluir_originales, incluir_retweets, incluir_quotes):
    """
    Devuelve:
      - tweets_data: lista combinada de tweets
      - users_by_id: dict combinado
    Estrategia:
      - Si solo 1 tipo seleccionado -> 1 llamada con filtros (ahorra cuota)
      - Si 2 o 3 tipos -> N llamadas (una por tipo seleccionado) con filtros estrictos
    """

    seleccionados = []
    if incluir_originales: seleccionados.append("ORIG")
    if incluir_retweets:   seleccionados.append("RT")
    if incluir_quotes:     seleccionados.append("QUOTE")

    if len(seleccionados) == 0:
        return [], {}

    # Reparto de cupo si hay varias llamadas (para no multiplicar cuota)
    if max_posts is None:
        limites = {t: None for t in seleccionados}  # sin límite por tipo
    else:
        per = max(10, max_posts // len(seleccionados))  # mínimo 10 por llamada
        limites = {t: per for t in seleccionados}
        # Reparto del остаток (remainder)
        rem = max_posts - per * len(seleccionados)
        i = 0
        while rem > 0:
            limites[seleccionados[i]] += 1
            rem -= 1
            i = (i + 1) % len(seleccionados)

    # Construimos queries por tipo
    q_base = f"({base_query.strip()})"

    queries = {}
    if len(seleccionados) == 1:
        # ✅ 1 sola llamada (usa build_x_query optimizado)
        q_final = build_x_query(base_query, incluir_originales, incluir_retweets, incluir_quotes)
        queries[seleccionados[0]] = q_final
    else:
        # ✅ varias llamadas, una por tipo
        if incluir_originales:
            queries["ORIG"] = f"{q_base} -is:retweet -is:quote"
        if incluir_retweets:
            queries["RT"] = f"{q_base} is:retweet -is:quote"
        if incluir_quotes:
            queries["QUOTE"] = f"{q_base} is:quote"

    # Ejecutamos llamadas y unimos resultados
    tweets_all = []
    users_all = {}

    for tipo, q in queries.items():
        tdata, udata = fetch_tweets_paginado(
            client=client,
            query=q,
            start_time=start_time,
            max_posts=limites.get(tipo),
            tweet_fields=tweet_fields_req,
            expansions=expansions_req,
            user_fields=user_fields_req
        )

        if tdata:
            tweets_all.extend(tdata)
        if udata:
            users_all.update(udata)

    # ✅ Deduplicar por tweet_id (por seguridad)
    seen = set()
    tweets_unique = []
    for t in tweets_all:
        tid = str(getattr(t, "id", ""))
        if tid and tid not in seen:
            seen.add(tid)
            tweets_unique.append(t)

    return tweets_unique, users_all

def _make_open_link(url: str) -> str:
    return f'<a href="{url}" target="_blank">Abrir</a>' if isinstance(url, str) and url else ""

def _make_open_link_reply(reply_id: str) -> str:
    rid = str(reply_id) if reply_id else ""
    url = f"https://x.com/i/web/status/{rid}" if rid else ""
    return _make_open_link(url)

def _sent_rank_for_sort(sent: str) -> int:
    # Para "Negativos primero": Negativo (0), Neutral (1), Positivo (2), N/A (3)
    if sent == "Negativo":
        return 0
    if sent == "Neutral":
        return 1
    if sent == "Positivo":
        return 2
    return 3

def render_replies_expanders_top10(
    df_top10: pd.DataFrame,
    df_replies: pd.DataFrame,
    scope: str,
    id_col: str,
    title_prefix: str = "💬 Replies"
):
    """
    UX 1: expanders por fila (solo Top 10).
    - scope: "CONV" o "AMP"
    - id_col: "tweet_id" (conv) o "original_id" (amp)
    """
    if df_top10 is None or df_top10.empty:
        return
    if df_replies is None or df_replies.empty:
        st.caption("No hay replies cargados para visualizar.")
        return

    # controles globales (aplican a todos los expanders)
    c1, c2, c3 = st.columns([1.2, 1.2, 2.6])
    with c1:
        ver_n = st.selectbox(
            "Mostrar replies por hilo",
            [10, 20, 50],
            index=0,
            key=f"replies_ver_n_{scope}"
        )
    with c2:
        orden = st.selectbox(
            "Orden",
            ["Negativos primero", "Más recientes"],
            index=0,
            key=f"replies_orden_{scope}"
        )
    with c3:
        filt_sent = st.multiselect(
            "Filtrar sentimiento",
            ["Negativo", "Neutral", "Positivo"],
            default=["Negativo", "Neutral", "Positivo"],
            key=f"replies_filtro_sent_{scope}"
        )

    st.caption("Tip: por defecto muestra Top 10 replies; puedes cambiar a 20/50. No consume cuota de X.")

    # loop por cada fila (Top 10)
    for i, row in df_top10.iterrows():
        target_id = str(row.get(id_col, "") or "")
        if not target_id:
            continue

        # filtrar replies de ese hilo/objetivo
        dfr = df_replies[(df_replies["scope"] == scope) & (df_replies["target_id"].astype(str) == target_id)].copy()

        n_total = len(dfr)
        autor = str(row.get("Autor", ""))
        fecha = row.get("Fecha") or row.get("Fechaua") or ""
        # muestra corta del texto para el título
        texto_base = str(row.get("Texto", "") or row.get("Texto_original", "") or "")
        texto_short = (texto_base[:80] + "…") if len(texto_base) > 80 else texto_base

        exp_title = f"{title_prefix} ({n_total}) — {autor} — {texto_short}"
        with st.expander(exp_title, expanded=False):
            if dfr.empty:
                st.info("Sin replies para este hilo dentro del rango.")
                continue

            # normalizar fecha
            dfr["Fecha"] = pd.to_datetime(dfr["Fecha"], errors="coerce")

            # filtrar sentimiento
            if filt_sent:
                dfr = dfr[dfr["Sentimiento"].isin(filt_sent)].copy()

            if dfr.empty:
                st.info("No hay replies con ese filtro de sentimiento.")
                continue

            # ordenar
            if orden == "Más recientes":
                dfr = dfr.sort_values("Fecha", ascending=False)
            else:
                # Negativos primero + más recientes dentro de cada grupo
                dfr["__srank"] = dfr["Sentimiento"].apply(_sent_rank_for_sort)
                dfr = dfr.sort_values(["__srank", "Fecha"], ascending=[True, False])

            # recortar N
            dfr = dfr.head(int(ver_n)).copy()

            # armar tabla simple de replies
            dfr["Link"] = dfr["reply_id"].apply(_make_open_link_reply)
            cols_rep = ["Fecha", "Sentimiento", "Likes", "Retweets", "Texto", "Link"]

            # formateo fecha amigable (sin romper NaT)
            dfr["Fecha"] = dfr["Fecha"].dt.strftime("%Y-%m-%d %H:%M:%S").fillna("")

            st.markdown(
                dfr[cols_rep].to_html(escape=False, index=False),
                unsafe_allow_html=True
            )

def _make_open_link(url: str) -> str:
    return f'<a href="{url}" target="_blank">Abrir</a>' if isinstance(url, str) and url else ""
        
def render_table(df_show: pd.DataFrame, title: str, cols: list[str], top: int | None = None):
    st.markdown(f"### {title}")
    if df_show is None or df_show.empty:
        st.info("No hay datos para mostrar en esta tabla con los filtros actuales.")
        return
        
    _df = df_show.copy()
        
    # Top N si aplica
    if isinstance(top, int) and top > 0:
        _df = _df.head(top).copy()
        
    # Link HTML "Abrir"
    if "Link" in cols:
        if "URL" in _df.columns:
            _df["Link"] = _df["URL"].apply(_make_open_link)
        elif "URL_original" in _df.columns:
            _df["Link"] = _df["URL_original"].apply(_make_open_link)
        else:
            _df["Link"] = ""
        
    # Mostrar
    st.markdown(
        _df[cols].to_html(escape=False, index=False),
        unsafe_allow_html=True
    )

def _safe_str(x):
    if x is None:
        return ""
    try:
        if isinstance(x, float) and pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x)

def _strip_html(s: str) -> str:
    # Para convertir el HTML "<a ...>Abrir</a>" a texto "Abrir: URL"
    if not isinstance(s, str) or not s:
        return ""
    # extraer href si existe
    m = re.search(r'href="([^"]+)"', s)
    if m:
        return m.group(1)
    return re.sub(r"<[^>]+>", "", s).strip()

def _df_prepare_for_pdf(df: pd.DataFrame, cols: list[str], mode: str, max_text_chars: int = 280):
    """
    Prepara DF para PDF:
    - respeta cols
    - convierte fechas
    - convierte Link/URL html a URL
    - trunca Texto para evitar tablas inleíbles
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=cols)

    d = df.copy()

    # asegurar columnas
    for c in cols:
        if c not in d.columns:
            d[c] = ""

    # Normalizar fechas típicas
    for c in ["Fecha", "Fechaua"]:
        if c in d.columns:
            d[c] = pd.to_datetime(d[c], errors="coerce")
            d[c] = d[c].dt.strftime("%Y-%m-%d %H:%M:%S").fillna("")

    # Link: asegurar que haya URL para PDF
    # - Si Link viene vacío, lo construimos desde URL o URL_original
    # - Si Link viene como HTML (<a href=...>), lo convertimos a URL limpia
    if "Link" in cols:
        # 1) Asegurar columna Link exista
        if "Link" not in d.columns:
            d["Link"] = ""
    
        # 2) Si Link está vacío, rellenar desde URL / URL_original
        def _fill_link(row):
            link_val = _safe_str(row.get("Link", "")).strip()
    
            # Si viene en HTML: extraer href
            if "<a" in link_val or "href=" in link_val:
                return _strip_html(link_val)
    
            # Si ya es URL directa, dejarla
            if link_val.startswith("http"):
                return link_val
    
            # Si está vacío: usar URL o URL_original
            url = _safe_str(row.get("URL", "")).strip()
            if url.startswith("http"):
                return url
    
            url2 = _safe_str(row.get("URL_original", "")).strip()
            if url2.startswith("http"):
                return url2
    
            return ""  # nada disponible
    
        d["Link"] = d.apply(_fill_link, axis=1)


    # Texto largo: truncar en Ejecutivo, menos truncado en Completo (igual conviene algo)
    if "Texto" in d.columns:
        lim = max_text_chars if mode == "EJECUTIVO" else max(500, max_text_chars)
        d["Texto"] = d["Texto"].apply(lambda t: (_safe_str(t)[:lim] + "…") if len(_safe_str(t)) > lim else _safe_str(t))

    if "Texto_original" in d.columns:
        lim = max_text_chars if mode == "EJECUTIVO" else max(500, max_text_chars)
        d["Texto_original"] = d["Texto_original"].apply(lambda t: (_safe_str(t)[:lim] + "…") if len(_safe_str(t)) > lim else _safe_str(t))

    return d[cols].copy()

def _plotly_to_png_bytes(fig, width=1200, height=650, scale=2):
    if fig is None:
        return None
    try:
        return fig.to_image(format="png", width=width, height=height, scale=scale)
    except Exception as e:
        # 🔎 dejar rastro para diagnóstico
        st.session_state["LAST_PNG_ERROR"] = f"{type(e).__name__}: {str(e)[:200]}"
        return None

def _make_styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="H1",
        parent=styles["Heading1"],
        fontSize=16,
        leading=18,
        spaceAfter=10
    ))
    styles.add(ParagraphStyle(
        name="H2",
        parent=styles["Heading2"],
        fontSize=12,
        leading=14,
        spaceBefore=8,
        spaceAfter=6
    ))
    styles.add(ParagraphStyle(
        name="Body",
        parent=styles["BodyText"],
        fontSize=9.5,
        leading=12,
        spaceAfter=6
    ))
    styles.add(ParagraphStyle(
        name="Small",
        parent=styles["BodyText"],
        fontSize=8.2,
        leading=10,
        spaceAfter=4
    ))
    styles.add(ParagraphStyle(
        name="CenterSmall",
        parent=styles["BodyText"],
        fontSize=8.2,
        leading=10,
        alignment=TA_CENTER,
        spaceAfter=4
    ))
    return styles

def _table_style_basic(repeat_header=True):
    ts = TableStyle([
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 7.2),         # 👈 más chico
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2E7D32")),
        ("ALIGN", (0, 0), (-1, 0), "CENTER"),       # 👈 header centrado
        ("VALIGN", (0, 0), (-1, 0), "MIDDLE"),      # 👈 vertical centrado
        ("VALIGN", (0, 1), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("FONTSIZE", (0, 1), (-1, -1), 7.8),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ])
    # zebra
    ts.add("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.lightgrey])
    return ts

def _df_to_longtable_story(df: pd.DataFrame, styles, title: str, col_widths=None):
    """
    Convierte dataframe a LongTable con encabezado repetido.
    Usa Paragraph para wrap.
    """
    story = []
    story.append(Paragraph(title, styles["H2"]))
    if df is None or df.empty:
        story.append(Paragraph("Sin datos.", styles["Body"]))
        return story

    # Header como Paragraph (wrap)
    hdr = []
    for c in df.columns:
        label = _header_label(c)
        hdr.append(Paragraph(label, styles["CenterSmall"]))
    data = [hdr]

    for _, row in df.iterrows():
        r = []
        for c in df.columns:
            txt = _safe_str(row[c])
            # links como clickable (si parece url)
            if isinstance(txt, str) and txt.startswith("http"):
                txt = f'<link href="{txt}">Abrir</link>'
            r.append(Paragraph(txt.replace("\n", "<br/>"), styles["Small"]))
        data.append(r)

    tbl = LongTable(data, repeatRows=1, colWidths=col_widths)
    tbl.setStyle(_table_style_basic())
    story.append(tbl)
    story.append(Spacer(1, 10))
    return story

def _add_png_to_story(png_bytes: bytes, styles, title: str, max_width=26*cm):
    story = []
    if not png_bytes:
        return story
    story.append(Paragraph(title, styles["H2"]))
    img = Image(io.BytesIO(png_bytes))
    # auto-scale (mantener proporción)
    iw, ih = img.imageWidth, img.imageHeight
    if iw > 0:
        scale = min(1.0, float(max_width) / float(iw))
        img.drawWidth = iw * scale
        img.drawHeight = ih * scale
    story.append(img)
    story.append(Spacer(1, 12))
    return story

def build_report_payload_from_state(mode: str):
    """
    mode: 'EJECUTIVO' | 'COMPLETO'
    Construye payload 100% desde session_state.
    """
    df_conv_rank = st.session_state.get("DF_CONV_RANK", pd.DataFrame())
    df_amp_rank  = st.session_state.get("DF_AMP_RANK", pd.DataFrame())
    df_replies   = st.session_state.get("DF_REPLIES", pd.DataFrame())

    cols_conv    = st.session_state.get("COLS_CONV", [])
    cols_top_amp = st.session_state.get("COLS_TOP_AMP", [])

    meta = {
        "query": st.session_state.get("query_final", st.session_state.get("query", "")) or st.session_state.get("query", ""),
        "time_range": st.session_state.get("time_range", ""),
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "incl_originales": st.session_state.get("incl_originales", True),
        "incl_retweets": st.session_state.get("incl_retweets", True),
        "incl_quotes": st.session_state.get("incl_quotes", True),
        "incl_replies": st.session_state.get("incl_replies", False),
    }

    # KPIs/alertas/resumen/figs: los guardaremos en state con _save_results (ver punto 3)
    kpis = st.session_state.get("REPORT_KPIS", {})
    alertas = st.session_state.get("REPORT_ALERTAS", [])
    resumen_md = st.session_state.get("REPORT_RESUMEN_MD", "")
    nota_metodo = st.session_state.get("REPORT_NOTA_METODO", "")

    # ✅ NUEVO: Traer figuras Plotly (NO PNG)
    figs_plotly = st.session_state.get("REPORT_FIGS_PLOTLY", {})  # dict de figuras plotly

    # Límites por modo
    if mode == "EJECUTIVO":
        max_rows_all = 100
        replies_per_thread = 20
    else:
        max_rows_all = None  # todo
        replies_per_thread = 50

    # Tablas
    df_conv_top10 = df_conv_rank.head(10) if not df_conv_rank.empty else pd.DataFrame()
    df_conv_all = df_conv_rank.head(max_rows_all) if (max_rows_all and not df_conv_rank.empty) else df_conv_rank

    df_amp_top10 = df_amp_rank.head(10) if not df_amp_rank.empty else pd.DataFrame()
    df_amp_all = df_amp_rank.head(max_rows_all) if (max_rows_all and not df_amp_rank.empty) else df_amp_rank

    # Replies detalle: solo para top10 hilos (para no reventar PDF)
    # Si quieres en COMPLETO: cambia 10 -> 20 en ambas líneas.
    top_threads_conv = df_conv_top10["tweet_id"].astype(str).tolist() if ("tweet_id" in df_conv_top10.columns) else []
    top_threads_amp  = df_amp_top10["original_id"].astype(str).tolist() if ("original_id" in df_amp_top10.columns) else []

    # ✅ NUEVO: metadata por hilo para imprimir “Tweet objetivo” bonito en el PDF
    threads_meta_conv = build_threads_meta(
        df_top=df_conv_top10,
        id_col="tweet_id",
        text_col="Texto",
        url_fallback_col="URL"
    )

    threads_meta_amp = build_threads_meta(
        df_top=df_amp_top10,
        id_col="original_id",
        text_col="Texto_original",
        url_fallback_col="URL_original"
    )

    return {
        "mode": mode,
        "meta": meta,
        "kpis": kpis,
        "alertas": alertas,
        "resumen_md": resumen_md,
        "nota_metodo": nota_metodo,
        "REPORT_FIGS_PLOTLY": figs_plotly,
        "tables": {
            "conv_top10": _df_prepare_for_pdf(df_conv_top10, cols_conv, mode=mode),
            "conv_all": _df_prepare_for_pdf(df_conv_all, cols_conv, mode=mode),
            "amp_top10": _df_prepare_for_pdf(df_amp_top10, cols_top_amp, mode=mode),
            "amp_all": _df_prepare_for_pdf(df_amp_all, cols_top_amp, mode=mode),
            "cols_conv": cols_conv,
            "cols_amp": cols_top_amp,
        },
        "replies": {
            "df": df_replies.copy() if df_replies is not None else pd.DataFrame(),
            "top_threads_conv": top_threads_conv,
            "top_threads_amp": top_threads_amp,
            "per_thread": replies_per_thread,
            # ✅ NUEVO
            "threads_meta_conv": threads_meta_conv,
            "threads_meta_amp": threads_meta_amp
        }
    }

def generate_pdf_report(payload: dict) -> bytes:
    styles = _make_styles()

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=landscape(A4),   # recomendado por tablas
        leftMargin=1.3*cm,
        rightMargin=1.3*cm,
        topMargin=1.2*cm,
        bottomMargin=1.0*cm
    )

    story = []

    meta = payload["meta"]
    mode = payload["mode"]

    # ── Portada
    story.append(Paragraph("Reporte de Clima Social del Tema en X (Twiter)", styles["H1"]))
    story.append(Paragraph("Análisis de conversación, amplificación y reacciones (replies)", styles["Body"]))
    story.append(Paragraph(f"<b>Modo:</b> {mode}", styles["Body"]))
    story.append(Paragraph(f"<b>Query:</b> { _safe_str(meta.get('query')) }", styles["Body"]))
    story.append(Paragraph(f"<b>Rango:</b> { _safe_str(meta.get('time_range')) }", styles["Body"]))
    story.append(Paragraph(f"<b>Generado:</b> { _safe_str(meta.get('generated_at')) }", styles["Body"]))
    story.append(Spacer(1, 5))

    filtros = []
    if meta.get("incl_originales"): filtros.append("Tweets Originales")
    if meta.get("incl_quotes"): filtros.append("RT con cita - Quotes")
    if meta.get("incl_retweets"): filtros.append("RT puros")
    if meta.get("incl_replies"): filtros.append("Replies/Comentarios")
    story.append(Paragraph(f"<b>Incluye:</b> {', '.join(filtros) if filtros else 'N/A'}", styles["Body"]))
    story.append(Spacer(1, 7))

    # ── Panel ejecutivo (KPIs)
    kpis = payload.get("kpis", {}) or {}
    if kpis:
        story.append(Paragraph("Panel ejecutivo", styles["H2"]))
        # tabla compacta KPIs (key-value)
        kv = [["Indicador", "Valor"]]
        for k, v in kpis.items():
            kv.append([Paragraph(_safe_str(k), styles["Small"]), Paragraph(_safe_str(v), styles["Small"])])
        t = Table(kv, repeatRows=1, colWidths=[10*cm, 16*cm])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#1B5E20")),
            ("TEXTCOLOR", (0,0), (-1,0), colors.white),
            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
            ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.whitesmoke, colors.lightgrey]),
            ("VALIGN", (0,0), (-1,-1), "TOP"),
            ("FONTSIZE", (0,1), (-1,-1), 8),
        ]))
        story.append(t)
        story.append(Spacer(1, 7))
    # ── Leyenda explicativa (clave para lector no experto)
    story.append(Paragraph("Leyenda de lectura", styles["H2"]))
    
    leyenda_texto = """
    <b>Conversación (conv):</b> Publicaciones que aportan contenido propio al debate
    (posts originales y retweets con cita). Refleja qué se dice y cómo se argumenta.<br/><br/>
    
    <b>Amplificación (amp):</b> Difusión de mensajes mediante retweets puros (RT).
    Cada tweet original se muestra una sola vez y los RT se agregan como volumen.
    Refleja qué mensajes se están propagando.<br/><br/>
    
    <b>Replies (comentarios):</b> Respuestas directas a un tweet dentro de un hilo.
    Reflejan reacción inmediata y descarga emocional, y pueden mostrar mayor intensidad
    que la conversación general.<br/><br/>
    
    <b>Sentimiento:</b> Clasificación automática del contenido en
    Positivo, Neutral o Negativo mediante modelos de lenguaje (IA) y reglas léxicas de respaldo.<br/><br/>
    
    <b>Temperatura:</b> Indicador sintético del clima general:
    🔴 Riesgo reputacional (negativo alto),
    🟡 Mixto / neutro,
    🟢 Clima favorable (positivo predominante).
    En replies, la temperatura solo se calcula cuando hay una muestra suficiente.
    """
    story.append(Paragraph(leyenda_texto, styles["Small"]))
    story.append(Spacer(1, 12))

    # ── Alertas
    alertas = payload.get("alertas", []) or []
    story.append(Paragraph("Alertas", styles["H2"]))
    if alertas:
        for a in alertas:
            story.append(Paragraph(_safe_str(a), styles["Body"]))
    else:
        story.append(Paragraph("Sin alertas fuertes con los umbrales actuales.", styles["Body"]))
    story.append(Spacer(1, 10))

    # ── Resumen ejecutivo (texto)
    resumen_md = payload.get("resumen_md", "") or ""
    story.append(Paragraph("Resumen ejecutivo", styles["H2"]))
    if resumen_md:
        # convertir markdown simple a párrafos (mínimo viable)
        txt = resumen_md.replace("**", "")
        for para in txt.split("\n\n"):
            story.append(Paragraph(_safe_str(para).replace("\n", "<br/>"), styles["Body"]))
    else:
        story.append(Paragraph("No disponible.", styles["Body"]))
    story.append(Spacer(1, 10))

    # ── Nota metodológica
    nota = payload.get("nota_metodo", "") or ""
    if nota:
        story.append(Paragraph("Nota metodológica", styles["H2"]))
        story.append(Paragraph(_safe_str(nota).replace("\n", "<br/>"), styles["Small"]))
        story.append(Spacer(1, 10))

    story.append(PageBreak())

    # ── Gráficos
    # ── Gráficos (desde figuras Plotly guardadas en payload)
    figs_png = {}
    error_png = ""

    # 1) Convertir Plotly -> PNG (si existe REPORT_FIGS_PLOTLY en el payload)
    try:
        figs_plotly = payload.get("REPORT_FIGS_PLOTLY", {}) or {}
        figs_png = {
            k: _plotly_to_png_bytes(v)
            for k, v in figs_plotly.items()
            if v is not None
        }
        # filtrar Nones (por si alguna conversión falló)
        figs_png = {k: v for k, v in figs_png.items() if v is not None}

    except Exception as e:
        figs_png = {}
        error_png = str(e)

    # 2) Renderizar en el PDF SOLO si tenemos PNG
    if figs_png:
        story.append(Paragraph("Tablero visual", styles["H1"]))

        for key, title in [
            ("fig_vol", "Volumen por día (Conversación vs RT puros)"),
            ("fig_sent_conv", "Sentimiento — Conversación"),
            ("fig_sent_amp", "Sentimiento — Amplificación (ponderado)"),
            ("fig_terms_conv", "Top términos — Conversación"),
            ("fig_terms_amp", "Top términos — Amplificación"),
            ("fig_rep_conv", "Replies — Conversación (sentimiento)"),
            ("fig_rep_amp", "Replies — Amplificación (sentimiento)"),
        ]:
            png = figs_png.get(key)
            story += _add_png_to_story(png, styles, title)

        story.append(PageBreak())

    else:
        # Si no pudimos exportar PNG (por ejemplo Chrome no instalado),
        # dejamos un aviso dentro del PDF en vez de “desaparecer” la sección.
        story.append(Paragraph("Tablero visual", styles["H1"]))
        story.append(Paragraph("No se pudieron incrustar gráficos en el PDF.", styles["Body"]))
        if error_png:
            story.append(Paragraph(f"Detalle técnico: { _safe_str(error_png) }", styles["Small"]))
        story.append(Spacer(1, 10))
        story.append(PageBreak())


    # ── Tablas 1–4 (CRÍTICO)
    tables = payload["tables"]

    # Anchos sugeridos (landscape A4 ~ 29.7cm ancho útil; aquí usamos 26cm aprox)
    # Ajusta si cambias columnas.
    def colwidths_for(cols):
        # default: repartir
        total_w = 26*cm
        w = total_w / max(1, len(cols))
        widths = [w]*len(cols)
        # dar más a Texto/Texto_original
        for i,c in enumerate(cols):
            if c in ("Texto", "Texto_original"):
                widths[i] = 9.0*cm
            if c in ("Link",):
                widths[i] = 2.4*cm
            if c in ("Autor",):
                widths[i] = 2.8*cm
            if c in ("Ubicación inferida","Ubicación_dominante"):
                widths[i] = 2.6*cm
            if c in ("Confianza","Confianza_dominante"):
                widths[i] = 1.7*cm
            if c in ("Interacción",):
                widths[i] = 2.2*cm
            if c in ("Sentimiento","Sentimiento_dominante","Sentimiento_replies"):
                widths[i] = 2.2*cm
            if c in ("Replies",):
                widths[i] = 1.8*cm
        # normalizar para no pasarnos
        s = sum(widths)
        if s > total_w:
            factor = total_w / s
            widths = [x*factor for x in widths]
        return widths

    story.append(Paragraph("Resultados en tablas", styles["H1"]))

    # Tabla 1
    story += _df_to_longtable_story(
        tables["conv_top10"],
        styles,
        "1) Top 10 — Conversación",
        col_widths=colwidths_for(list(tables["conv_top10"].columns))
    )

    # Tabla 2
    story += _df_to_longtable_story(
        tables["conv_all"],
        styles,
        "2) Toda la conversación",
        col_widths=colwidths_for(list(tables["conv_all"].columns))
    )

    story.append(PageBreak())

    # Tabla 3
    story += _df_to_longtable_story(
        tables["amp_top10"],
        styles,
        "3) Top 10 — Amplificación (tweet original agregado)",
        col_widths=colwidths_for(list(tables["amp_top10"].columns))
    )

    # Tabla 4
    story += _df_to_longtable_story(
        tables["amp_all"],
        styles,
        "4) Toda la amplificación (tweet original agregado)",
        col_widths=colwidths_for(list(tables["amp_all"].columns))
    )

    # ── Replies detalle (si aplica)
    rep_cfg = payload.get("replies", {})
    df_replies = rep_cfg.get("df", pd.DataFrame())
    incl_replies = meta.get("incl_replies", False)

    if incl_replies and df_replies is not None and not df_replies.empty:
        story.append(PageBreak())
        story.append(Paragraph("Detalle de replies (comentarios)", styles["H1"]))
        per_thread = int(rep_cfg.get("per_thread", 20))

        def add_replies_section(scope: str, title: str, target_ids: list[str]):
            nonlocal story
            story.append(Paragraph(title, styles["H2"]))
            # Orden sugerido: Negativos primero y recientes
            dfr = df_replies[df_replies["scope"] == scope].copy()
            if dfr.empty:
                story.append(Paragraph("Sin replies para este scope.", styles["Body"]))
                return

            dfr["Fecha"] = pd.to_datetime(dfr["Fecha"], errors="coerce")
            dfr["__srank"] = dfr["Sentimiento"].apply(_sent_rank_for_sort)
            dfr = dfr.sort_values(["target_id", "__srank", "Fecha"], ascending=[True, True, False])

            meta_conv = rep_cfg.get("threads_meta_conv", {}) or {}
            meta_amp  = rep_cfg.get("threads_meta_amp", {}) or {}
            meta_map = meta_conv if scope == "CONV" else meta_amp

            # por cada hilo objetivo
            for tid in target_ids:
                chunk = dfr[dfr["target_id"].astype(str) == str(tid)].copy()
                if chunk.empty:
                    continue                
                info = meta_map.get(str(tid), None)

                # Además del total “agregado”, calculamos cuántos replies tenemos realmente en df_replies
                total_encontrados = int(len(chunk))

                if info:
                    story.append(Paragraph(
                        f"<b>Tweet objetivo:</b> { _safe_str(info.get('autor')) } "
                        f"— “{ _safe_str(info.get('texto')) }” "
                        f"— <b>Comentarios:</b> { _safe_str(info.get('replies')) } "
                        f"(<b>encontrados en muestra:</b> {total_encontrados}) "
                        f"— <b>Link:</b> { _safe_str(info.get('url')) }",
                        styles["Body"]
                    ))
                else:
                    story.append(Paragraph(
                        f"<b>Tweet objetivo:</b> {tid} — <b>encontrados en muestra:</b> {total_encontrados}",
                        styles["Body"]
                    ))

                # recortar a N por hilo
                chunk = chunk.head(per_thread).copy()

                # construir mini tabla replies
                chunk["Link"] = chunk["reply_id"].astype(str).apply(lambda rid: f"https://x.com/i/web/status/{rid}")
                cols = ["Fecha", "Sentimiento", "Likes", "Retweets", "Texto", "Link"]
                chunk2 = chunk[cols].copy()
                chunk2["Fecha"] = pd.to_datetime(chunk2["Fecha"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S").fillna("")
                # truncado de texto
                lim = 220 if mode == "EJECUTIVO" else 500
                chunk2["Texto"] = chunk2["Texto"].apply(lambda t: (_safe_str(t)[:lim] + "…") if len(_safe_str(t)) > lim else _safe_str(t))

                story += _df_to_longtable_story(
                    chunk2,
                    styles,
                    f"Replies (top {per_thread})",
                    col_widths=[4.0*cm, 2.3*cm, 1.6*cm, 1.8*cm, 12.0*cm, 4.0*cm]
                )

        add_replies_section("CONV", "Replies — Conversación (Top hilos)", rep_cfg.get("top_threads_conv", []))
        story.append(PageBreak())
        add_replies_section("AMP", "Replies — Amplificación (Top hilos)", rep_cfg.get("top_threads_amp", []))

    # Construir PDF
    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes

def render_pdf_controls():
    st.markdown("## 📄 Descargar reporte PDF")

    modo_pdf = st.selectbox(
        "Tipo de PDF",
        ["PDF Ejecutivo (recomendado)", "PDF Completo (100% literal)"],
        index=0,
        key="sel_modo_pdf"
    )

    colp1, colp2 = st.columns([1, 3])
    with colp1:
        if st.button("Generar PDF", key="btn_gen_pdf"):
            mode = "EJECUTIVO" if "Ejecutivo" in modo_pdf else "COMPLETO"
            with st.spinner("Generando PDF..."):
                payload = build_report_payload_from_state(mode)
                pdf_bytes = generate_pdf_report(payload)

            st.session_state["LAST_PDF_BYTES"] = pdf_bytes
            st.success("PDF generado.")

    with colp2:
        pdf_bytes = st.session_state.get("LAST_PDF_BYTES", None)
        if pdf_bytes:
            filename = f"reporte_clima_x_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
            st.download_button(
                "⬇️ Descargar PDF",
                data=pdf_bytes,
                file_name=filename,
                mime="application/pdf",
                key="btn_download_pdf"
            )

def render_persisted_header_kpis_alertas_resumen():
    # 1) KPIs (los guardaste en REPORT_KPIS)
    report_kpis = st.session_state.get("REPORT_KPIS", {}) or {}
    if report_kpis:
        st.markdown("## 🧾 Panel ejecutivo (persistente)")
        # mostramos como tabla simple (no consume cuota y no depende de variables locales)
        df_k = pd.DataFrame([{"Indicador": k, "Valor": v} for k, v in report_kpis.items()])
        st.dataframe(df_k, use_container_width=True, hide_index=True)

    # 2) Alertas
    st.markdown("### 🚨 Alertas (persistente)")
    alertas = st.session_state.get("REPORT_ALERTAS", []) or []
    if alertas:
        for a in alertas:
            st.warning(a)
    else:
        st.info("Sin alertas fuertes con los umbrales actuales.")

    # 3) Resumen ejecutivo
    st.markdown("## ⭐ Resumen ejecutivo (persistente)")
    resumen_md = st.session_state.get("REPORT_RESUMEN_MD", "") or ""
    if resumen_md:
        st.markdown(resumen_md)
    else:
        st.caption("Resumen IA no disponible en el estado persistente.")

    # 4) Nota metodológica
    nota = st.session_state.get("REPORT_NOTA_METODO", "") or ""
    if nota:
        st.caption(nota)

def render_persisted_visuals():
    # Re-render de imágenes PNG guardadas (no plotly, pero sí mantiene “lo anterior” visible)
    figs = st.session_state.get("REPORT_FIGS", {}) or {}
    if not figs:
        return

    st.markdown("## 📊 Tablero visual (persistente)")
    # orden similar al PDF
    order = [
        ("fig_vol", "📈 Volumen por día (Conversación vs RT puros)"),
        ("fig_sent_conv", "🧁 Sentimiento — Conversación"),
        ("fig_sent_amp", "🧁 Sentimiento — Amplificación (ponderado)"),
        ("fig_terms_conv", "🏷️ Top términos — Conversación"),
        ("fig_terms_amp", "🏷️ Top términos — Amplificación"),
        ("fig_rep_conv", "🧁 Replies — Conversación"),
        ("fig_rep_amp", "🧁 Replies — Amplificación"),
    ]
    for k, title in order:
        if k in figs:
            st.markdown(f"#### {title}")
            st.image(figs[k], use_container_width=True)

def render_persisted_tables_and_replies():
    df_conv_rank = st.session_state.get("DF_CONV_RANK", pd.DataFrame())
    df_amp_rank  = st.session_state.get("DF_AMP_RANK", pd.DataFrame())
    df_replies   = st.session_state.get("DF_REPLIES", pd.DataFrame())
    cols_conv    = st.session_state.get("COLS_CONV", [])
    cols_top_amp = st.session_state.get("COLS_TOP_AMP", [])
    incl_replies = st.session_state.get("incl_replies", False)

    st.markdown("## 📌 Resultados en tablas (persistente)")

    if df_conv_rank is not None and not df_conv_rank.empty:
        render_table(df_conv_rank, "1) 🔥 Top 10 — Conversación", cols=cols_conv, top=10)

        if incl_replies and (df_replies is not None) and (not df_replies.empty):
            st.markdown("#### 💬 Leer replies — TOP 10 (Conversación)")
            df_conv_top10 = df_conv_rank.head(10).copy()
            render_replies_expanders_top10(
                df_top10=df_conv_top10,
                df_replies=df_replies,
                scope="CONV",
                id_col="tweet_id",
                title_prefix="💬 Replies (conv)"
            )
    else:
        st.info("Sin resultados de conversación para mostrar.")

    if df_amp_rank is not None and not df_amp_rank.empty:
        render_table(
            df_amp_rank,
            "3) 📣 Top 10 — Amplificación (muestra el tweet ORIGINAL amplificado)",
            cols=cols_top_amp,
            top=10
        )

        if incl_replies and (df_replies is not None) and (not df_replies.empty):
            st.markdown("#### 💬 Leer replies — TOP 10 (Amplificación)")
            df_amp_top10 = df_amp_rank.head(10).copy()
            render_replies_expanders_top10(
                df_top10=df_amp_top10,
                df_replies=df_replies,
                scope="AMP",
                id_col="original_id",
                title_prefix="💬 Replies (amp)"
            )
    else:
        st.info("Sin resultados de amplificación para mostrar.")

def build_threads_meta(df_top: pd.DataFrame, id_col: str, text_col: str, url_fallback_col: str = "URL") -> dict:
    """
    Devuelve dict:
      target_id -> {"autor":..., "texto":..., "replies": int, "url": ...}
    Usa Link/URL/URL_original si existen.
    """
    meta = {}
    if df_top is None or df_top.empty or id_col not in df_top.columns:
        return meta

    d = df_top.copy()
    d[id_col] = d[id_col].astype(str)

    # Asegurar columnas
    if "Autor" not in d.columns: d["Autor"] = ""
    if text_col not in d.columns: d[text_col] = ""
    if "Replies" not in d.columns: d["Replies"] = 0
    if "Link" not in d.columns: d["Link"] = ""

    for _, row in d.iterrows():
        tid = str(row.get(id_col, "") or "").strip()
        if not tid:
            continue

        autor = _safe_str(row.get("Autor", "")).strip()

        texto = _safe_str(row.get(text_col, "")).strip()
        texto_short = (texto[:180] + "…") if len(texto) > 180 else texto

        # Replies: total agregado (no el recorte por per_thread)
        try:
            replies_total = int(pd.to_numeric(row.get("Replies", 0), errors="coerce") or 0)
        except Exception:
            replies_total = 0

        # URL: preferir Link (si es HTML o URL), luego URL/URL_original
        link_val = _safe_str(row.get("Link", "")).strip()
        url = ""
        if link_val:
            # si Link viene como HTML <a href=...> -> extraer
            url = _strip_html(link_val) if ("<a" in link_val or "href=" in link_val) else link_val

        if not url:
            # probar URL / URL_original / fallback
            for c in ["URL", "URL_original", url_fallback_col]:
                v = _safe_str(row.get(c, "")).strip()
                if v.startswith("http"):
                    url = v
                    break

        meta[tid] = {
            "autor": autor or "N/A",
            "texto": texto_short or "N/A",
            "replies": replies_total,
            "url": url or f"https://x.com/i/web/status/{tid}"
        }

    return meta

def _header_label(col: str) -> str:
    """
    Devuelve un header con saltos de línea para evitar superposición.
    Ajusta aquí los nombres “largos” que se ven feos en PDF.
    """
    mapping = {
        "Interacción": "Interac<br/>ción",
        "Sentimiento": "Sentim<br/>iento",
        "Replies": "Repl<br/>ies",
        "Sentimiento_replies": "Sentim<br/>replies",
        "Ubicación inferida": "Ubicación<br/>inferida",
        "Ubicación_dominante": "Ubicación<br/>dominante",
        "Confianza": "Conf<br/>ianza",
        "Confianza_dominante": "Conf<br/>dominante",
        "Texto": "Texto",  # aquí no hacemos split porque ya tiene mucho ancho
        "Texto_original": "Texto<br/>original",
        "RT_puros_en_rango": "RT<br/>puros",
        "Ampl_total": "Ampl<br/>total",
        "Fechaua": "Fecha<br/>últ. ampl.",
        "Likesta": "Likes<br/>total",
        "Link": "Link",
        "Autor": "Autor",
    }
    return mapping.get(col, col)


# ─────────────────────────────
# Botones alineados a la izquierda (tamaño normal)
# ─────────────────────────────
col_buscar, col_limpiar, col_spacer = st.columns([2, 2, 8])

with col_buscar:
    clicked_buscar = st.button("🔍 Buscar en X")

with col_limpiar:
    clicked_limpiar = st.button("🧹 Limpiar resultados")

# ─────────────────────────────
# Acciones asociadas a los botones
# ─────────────────────────────
if clicked_limpiar:
    _clear_results()
    st.rerun()

if clicked_buscar:
    now = time.time()
    if now - st.session_state["last_search_ts"] < 20:
        st.warning("Espera 20 segundos entre búsquedas para evitar límites de X.")
        st.stop()
    st.session_state["last_search_ts"] = now

    if not query:
        st.warning("Ingresa una palabra clave")
    else:
        start_time = get_start_time(time_range).isoformat("T") + "Z"

        # ─────────────────────────────
        # Presupuesto de cuota (límite) incluyendo Replies
        # ─────────────────────────────
        incl_replies = st.session_state.get("incl_replies", False)
        
        # Presupuesto base = max_posts (si no hay "Sin límite")
        max_posts_base = max_posts
        replies_budget = None
        
        # Si replies está activo y hay límite numérico, repartimos presupuesto
        if incl_replies and (max_posts is not None):
            # 60% posts base / 40% replies
            max_posts_base = max(50, int(round(max_posts * 0.60)))
            replies_budget = max_posts - max_posts_base
        
            # Ajustar targets y replies por tweet para no reventar
            # OJO: en este punto aún no sabemos cuántos targets reales habrá,
            # así que hacemos una cota conservadora por defecto:
            #  - Objetivos máx = TOP_CONV + TOP_AMP
            max_objetivos = TOP_TWEETS_CONV_REPLIES + TOP_TWEETS_AMP_REPLIES
            if max_objetivos <= 0:
                max_objetivos = 1
        
            # Replies por tweet objetivo (mínimo 10)
            max_replies_dyn = max(10, int(replies_budget // max_objetivos))
        
            # Cap superior para no abusar (tú puedes ajustar 50/100)
            max_replies_dyn = min(max_replies_dyn, 50)
        
            # Aplicar dinámico
            MAX_REPLIES_POR_TWEET = max_replies_dyn
        
            st.caption(
                f"Presupuesto por límite={max_posts}. "
                f"Posts base={max_posts_base}, Replies total≈{replies_budget}. "
                f"MAX_REPLIES_POR_TWEET={MAX_REPLIES_POR_TWEET} "
                f"(objetivos máx={max_objetivos})."
            )
        else:
            # sin replies o sin límite, se mantiene tu configuración
            max_posts_base = max_posts

        # Pedimos también info del autor vía expansions
        try:             
            # =========================
            # PARTE 2 — Ajuste de consulta a X (1 sola llamada) + campos para diferenciar Original/RT/Quote
            # =========================
            
            # Recuperamos selección (por si el usuario cambió checks)
            incl_originales = st.session_state.get("incl_originales", True)
            incl_retweets = st.session_state.get("incl_retweets", True)
            incl_quotes = st.session_state.get("incl_quotes", True)
            
            query_final = st.session_state.get("query_final", query)
            
            # 🚩 Importante:
            # - Traemos referenced_tweets y conversation_id para clasificar.
            # - Incluimos expansion referenced_tweets.id para que X devuelva el tweet original en includes si está disponible.
            tweet_fields_req = [
                "created_at",
                "public_metrics",
                "author_id",
                "referenced_tweets",
                "conversation_id",
                "lang"
            ]
            
            expansions_req = [
                "author_id",
                "referenced_tweets.id",
                "in_reply_to_user_id"
            ]
            
            user_fields_req = ["username", "name", "location", "description"]
            
            # ✅ Llamadas inteligentes (1 o varias según checks)
            base_query = query.strip()
            tweets_data, users_by_id = fetch_por_tipo(
                client=client,
                base_query=base_query,
                start_time=start_time,
                max_posts=max_posts_base,
                tweet_fields_req=tweet_fields_req,
                expansions_req=expansions_req,
                user_fields_req=user_fields_req,
                incluir_originales=incl_originales,
                incluir_retweets=incl_retweets,
                incluir_quotes=incl_quotes
            )
             
            # Guardamos en session_state por si luego quieres exportar / depurar
            st.session_state["tweets_data_count"] = len(tweets_data) if tweets_data else 0
        
        except tweepy.errors.TooManyRequests as e:
            # Intentar leer "reset time" si existe
            reset_info = ""
            try:
                reset_ts = int(e.response.headers.get("x-rate-limit-reset", "0"))
                if reset_ts:
                    wait_sec = max(0, reset_ts - int(time.time()))
                    wait_min = max(1, int(round(wait_sec / 60)))
                    reset_info = f"⏳ Intenta nuevamente en ~{wait_min} min."
            except Exception:
                pass
        
            st.error(
                "⚠️ Límite de consultas alcanzado en la API de X (rate limit).\n\n"
                "Esto ocurre cuando se hacen varias búsquedas en poco tiempo (por el mismo token o porque la app es pública). "
                + reset_info
            )
            st.stop()
        except Exception as e:
            st.error(f"⚠️ Error inesperado al consultar X: {type(e).__name__}")
            st.stop()
        
        # =========================
        # PARTE 3 — Armar df_raw + clasificar Original / RT puro / Quote + separar en 3 dataframes
        # =========================
        # ✅ DÓNDE PEGAR:
        # Pega este bloque JUSTO DESPUÉS de la PARTE 2 (después de obtener tweets_data, users_by_id)
        # y ANTES de tu bloque actual que arma "data = []" y "df = pd.DataFrame(data)".
        
        # Recuperamos selección (checks)
        incl_originales = st.session_state.get("incl_originales", True)
        incl_retweets = st.session_state.get("incl_retweets", True)
        incl_quotes = st.session_state.get("incl_quotes", True)
        
        # 1) Armamos un diccionario de "tweets incluidos" (cuando expansions trae referenced_tweets.id)
        #    Esto permite obtener texto del tweet original si X lo incluyó en includes.
        includes_tweets_by_id = {}
        try:
            # En Tweepy v2, el objeto Response puede traer resp.includes; aquí solo tenemos tweets_data y users_by_id.
            # fetch_tweets_paginado no devuelve includes de tweets, así que NO los tenemos aún.
            # 👉 Solución: en PARTE 3 trabajamos SIN includes de tweets (robusto).
            # (Si luego quieres, ajustamos fetch_tweets_paginado para que también devuelva resp.includes["tweets"].)
            pass
        except Exception:
            pass
        
        def clasificar_tipo_y_original_id(tweet_obj) -> tuple[str, str | None]:
            """
            Devuelve (tipo, original_id)
            tipo ∈ {"Original", "RT", "Quote"}
            original_id: id del tweet original al que referencia (si aplica)
            """
            refs = getattr(tweet_obj, "referenced_tweets", None)
        
            # Sin referencias -> es original (o respuesta sin ref; igual lo tratamos como "Original" para MVP)
            if not refs:
                return "Original", None
        
            # referenced_tweets suele ser lista de dict/obj con .type y .id
            for r in refs:
                r_type = getattr(r, "type", None) or (r.get("type") if isinstance(r, dict) else None)
                r_id = getattr(r, "id", None) or (r.get("id") if isinstance(r, dict) else None)
        
                if r_type == "retweeted":
                    return "RT", str(r_id) if r_id else None
                if r_type == "quoted":
                    return "Quote", str(r_id) if r_id else None
        
            # Si viene otra referencia (replied_to), lo dejamos como "Original" para no romper flujo MVP
            return "Original", None
        
        def extraer_username_y_url(tweet_id: str, user_obj) -> tuple[str | None, str]:
            username = getattr(user_obj, "username", None) if user_obj else None
            url = f"https://x.com/{username}/status/{tweet_id}" if username else ""
            return username, url
        
        # 2) Convertimos tweets_data a df_raw con campos mínimos + clasificación
        rows = []
        for t in (tweets_data or []):
            u = users_by_id.get(t.author_id)
        
            username, tweet_url = extraer_username_y_url(str(t.id), u)
        
            name = getattr(u, "name", None) if u else None
            profile_location = getattr(u, "location", None) if u else None
            profile_desc = getattr(u, "description", None) if u else None
        
            ubicacion, confianza, fuente = infer_peru_location(profile_location, profile_desc)
        
            tipo, original_id = clasificar_tipo_y_original_id(t)
        
            rows.append({
                "tweet_id": str(t.id),
                "conversation_id": str(getattr(t, "conversation_id", "")) if getattr(t, "conversation_id", None) else str(t.id),
                "original_id": str(original_id) if original_id else None,   # si es RT/Quote -> id del tweet original
                "tipo": tipo,                                               # Original / RT / Quote
                "Autor": f"@{username}" if username else (name or "Desconocido"),
                "URL": tweet_url,
                "Texto": getattr(t, "text", ""),
                "Fecha": getattr(t, "created_at", None),
                "Likes": (getattr(t, "public_metrics", None) or {}).get("like_count", 0),
                "Retweets": (getattr(t, "public_metrics", None) or {}).get("retweet_count", 0),
                "Ubicación inferida": ubicacion,
                "Confianza": confianza,
                "Fuente ubic.": fuente
            })
        
        df_raw = pd.DataFrame(rows)
        
        if df_raw.empty:
            st.warning("No se encontraron publicaciones para ese criterio o rango seleccionado.")
            st.stop()

        df_raw["Fecha"] = pd.to_datetime(df_raw["Fecha"], errors="coerce")
        df_raw["Likes"] = pd.to_numeric(df_raw["Likes"], errors="coerce").fillna(0)
        df_raw["Retweets"] = pd.to_numeric(df_raw["Retweets"], errors="coerce").fillna(0)
        df_raw["Interacción"] = df_raw["Likes"] + df_raw["Retweets"]
        
        # 3) Filtramos según los checks del usuario (sin hacer nueva consulta)
        tipos_permitidos = set()
        if incl_originales:
            tipos_permitidos.add("Original")
        if incl_retweets:
            tipos_permitidos.add("RT")
        if incl_quotes:
            tipos_permitidos.add("Quote")
        
        df_raw = df_raw[df_raw["tipo"].isin(tipos_permitidos)].copy()
        
        if df_raw.empty:
            st.warning("Con los filtros seleccionados (Original/RT/Quote) no hay resultados en el rango.")
            st.stop()
        
        # 4) Separamos en 3 dfs base
        df_originales = df_raw[df_raw["tipo"] == "Original"].copy()
        df_rt_puros   = df_raw[df_raw["tipo"] == "RT"].copy()
        df_quotes     = df_raw[df_raw["tipo"] == "Quote"].copy()

        # Conversación vista según checks:
        df_conversacion_view = pd.DataFrame()
        
        if incl_originales and incl_quotes:
            df_conversacion_view = pd.concat([df_originales, df_quotes], ignore_index=True)
        elif incl_originales and (not incl_quotes):
            df_conversacion_view = df_originales.copy()
        elif (not incl_originales) and incl_quotes:
            df_conversacion_view = df_quotes.copy()
        else:
            # si no está originales ni quotes, entonces conversación no aplica
            df_conversacion_view = pd.DataFrame()

        # Tip: para depurar rápido
        st.session_state["df_raw_rows"] = int(len(df_raw))
        st.session_state["df_originales_rows"] = int(len(df_originales))
        st.session_state["df_rt_puros_rows"] = int(len(df_rt_puros))
        st.session_state["df_quotes_rows"] = int(len(df_quotes))
        
        # 5) Normalizamos tipos básicos (fechas y métricas)
        for _df in [df_originales, df_rt_puros, df_quotes]:
            if _df.empty:
                continue
            _df["Fecha"] = pd.to_datetime(_df["Fecha"], errors="coerce")
            _df["Likes"] = pd.to_numeric(_df["Likes"], errors="coerce").fillna(0)
            _df["Retweets"] = pd.to_numeric(_df["Retweets"], errors="coerce").fillna(0)
            _df["Interacción"] = _df["Likes"] + _df["Retweets"]
        
        # ✅ A partir de aquí ya NO uses la variable "df" antigua.
        # Ahora trabajarás con:
        # - df_originales (conversación base)
        # - df_quotes (conversación + amplificación, porque trae comentario)
        # - df_rt_puros (amplificación pura; NO lo usaremos para sentimiento por fila en la PARTE 4)


        st.markdown("## 🧠 ANALISIS Y RESULTADOS")

        # =========================
        # PARTE 4 — Sentimiento “sin inflar” + df_conversacion + base para df_amplificacion
        # =========================
        # ✅ DÓNDE PEGAR:
        # Pega este bloque JUSTO DESPUÉS de tu:
        #   st.markdown("## 🧠 ANALISIS Y RESULTADOS")
        # y ANTES de cualquier lógica vieja que use "df" (ya NO usamos df).
            
        # ---------------------------------------------------------
        # 4.1) Definir “conversación”:
        # - Conversación incluye: originales + quotes (porque quotes sí aportan comentario nuevo)
        # - RT puros NO entran a conversación (son amplificación pura y repiten texto)
        # ---------------------------------------------------------
 
        df_conversacion = df_conversacion_view.copy()

        hay_conversacion = not df_conversacion.empty
        hay_rt_puros = not df_rt_puros.empty
        
        
        if (not hay_conversacion) and (not hay_rt_puros):
            st.warning("No se encontraron publicaciones para los filtros y rango seleccionados.")
            st.stop()
        
        if not hay_conversacion:
            st.info("No hay 'conversación' (originales + quotes) en el rango seleccionado. Se mostrará solo Amplificación (RT puros).")
        
        # ---------------------------------------------------------
        # 4.2) Sentimiento por fila SOLO en conversación (originales + quotes)
        #     (acá sí tiene sentido por fila porque el texto cambia)
        # ---------------------------------------------------------
        if hay_conversacion:
            sent_hf_conv = []
            score_hf_conv = []
            
            for txt in df_conversacion["Texto"].tolist():
                s, sc = sentimiento_hf(txt)
                sent_hf_conv.append(s)
                score_hf_conv.append(sc)
            
            df_conversacion["Sentimiento_HF"] = sent_hf_conv
            df_conversacion["Score_HF"] = score_hf_conv
            df_conversacion["Sentimiento_Lex"] = df_conversacion["Texto"].apply(calcular_sentimiento)
            df_conversacion["Sentimiento"] = df_conversacion["Sentimiento_HF"].fillna(df_conversacion["Sentimiento_Lex"])
            
            metodo_sent_conv = "IA (Hugging Face)" if df_conversacion["Sentimiento_HF"].notna().any() else "Léxico (fallback)"
        else:
            metodo_sent_conv = "N/A (sin conversación)"
        
        # ---------------------------------------------------------
        # 4.3) Sentimiento para RT puros:
        #     ✅ “1 sola vez por tweet original” (no por cada RT)
        #     - Agrupamos RT puros por original_id
        #     - Para cada original_id, calculamos sentimiento UNA sola vez usando texto del original (si lo tenemos)
        #       Si no lo tenemos, usamos el texto del primer RT (suele ser idéntico en RT puros)
        # ---------------------------------------------------------
        def sentimiento_unico_para_texto(texto: str):
            s, sc = sentimiento_hf(texto)
            if s is None:
                s = calcular_sentimiento(texto)
                sc = None
            return s, sc
        
        # Mapa id -> texto del tweet original (solo si el original está dentro del rango y lo capturamos)
        texto_por_tweet_id = {}
        if not df_originales.empty:
            # En originales, tweet_id es su propio id
            for _id, _txt in zip(df_originales["tweet_id"].tolist(), df_originales["Texto"].tolist()):
                if _id and isinstance(_txt, str) and _txt.strip():
                    texto_por_tweet_id[str(_id)] = _txt
        
        # Construimos df_rt_agregado: una fila por original_id (aunque haya 500 RT)
        df_rt_agregado = pd.DataFrame()
        if not df_rt_puros.empty:
            tmp = df_rt_puros.copy()
            tmp = tmp[tmp["original_id"].notna()].copy()
        
            if not tmp.empty:
                # Sentimiento 1 sola vez por original_id
                registros = []
                for original_id, g in tmp.groupby("original_id"):
                    # Elegimos texto para “ese original”
                    texto_base = texto_por_tweet_id.get(str(original_id))
                    if not texto_base:
                        # fallback: el texto del primer RT puro
                        texto_base = str(g.iloc[0].get("Texto", ""))
        
                    s_uni, sc_uni = sentimiento_unico_para_texto(texto_base)
        
                    registros.append({
                        "original_id": str(original_id),
                        "Texto_base_original": texto_base,
                        "Sentimiento_original": s_uni,
                        "Score_original": sc_uni
                    })
        
                df_rt_sent = pd.DataFrame(registros)
                df_rt_agregado = (
                    tmp.groupby("original_id")
                       .agg(
                           RT_puros_en_rango=("tweet_id", "count"),
                           Likes_total_amplificacion=("Likes", "sum"),
                           Retweets_total_amplificacion=("Retweets", "sum"),
                           Fecha_ultima_amplificacion=("Fecha", "max"),
                       )
                       .reset_index()
                )
        
                df_rt_agregado["original_id"] = df_rt_agregado["original_id"].astype(str)
                df_rt_agregado = df_rt_agregado.merge(df_rt_sent, on="original_id", how="left")
           
        # ---------------------------------------------------------
        # 4.4) Agregar QUOTES como amplificación (quotes también son conversación)
        # ---------------------------------------------------------
        # ✅ IMPORTANTE: aunque quede vacío, debe tener columna "original_id" para no romper merges
        df_quotes_agregado = pd.DataFrame(columns=[
            "original_id", "Quotes_en_rango", "Likes_total_quotes", "Retweets_total_quotes", "Fecha_ultima_quote"
        ])
        
        if not df_quotes.empty:
            qtmp = df_quotes.copy()
            qtmp = qtmp[qtmp["original_id"].notna()].copy()
        
            if not qtmp.empty:
                df_quotes_agregado = (
                    qtmp.groupby("original_id")
                        .agg(
                            Quotes_en_rango=("tweet_id", "count"),
                            Likes_total_quotes=("Likes", "sum"),
                            Retweets_total_quotes=("Retweets", "sum"),
                            Fecha_ultima_quote=("Fecha", "max"),
                        )
                        .reset_index()
                )
                df_quotes_agregado["original_id"] = df_quotes_agregado["original_id"].astype(str)

        # ---------------------------------------------------------
        # 4.5) Construir df_amplificacion (una sola tabla, 1 fila por tweet original)        
        #     Construir df_amplificacion SOLO con RT puros
        #     (Quotes NO entran aquí; ya están en Conversación)
        # ---------------------------------------------------------
        df_amplificacion = pd.DataFrame()
        
        if incl_retweets and (df_rt_agregado is not None) and (not df_rt_agregado.empty):
        
            base = df_rt_agregado.copy()
            base["original_id"] = base["original_id"].astype(str)
        
            # Amplificación total = SOLO RT puros
            base["Ampl_total"] = pd.to_numeric(base["RT_puros_en_rango"], errors="coerce").fillna(0)
        
            # Fecha última amplificación
            base["Fecha_ultima_amplificacion"] = pd.to_datetime(base.get("Fecha_ultima_amplificacion"), errors="coerce", utc=True)
            base["Fechaua"] = base["Fecha_ultima_amplificacion"].dt.tz_convert(None)
        
            # Likes totales amplificación (RT puros)
            base["Likesta"] = pd.to_numeric(base.get("Likes_total_amplificacion"), errors="coerce").fillna(0)
        
            # Retweets totales amplificación (RT puros)
            base["Retweets"] = pd.to_numeric(base.get("Retweets_total_amplificacion"), errors="coerce").fillna(0)
        
            # ---------------------------
            # Sentimiento dominante = SOLO RT puros (ya viene por original)
            # ---------------------------
            base["Sentimiento_dominante"] = base.get("Sentimiento_original")
            base["Score_sent_dominante"] = base.get("Score_original")
        
            # ---------------------------
            # Ubicación/Confianza dominante SOLO desde RT puros
            # ---------------------------
            def modo_safe(series):
                if series is None or len(series) == 0:
                    return None
                s = series.dropna()
                if s.empty:
                    return None
                try:
                    return s.mode().iloc[0]
                except Exception:
                    return s.iloc[0]
        
            if not df_raw.empty:
                amp_rows = df_raw[(df_raw["tipo"] == "RT") & df_raw["original_id"].notna()].copy()
        
                ubis = []
                confs = []
                for oid in base["original_id"].astype(str).tolist():
                    g = amp_rows[amp_rows["original_id"].astype(str) == str(oid)]
                    ubis.append(modo_safe(g["Ubicación inferida"]) if not g.empty else None)
                    confs.append(modo_safe(g["Confianza"]) if not g.empty else None)
        
                base["Ubicación_dominante"] = ubis
                base["Confianza_dominante"] = confs
        
            # URL original (si no lo tenemos, fallback)
            url_por_original_id = {}
            if not df_originales.empty:
                for _id, _url in zip(df_originales["tweet_id"].tolist(), df_originales["URL"].tolist()):
                    if _id:
                        url_por_original_id[str(_id)] = _url
        
            base["URL_original"] = base["original_id"].astype(str).apply(
                lambda oid: url_por_original_id.get(oid, f"https://x.com/i/web/status/{oid}")
            )
                
            # Texto del original (fallback): si no podemos traer el original real, usamos el texto base del RT
            base["Texto_original"] = base.get("Texto_base_original", "").fillna("").astype(str)
            
            # --- Completar datos del tweet ORIGINAL real (autor, url, texto) ---
            original_ids_list = base["original_id"].dropna().astype(str).unique().tolist()
            originals_info = fetch_originals_by_ids(client, original_ids_list)
            
            # Autor original real
            base["Autor"] = base["original_id"].astype(str).apply(
                lambda oid: originals_info.get(oid, {}).get("autor", "Desconocido")
            )
            
            # URL original real (mejor que el fallback i/web)
            base["URL_original"] = base["original_id"].astype(str).apply(
                lambda oid: originals_info.get(oid, {}).get("url", f"https://x.com/i/web/status/{oid}")
            )
            
            # Texto original real (si se pudo traer, reemplaza el fallback)
            base["Texto_original_real"] = base["original_id"].astype(str).apply(
                lambda oid: originals_info.get(oid, {}).get("texto", "")
            )
            
            base["Texto_original"] = base["Texto_original_real"].where(
                base["Texto_original_real"].astype(str).str.len() > 0,
                other=base["Texto_original"]
            )
            
            # limpiamos auxiliar
            base.drop(columns=["Texto_original_real"], inplace=True, errors="ignore")

        
            # ✅ muy importante: solo dejamos filas con RT>0
            base = base[base["Ampl_total"] > 0].copy()
        
            df_amplificacion = base.copy()

        # ---------------------------------------------------------
        # 4.6) Mensajes de control (para neófitos)
        # ---------------------------------------------------------
        
        st.caption(
            f"Método de sentimiento (conversación): {metodo_sent_conv}. "
            f"En amplificación: dominante por (RT_puros)."
        )
        
        # Guardamos dfs clave para PARTE 5/6 (KPIs + tablas + gráficos)
        st.session_state["df_conversacion_rows"] = int(len(df_conversacion))
        st.session_state["df_amplificacion_rows"] = int(len(df_amplificacion)) if df_amplificacion is not None else 0

        # =========================
        # PARTE 4.7 — Replies (comentarios) por Conversación y Amplificación (opcional)
        # =========================
        incl_replies = st.session_state.get("incl_replies", False)
        
        df_replies = pd.DataFrame()
        df_replies_agg = pd.DataFrame()      # agregado por objetivo (tweet objetivo)
        df_replies_conv_agg = pd.DataFrame() # agregado para conversación
        df_replies_amp_agg = pd.DataFrame()  # agregado para amplificación (por original_id)
        
        if incl_replies:
            st.markdown("### 💬 Replies (comentarios) — activado")
        
            # 1) Targets Conversación: top por Interacción (tweets de df_conversacion)
            conv_targets = []
            if df_conversacion is not None and not df_conversacion.empty:
                df_conv_targets = df_conversacion.copy()
                if "Interacción" not in df_conv_targets.columns:
                    df_conv_targets["Interacción"] = (
                        pd.to_numeric(df_conv_targets.get("Likes", 0), errors="coerce").fillna(0) +
                        pd.to_numeric(df_conv_targets.get("Retweets", 0), errors="coerce").fillna(0)
                    )

                if "conversation_id" not in df_conv_targets.columns:
                    df_conv_targets["conversation_id"] = df_conv_targets["tweet_id"].astype(str)

                df_conv_targets = df_conv_targets.sort_values("Interacción", ascending=False).head(TOP_TWEETS_CONV_REPLIES)
                # para conv, el objetivo es el propio tweet (tweet_id) y conversation_id ya viene
                conv_targets = df_conv_targets[["tweet_id", "conversation_id"]].dropna().astype(str).to_dict("records")
        
            # 2) Targets Amplificación: top por Ampl_total (tweets originales amplificados)
            amp_targets = []
            if incl_retweets and df_amplificacion is not None and not df_amplificacion.empty:
                # Para amplificación, debemos asegurar conversation_id del ORIGINAL
                # Ya trajimos originals_info en 4.5; lo reutilizamos:
                try:
                    # Si originals_info no existe por scope, lo reconstruimos rápido:
                    if "originals_info" not in locals():
                        original_ids_list = df_amplificacion["original_id"].dropna().astype(str).unique().tolist()
                        originals_info = fetch_originals_by_ids(client, original_ids_list)
                except Exception:
                    originals_info = {}
        
                df_amp_targets = df_amplificacion.sort_values("Ampl_total", ascending=False).head(TOP_TWEETS_AMP_REPLIES).copy()
                df_amp_targets["conversation_id"] = df_amp_targets["original_id"].astype(str).apply(
                    lambda oid: originals_info.get(str(oid), {}).get("conversation_id", None)
                )
                # objetivo = original_id (porque replies se miden sobre el original)
                amp_targets = df_amp_targets[["original_id", "conversation_id"]].dropna().astype(str).to_dict("records")
        
            # 3) Fetch replies por cada conversation_id objetivo
            reply_rows = []
            total_objetivos = len(conv_targets) + len(amp_targets)
        
            if total_objetivos == 0:
                st.info("Replies activado, pero no hay tweets objetivo (conversación/amplificación) para buscar comentarios.")
            else:
                st.caption(
                    f"Buscando replies en hasta {len(conv_targets)} tweet(s) de conversación y {len(amp_targets)} tweet(s) amplificados. "
                    f"(máx {MAX_REPLIES_POR_TWEET} replies por tweet objetivo)"
                )

                fallos_replies = 0
                saltados_por_conv_id_invalido = 0

                # --- Conversación replies ---
                for item in conv_targets:
                    target_tweet_id = str(item.get("tweet_id", "")).strip()
                    conv_id = str(item.get("conversation_id", "")).strip()
                
                    # ✅ Blindaje: si conv_id no es numérico, saltamos
                    if (not conv_id) or (not conv_id.isdigit()):
                        saltados_por_conv_id_invalido += 1
                        continue
                
                    replies_list = fetch_replies_for_conversation_id(
                        client=client,
                        conversation_id=conv_id,
                        start_time=start_time,
                        max_replies=MAX_REPLIES_POR_TWEET
                    )
                
                    # ✅ Si por algún motivo falla y retorna vacío, solo continúa
                    if replies_list is None:
                        fallos_replies += 1
                        continue
                
                    for r in replies_list:
                        reply_rows.append({
                            "scope": "CONV",
                            "target_id": target_tweet_id,
                            "conversation_id": conv_id,
                            "reply_id": str(getattr(r, "id", "")),
                            "Texto": getattr(r, "text", "") or "",
                            "Fecha": getattr(r, "created_at", None),
                            "Likes": (getattr(r, "public_metrics", None) or {}).get("like_count", 0),
                            "Retweets": (getattr(r, "public_metrics", None) or {}).get("retweet_count", 0),
                        })
        
                # --- Amplificación replies (por original) ---
                for item in amp_targets:
                    original_id = str(item.get("original_id", "")).strip()
                    conv_id = str(item.get("conversation_id", "")).strip()
                
                    if (not conv_id) or (not conv_id.isdigit()):
                        saltados_por_conv_id_invalido += 1
                        continue
                
                    replies_list = fetch_replies_for_conversation_id(
                        client=client,
                        conversation_id=conv_id,
                        start_time=start_time,
                        max_replies=MAX_REPLIES_POR_TWEET
                    )

                    # ⏸️ PAUSA SUAVE para no saturar la API de X
                    time.sleep(0.4)

                    if replies_list is None:
                        fallos_replies += 1
                        continue
                
                    for r in replies_list:
                        reply_rows.append({
                            "scope": "AMP",
                            "target_id": original_id,
                            "conversation_id": conv_id,
                            "reply_id": str(getattr(r, "id", "")),
                            "Texto": getattr(r, "text", "") or "",
                            "Fecha": getattr(r, "created_at", None),
                            "Likes": (getattr(r, "public_metrics", None) or {}).get("like_count", 0),
                            "Retweets": (getattr(r, "public_metrics", None) or {}).get("retweet_count", 0),
                        })
        
                df_replies = pd.DataFrame(reply_rows)

                st.caption(
                    f"Replies: objetivos={total_objetivos}, "
                    f"saltados_por_conv_id_invalido={saltados_por_conv_id_invalido}, "
                    f"fallos={fallos_replies}"
                )

                if df_replies.empty:
                    st.info("No se encontraron replies dentro del rango temporal seleccionado (o la API no devolvió resultados).")
                else:
                    # 4) Sentimiento por reply (IA o fallback léxico)
                    sent_list = []
                    for txt in df_replies["Texto"].tolist():
                        s, _ = sentimiento_hf(txt)
                        if s is None:
                            s = calcular_sentimiento(txt)
                        sent_list.append(s)
        
                    df_replies["Sentimiento"] = sent_list
                    df_replies["Fecha"] = pd.to_datetime(df_replies["Fecha"], errors="coerce")
        
                    # 5) Agregado por tweet objetivo (cantidad + dominante con desempate conservador)
                    df_replies_agg = (
                        df_replies.groupby(["scope", "target_id"])
                                  .agg(
                                      Replies=("reply_id", "count"),
                                      Sentimiento_replies=("Sentimiento", sentimiento_dominante_conservador),
                                      Negativos=("Sentimiento", lambda s: (pd.Series(s) == "Negativo").sum()),
                                      Positivos=("Sentimiento", lambda s: (pd.Series(s) == "Positivo").sum()),
                                      Neutrales=("Sentimiento", lambda s: (pd.Series(s) == "Neutral").sum()),
                                  )
                                  .reset_index()
                    )
        
                    # % negativos por objetivo (por si luego quieres ranking)
                    df_replies_agg["Pct_neg_replies"] = df_replies_agg.apply(
                        lambda r: round((r["Negativos"] / r["Replies"] * 100), 1) if r["Replies"] else 0.0,
                        axis=1
                    )
        
                    # 6) Separar agregados por scope
                    df_replies_conv_agg = df_replies_agg[df_replies_agg["scope"] == "CONV"].copy()
                    df_replies_amp_agg  = df_replies_agg[df_replies_agg["scope"] == "AMP"].copy()
        
        else:
            st.caption("Replies desactivado (no se medirán comentarios).")
                
        # =========================
        # PARTE 7 — KPI + Alertas + Resumen Gemini + Gráficos (con nueva lógica de sentimientos)
        # =========================
        # ✅ Requisitos previos:
        # - df_originales_rank (de PARTE 6) o df_originales
        # - df_conversacion (originales + quotes con sentimiento por fila)
        # - df_amplificacion (agregada por tweet original)
        # - top_terminos_conversacion, top_terminos_amplificacion (si no existen, los calculamos aquí)
        
        st.markdown("## 🧾 Panel ejecutivo (métricas separadas)")
        
        # -----------------------------
        # 1) KPIs base por subconjunto
        # -----------------------------
        n_originales = int(len(df_originales)) if df_originales is not None else 0
        n_quotes = int(len(df_quotes)) if df_quotes is not None else 0
        n_rt_puros = int(len(df_rt_puros)) if df_rt_puros is not None else 0
        
        # Conversación: Originales + Quotes (sin RT puros)
        n_conversacion = int(len(df_conversacion)) if df_conversacion is not None else 0
        
        # Amplificación: agregada por original (cada fila = 1 tweet original amplificado)
        n_originales_amplificados = int(len(df_amplificacion)) if df_amplificacion is not None else 0
        
        # Totales de amplificación
        total_rt_puros = int(df_amplificacion["RT_puros_en_rango"].sum()) if (df_amplificacion is not None and not df_amplificacion.empty) else 0
        total_quotes = int(len(df_quotes)) if (df_quotes is not None and not df_quotes.empty) else 0
        total_ampl = int(df_amplificacion["Ampl_total"].sum()) if (df_amplificacion is not None and not df_amplificacion.empty) else 0
    
        likes_total_amp = int(df_amplificacion["Likesta"].sum()) if (df_amplificacion is not None and not df_amplificacion.empty) else 0
        
        # Interacción conversación (likes+RT de originales+quotes)
        interaccion_conversacion = int(df_conversacion["Interacción"].sum()) if (df_conversacion is not None and not df_conversacion.empty and "Interacción" in df_conversacion.columns) else 0
        
        # -----------------------------
        # 2) Sentimiento conversación (sin duplicar por RT puros)
        # -----------------------------
        def pct_sent(df_x: pd.DataFrame):
            if df_x is None or df_x.empty or "Sentimiento" not in df_x.columns:
                return 0.0, 0.0, 0.0
            total = len(df_x)
            pct_pos = round((df_x["Sentimiento"] == "Positivo").mean() * 100, 1) if total else 0
            pct_neu = round((df_x["Sentimiento"] == "Neutral").mean() * 100, 1) if total else 0
            pct_neg = round((df_x["Sentimiento"] == "Negativo").mean() * 100, 1) if total else 0
            return pct_pos, pct_neu, pct_neg
        
        pct_pos_conv, pct_neu_conv, pct_neg_conv = pct_sent(df_conversacion)
        
        # -----------------------------
        # 3) Sentimiento amplificación (ponderado por RT puros + quotes)
        #    (ya viene calculado en df_amplificacion como Sentimiento_dominante,
        #     pero aquí construimos una "distribución ponderada" para KPI)
        # -----------------------------
        def distribucion_amp_ponderada(df_amp: pd.DataFrame):
            if df_amp is None or df_amp.empty:
                return 0.0, 0.0, 0.0
        
            # Peso = RT_puros_en_rango + Quotes_en_rango (confirmado por ti)
            df_tmp = df_amp.copy()
            df_tmp["peso"] = pd.to_numeric(df_tmp["RT_puros_en_rango"], errors="coerce").fillna(0) + \
                             pd.to_numeric(df_tmp["Quotes_en_rango"], errors="coerce").fillna(0)
        
            total_peso = float(df_tmp["peso"].sum())
            if total_peso <= 0:
                return 0.0, 0.0, 0.0
        
            pos = float(df_tmp.loc[df_tmp["Sentimiento_dominante"] == "Positivo", "peso"].sum())
            neu = float(df_tmp.loc[df_tmp["Sentimiento_dominante"] == "Neutral", "peso"].sum())
            neg = float(df_tmp.loc[df_tmp["Sentimiento_dominante"] == "Negativo", "peso"].sum())
        
            return round(pos/total_peso*100, 1), round(neu/total_peso*100, 1), round(neg/total_peso*100, 1)
        
        def distribucion_amp_rt_only(df_amp: pd.DataFrame):
            if df_amp is None or df_amp.empty:
                return 0.0, 0.0, 0.0
        
            df_tmp = df_amp.copy()
            df_tmp["peso"] = pd.to_numeric(df_tmp["RT_puros_en_rango"], errors="coerce").fillna(0)
        
            total_peso = float(df_tmp["peso"].sum())
            if total_peso <= 0:
                return 0.0, 0.0, 0.0
        
            pos = float(df_tmp.loc[df_tmp["Sentimiento_dominante"] == "Positivo", "peso"].sum())
            neu = float(df_tmp.loc[df_tmp["Sentimiento_dominante"] == "Neutral", "peso"].sum())
            neg = float(df_tmp.loc[df_tmp["Sentimiento_dominante"] == "Negativo", "peso"].sum())
        
            return round(pos/total_peso*100, 1), round(neu/total_peso*100, 1), round(neg/total_peso*100, 1)

        if incl_retweets:            
            pct_pos_amp, pct_neu_amp, pct_neg_amp = distribucion_amp_rt_only(df_amplificacion)
        else:
            pct_pos_amp, pct_neu_amp, pct_neg_amp = 0.0, 0.0, 0.0

        # -----------------------------
        # 4) Temperatura (dos semáforos)
        # -----------------------------
        def calc_temperatura(pct_neg: float, pct_pos: float):
            if pct_neg >= 40:
                return "🔴 Riesgo reputacional"
            if pct_pos >= 60 and pct_neg < 25:
                return "🟢 Clima favorable"
            return "🟡 Mixto / neutro"
        
        temp_conv = calc_temperatura(pct_neg_conv, pct_pos_conv)
        temp_amp = calc_temperatura(pct_neg_amp, pct_pos_amp)
        
        # -----------------------------
        # 5) Narrativas (top términos)
        #    - Conversación: df_conversacion.Texto
        #    - Amplificación: top textos originales amplificados (Texto_original)
        # -----------------------------

        def top_terminos_de_textos(lista_textos: list, top_n: int = 15):
            all_words = []
            for t in (lista_textos or []):
                if t is None:
                    continue
                # evitar NaN
                try:
                    if isinstance(t, float) and pd.isna(t):
                        continue
                except Exception:
                    pass
                all_words.extend(limpiar_texto(t))
        
            if not all_words:
                return pd.Series(dtype=int), []
        
            s = pd.Series(all_words).value_counts().head(top_n)
            return s, s.index.tolist()

        
        # Conversación
        if df_conversacion is not None and not df_conversacion.empty:
            top_terms_conv, top_terms_conv_list = top_terminos_de_textos(df_conversacion["Texto"].tolist(), top_n=15)
        else:
            top_terms_conv, top_terms_conv_list = pd.Series(dtype=int), []
        
        # Amplificación (usa textos originales, NO textos repetidos)
        if df_amplificacion is not None and not df_amplificacion.empty and "Texto_original" in df_amplificacion.columns:
            # priorizamos los más amplificados (top 50) para que el análisis represente lo “grande”
            df_amp_top = df_amplificacion.sort_values("Ampl_total", ascending=False).head(50)
            top_terms_amp, top_terms_amp_list = top_terminos_de_textos(df_amp_top["Texto_original"].tolist(), top_n=15)
        else:
            top_terms_amp, top_terms_amp_list = pd.Series(dtype=int), []
        
        narrativa_conv_1 = top_terms_conv_list[0] if len(top_terms_conv_list) else "N/A"
        narrativa_amp_1 = top_terms_amp_list[0] if len(top_terms_amp_list) else "N/A"
        
        # Top autor (en conversación por interacción)
        if df_conversacion is not None and not df_conversacion.empty:
            # ✅ Si por alguna razón Interacción no existe, la creamos al vuelo
            if "Interacción" not in df_conversacion.columns:
                if ("Likes" in df_conversacion.columns) and ("Retweets" in df_conversacion.columns):
                    df_conversacion["Interacción"] = (
                        pd.to_numeric(df_conversacion["Likes"], errors="coerce").fillna(0) +
                        pd.to_numeric(df_conversacion["Retweets"], errors="coerce").fillna(0)
                    )
                else:
                    df_conversacion["Interacción"] = 0
        
            top_row = df_conversacion.sort_values("Interacción", ascending=False).head(1)
            top_autor = str(top_row.iloc[0].get("Autor", "N/A")) if len(top_row) else "N/A"
        else:
           top_autor = "N/A"

        # -----------------------------
        # 6) Mostrar KPIs (separados) + Replies (opcional)
        # -----------------------------
        incl_replies = st.session_state.get("incl_replies", False)
        
        # --- KPIs Replies (default)
        replies_conv_total = 0
        pct_neg_replies_conv = 0.0
        pct_pos_replies_conv = 0.0
        temp_replies_conv = "N/A"
        
        replies_amp_total = 0
        pct_neg_replies_amp = 0.0
        pct_pos_replies_amp = 0.0
        temp_replies_amp = "N/A"
        
        # Calculamos KPI global de replies por scope (si hay df_replies)
        if incl_replies and (df_replies is not None) and (not df_replies.empty):
            # Conversación
            df_rep_conv = df_replies[df_replies["scope"] == "CONV"].copy()
            replies_conv_total = int(len(df_rep_conv))
            if replies_conv_total > 0:
                pct_neg_replies_conv = round((df_rep_conv["Sentimiento"] == "Negativo").mean() * 100, 1)
                pct_pos_replies_conv = round((df_rep_conv["Sentimiento"] == "Positivo").mean() * 100, 1)
                temp_replies_conv = calc_temperatura_con_min(pct_neg_replies_conv, pct_pos_replies_conv, replies_conv_total, MIN_REPLIES_ALERTA)
        
            # Amplificación
            df_rep_amp = df_replies[df_replies["scope"] == "AMP"].copy()
            replies_amp_total = int(len(df_rep_amp))
            if replies_amp_total > 0:
                pct_neg_replies_amp = round((df_rep_amp["Sentimiento"] == "Negativo").mean() * 100, 1)
                pct_pos_replies_amp = round((df_rep_amp["Sentimiento"] == "Positivo").mean() * 100, 1)
                temp_replies_amp = calc_temperatura_con_min(pct_neg_replies_amp, pct_pos_replies_amp, replies_amp_total, MIN_REPLIES_ALERTA)
        
        # 6.1 Conversación (solo si está seleccionado Originales o Quotes)
        if incl_originales or incl_quotes:
            if incl_replies:
                k1, k2, k3, k13, k10, k12, k15, k16, k17 = st.columns(9)
                k1.metric("Conversación (posts)", f"{n_conversacion}")
                k2.metric("Temp. conversación", temp_conv)
                k3.metric("% Neg (conv)", f"{pct_neg_conv}%")
                k13.metric("% Pos (conv)", f"{pct_pos_conv}%")
                k10.metric("Interacción (conv)", f"{interaccion_conversacion}")
                k12.metric("Narrativa #1 (conv)", narrativa_conv_1)
        
                k15.metric("Replies (conv)", f"{replies_conv_total}")
                k16.metric("Temp. replies (conv)", temp_replies_conv)
                k17.metric("% Neg (replies conv)", f"{pct_neg_replies_conv}%")
            else:
                k1, k2, k3, k13, k10, k12 = st.columns(6)
                k1.metric("Conversación (posts)", f"{n_conversacion}")
                k2.metric("Temp. conversación", temp_conv)
                k3.metric("% Neg (conv)", f"{pct_neg_conv}%")
                k13.metric("% Pos (conv)", f"{pct_pos_conv}%")
                k10.metric("Interacción (conv)", f"{interaccion_conversacion}")
                k12.metric("Narrativa #1 (conv)", narrativa_conv_1)
        else:
            st.info("Conversación oculta: no está seleccionado 'Posts originales' o 'RT con cita'.")
        
        # 6.2 Amplificación (solo si RT puros)
        if incl_retweets:
            if incl_replies:
                k4, k5, k6, k14, k8, k9, k18, k19, k20 = st.columns(9)
                k4.metric("Amplificación (RT puros)", f"{total_ampl}")
                k5.metric("Temp. amplificación", temp_amp)
                k6.metric("% Neg (amp)", f"{pct_neg_amp}%")
                k14.metric("% Pos (amp)", f"{pct_pos_amp}%")
                k8.metric("Likesta (amp)", f"{likes_total_amp}")
                k9.metric("Narrativa #1 (amp)", narrativa_amp_1)
        
                k18.metric("Replies (amp)", f"{replies_amp_total}")
                k19.metric("Temp. replies (amp)", temp_replies_amp)
                k20.metric("% Neg (replies amp)", f"{pct_neg_replies_amp}%")
            else:
                k4, k5, k6, k14, k8, k9 = st.columns(6)
                k4.metric("Amplificación (RT puros)", f"{total_ampl}")
                k5.metric("Temp. amplificación", temp_amp)
                k6.metric("% Neg (amp)", f"{pct_neg_amp}%")
                k14.metric("% Pos (amp)", f"{pct_pos_amp}%")
                k8.metric("Likesta (amp)", f"{likes_total_amp}")
                k9.metric("Narrativa #1 (amp)", narrativa_amp_1)
        else:
            st.info("Amplificación oculta: no está seleccionado 'RT puros'.")
        
        # Caption contextual (para no confundir)
        caption_parts = []
        if (incl_originales or incl_quotes):
            caption_parts.append(f"Conv: Pos {pct_pos_conv}% | Neu {pct_neu_conv}% | Neg {pct_neg_conv}%")
            if incl_replies:
                caption_parts.append(f"Replies(conv): Neg {pct_neg_replies_conv}% | Temp {temp_replies_conv}")
        
        if incl_retweets:
            caption_parts.append(f"Amp: Pos {pct_pos_amp}% | Neu {pct_neu_amp}% | Neg {pct_neg_amp}%")
            if incl_replies:
                caption_parts.append(f"Replies(amp): Neg {pct_neg_replies_amp}% | Temp {temp_replies_amp}")
        
        if caption_parts:
            st.caption(" — ".join(caption_parts))

        
        # -----------------------------
        # 7) Alertas (ajustadas a nueva lógica)
        # -----------------------------
        st.markdown("### 🚨 Alertas")
        alertas = []
        
        # Riesgo por conversación
        if pct_neg_conv >= 40 and n_conversacion >= 10:
            alertas.append("⚠️ Conversación con tono negativo alto. Priorizar mensajes de contención y datos verificables.")
        
        # Riesgo por amplificación (algo negativo se está difundiendo)
        if pct_neg_amp >= 40 and total_ampl >= 20:
            alertas.append("📣 Se está amplificando contenido predominantemente negativo (RT/quotes). Vigilar escalamiento y fuentes.")
        
        # Amplificación alta (viralización)
        if total_ampl >= 200:
            alertas.append("🔥 Amplificación alta. Probable viralización: monitorear evolución por hora/día y cuentas amplificadoras.")
        
        # Poco volumen
        if n_conversacion < 5 and total_ampl < 10:
            alertas.append("ℹ️ Muestra pequeña. Interpretar resultados como señal preliminar (no concluyente).")

        # Replies (conversación)
        if incl_replies and (incl_originales or incl_quotes):
            if replies_conv_total >= MIN_REPLIES_ALERTA and pct_neg_replies_conv >= 40:
                alertas.append("💬 Replies en conversación con tono negativo alto. Señal de descarga emocional: priorizar respuesta/clarificación y monitorear escalamiento.")
        
        # Replies (amplificación)
        if incl_replies and incl_retweets:
            if replies_amp_total >= MIN_REPLIES_ALERTA and pct_neg_replies_amp >= 40:
                alertas.append("💬 Replies en tweets amplificados con tono negativo alto. Riesgo de bola de nieve reputacional: vigilar hilo original, vocerías y contexto.")
        
        # Señal: muchos replies con pocos RT (puede ser debate intenso)
        if incl_replies and (replies_conv_total + replies_amp_total) >= 50 and total_ampl < 20:
            alertas.append("🗣️ Alto volumen de replies con baja amplificación: conversación intensa (debate/descarga) aunque no necesariamente viral. Conviene lectura cualitativa de hilos clave.")
        
        if alertas:
            for a in alertas:
                st.warning(a)
        else:
            st.info("Sin alertas fuertes con los umbrales actuales.")
        
        # -----------------------------
        # 8) Resumen Ejecutivo (Gemini) — con insumos de conversación + amplificación
        # -----------------------------
        st.markdown("## ⭐ Resumen ejecutivo")
        
        # Ejemplos de conversación (top interacción)
        ejemplos_conv = []
        if df_conversacion is not None and not df_conversacion.empty:
            ejemplos_conv = (
                df_conversacion.sort_values("Interacción", ascending=False)
                .head(6)["Texto"]
                .apply(lambda t: (t[:240] + "…") if isinstance(t, str) and len(t) > 240 else t)
                .tolist()
            )
        
        # Ejemplos de amplificación (top amplificados, texto original)
        ejemplos_amp = []
        if df_amplificacion is not None and not df_amplificacion.empty:
            ejemplos_amp = (
                df_amplificacion.sort_values("Ampl_total", ascending=False)
                .head(4)["Texto_original"]
                .apply(lambda t: (t[:240] + "…") if isinstance(t, str) and len(t) > 240 else t)
                .tolist()
            )
        
        payload = {
            "query": query,
            "time_range": time_range,
            "kpis": {
                "conversacion_posts": n_conversacion,
                "originales": n_originales,
                "quotes": n_quotes,
                "rt_puros": n_rt_puros,
                "ampl_total": total_ampl,
                "rt_puros_total": total_rt_puros,
                "quotes_total": total_quotes,
                "replies_conv_total": replies_conv_total if incl_replies else 0,
                "replies_amp_total": replies_amp_total if incl_replies else 0,
                "pct_neg_replies_conv": pct_neg_replies_conv if incl_replies else 0.0,
                "pct_neg_replies_amp": pct_neg_replies_amp if incl_replies else 0.0,
                "temp_replies_conv": temp_replies_conv if incl_replies else "N/A",
                "temp_replies_amp": temp_replies_amp if incl_replies else "N/A",
            },
            "sentimiento_conversacion_pct": {"positivo": pct_pos_conv, "neutral": pct_neu_conv, "negativo": pct_neg_conv},
            "sentimiento_amplificacion_pct_ponderado": {"positivo": pct_pos_amp, "neutral": pct_neu_amp, "negativo": pct_neg_amp},
            "temperatura_conversacion": temp_conv,
            "temperatura_amplificacion": temp_amp,
            "top_terminos_conversacion": top_terms_conv_list[:10],
            "top_terminos_amplificacion": top_terms_amp_list[:10],
            "ejemplos_top_interaccion_conversacion": ejemplos_conv,
            "ejemplos_top_amplificados": ejemplos_amp,
            "nota": "Quotes cuentan como conversación y como amplificación. RT puros solo amplificación. Sentimiento en conversación no se duplica por RT puros.",
            "nota_ubicacion": "Ubicación inferida desde perfil/bio; no es geolocalización exacta.",
            "nota_replies": "Los replies reflejan reacción directa/descarga emocional. Interpretar temperatura y sentimiento de replies como señal complementaria (puede ser más intensa que RT)."
        }
        
        bullets_ia, gemini_status = resumen_ejecutivo_gemini(payload, debug=debug_gemini)
        
        if bullets_ia:
            st.caption(f"Generado con IA (Gemini). Estado: {gemini_status}")
            st.markdown(bullets_ia)
        else:
            st.caption(f"IA no disponible o falló. Estado: {gemini_status}. Mostrando resumen por reglas.")
        
            # Resumen por reglas (sin viñetas largas)
            narrativa = ", ".join(top_terms_conv_list[:6]) if top_terms_conv_list else "sin términos dominantes claros"
            narrativa_amp = ", ".join(top_terms_amp_list[:6]) if top_terms_amp_list else "sin términos dominantes claros"
        
            st.markdown(
                f"**Narrativa:** La conversación reciente se concentra en {narrativa}. "
                f"En paralelo, la amplificación se concentra en {narrativa_amp}.\n\n"
                f"**Riesgos:** Cuando el componente negativo es alto en conversación o amplificación, "
                f"puede escalar rápido por retweets/quotes; conviene monitorear términos nuevos y cuentas amplificadoras.\n\n"
                f"**Oportunidades:** Responder con información verificable, aclaraciones breves y consistentes, "
                f"y mantener monitoreo de cambios de narrativa por día."
            )
        
        st.caption(
            "Advertencia metodológica: señal temprana basada en publicaciones públicas de X; "
            "sentimiento automatizado (IA/fallback) y ubicación inferida desde perfil/bio. "
            "No representa a toda la población."
        )
        
        # -----------------------------
        # 9) Tablero visual (actualizado)
        # -----------------------------
        st.markdown("## 📊 Tablero visual")

        # ─────────────────────────────
        # Inicialización de figuras (blindaje para PDF + rerun)
        # ─────────────────────────────
        fig_vol = None
        fig_sent_conv = None
        fig_sent_amp = None
        fig_terms = None
        fig_terms2 = None
        fig_rep_conv = None
        fig_rep_amp = None
        
        # --- 9.1 Volumen por día (conversación vs RT puros)
        def add_dia(df_x: pd.DataFrame, col_fecha="Fecha"):
            if df_x is None or df_x.empty:
                return df_x
            df_x = df_x.copy()
            df_x[col_fecha] = pd.to_datetime(df_x[col_fecha], errors="coerce")
            df_x["Día"] = df_x[col_fecha].dt.date.astype(str)
            return df_x
        
        df_conv_d = add_dia(df_conversacion)
        df_rt_d = add_dia(df_rt_puros)
        
        if df_conv_d is None or df_conv_d.empty:
            st.info("No hay datos suficientes de conversación para graficar.")
        else:
            vol_conv = df_conv_d.groupby("Día").size().reset_index(name="Conversación")
            if df_rt_d is not None and not df_rt_d.empty:
                vol_rt = df_rt_d.groupby("Día").size().reset_index(name="RT_puros")
                vol = pd.merge(vol_conv, vol_rt, on="Día", how="left").fillna(0)
            else:
                vol = vol_conv.copy()
                vol["RT_puros"] = 0
        
            fig_vol = px.line(vol, x="Día", y=["Conversación", "RT_puros"], markers=True, title="📈 Volumen por día (Conversación vs RT puros)")
            st.plotly_chart(fig_vol, use_container_width=True)
        
        # --- 9.2 Distribución de sentimiento (dos donuts: conversación vs amplificación ponderada)
        colA, colB = st.columns(2)
        
        with colA:
            if df_conversacion is not None and not df_conversacion.empty and "Sentimiento" in df_conversacion.columns:
                sent_counts = df_conversacion["Sentimiento"].value_counts().reset_index()
                sent_counts.columns = ["Sentimiento", "Cantidad"]
                fig_sent_conv = px.pie(sent_counts, names="Sentimiento", values="Cantidad", hole=0.45, title="🧁 Sentimiento — Conversación")
                st.plotly_chart(fig_sent_conv, use_container_width=True)
            else:
                st.info("Sin datos de sentimiento en conversación.")
        
        with colB:
            # ✅ Donut de amplificación SOLO si el usuario marcó RT puros
            if incl_retweets and (df_amplificacion is not None) and (not df_amplificacion.empty):
                tmp = df_amplificacion.copy()   
                tmp["peso"] = pd.to_numeric(tmp["RT_puros_en_rango"], errors="coerce").fillna(0)

                sent_w = tmp.groupby("Sentimiento_dominante")["peso"].sum().reset_index()
                sent_w.columns = ["Sentimiento", "Peso"]
                fig_sent_amp = px.pie(sent_w, names="Sentimiento", values="Peso", hole=0.45,
                                      title="🧁 Sentimiento — Amplificación (ponderado)")
                st.plotly_chart(fig_sent_amp, use_container_width=True)
            else:
                st.info("Gráfico de amplificación oculto: no está seleccionado 'RT puros'.")

        # --- 9.2b Donuts de sentimiento de Replies (si está activado)
        if incl_replies and (df_replies is not None) and (not df_replies.empty):
            st.markdown("### 💬 Sentimiento — Replies (comentarios)")
        
            rA, rB = st.columns(2)
            with rA:
                df_rep_conv = df_replies[df_replies["scope"] == "CONV"].copy()
                if not df_rep_conv.empty:
                    sent_counts_r = df_rep_conv["Sentimiento"].value_counts().reset_index()
                    sent_counts_r.columns = ["Sentimiento", "Cantidad"]
                    fig_rep_conv = px.pie(sent_counts_r, names="Sentimiento", values="Cantidad", hole=0.45, title="🧁 Replies — Conversación")
                    st.plotly_chart(fig_rep_conv, use_container_width=True)
                else:
                    st.info("Sin replies de conversación en el rango.")
        
            with rB:
                df_rep_amp = df_replies[df_replies["scope"] == "AMP"].copy()
                if not df_rep_amp.empty:
                    sent_counts_r2 = df_rep_amp["Sentimiento"].value_counts().reset_index()
                    sent_counts_r2.columns = ["Sentimiento", "Cantidad"]
                    fig_rep_amp = px.pie(sent_counts_r2, names="Sentimiento", values="Cantidad", hole=0.45, title="🧁 Replies — Amplificación")
                    st.plotly_chart(fig_rep_amp, use_container_width=True)
                else:
                    st.info("Sin replies de amplificación en el rango.")
        else:
            st.caption("Replies desactivado: no se muestran donuts de comentarios.")

        # --- 9.3 Sentimiento por día (solo conversación, porque RT puros no deben duplicar)
        if df_conv_d is not None and not df_conv_d.empty and "Sentimiento" in df_conv_d.columns:
            sent_por_dia = df_conv_d.groupby(["Día", "Sentimiento"]).size().reset_index(name="Cantidad")
            fig_sent_dia = px.bar(sent_por_dia, x="Día", y="Cantidad", color="Sentimiento", barmode="stack", title="📆 Sentimiento por día (solo conversación)")
            st.plotly_chart(fig_sent_dia, use_container_width=True)
        else:
            st.info("No hay datos suficientes para 'Sentimiento por día'.")
        
        # --- 9.4 Top términos (dos barras: conversación vs amplificación)
        cT1, cT2 = st.columns(2)
        with cT1:
            if top_terms_conv is not None and len(top_terms_conv) > 0:
                df_terms = top_terms_conv.reset_index()
                df_terms.columns = ["Término", "Frecuencia"]
                fig_terms = px.bar(df_terms, x="Frecuencia", y="Término", orientation="h", title="🏷️ Top términos — Conversación")
                st.plotly_chart(fig_terms, use_container_width=True)
            else:
                st.info("Sin términos dominantes en conversación.")
        
        with cT2:
            if top_terms_amp is not None and len(top_terms_amp) > 0:
                df_terms2 = top_terms_amp.reset_index()
                df_terms2.columns = ["Término", "Frecuencia"]
                fig_terms2 = px.bar(df_terms2, x="Frecuencia", y="Término", orientation="h", title="🏷️ Top términos — Amplificación (originales amplificados)")
                st.plotly_chart(fig_terms2, use_container_width=True)
            else:
                st.info("Sin términos dominantes en amplificación.")


        # =========================
        # PARTE 6 — 4 TABLAS FINALES (Originales + Amplificación) con "Abrir"
        # =========================
        # ✅ Requisitos previos (de PARTE 3–5):
        # - df_originales: solo posts originales dentro del rango (filas por tweet original)
        # - df_conversacion: originales + quotes (con Sentimiento por fila, Ubicación, Confianza, etc.)
        # - df_amplificacion: agregada por tweet ORIGINAL amplificado
        #   Debe contener (mínimo): original_id, Texto_original, URL_original,
        #   Ampl_total (RT_puros+Quotes), RT_puros_en_rango, Quotes_en_rango,
        #   Fechaua, Liketa,
        #   Sentimiento_dominante, Ubicación_dominante, Confianza_dominante
        #
        # Si tus nombres difieren, ajusta SOLO los nombres de columna en los selects.
        
        st.markdown("## 📌 Resultados en tablas (4 vistas)")
        
        # ------------------------------------------------------------
        # ------------------------------------------------------------
        # TABLA 1 y 2 — "Conversación" según checks
        #   - Si incl_originales: muestra Originales
        #   - Si NO incl_originales y sí incl_quotes: muestra Quotes
        #   - Si ninguno: no muestra conversación
        #     Conversación = Originales + Quotes
        # ------------------------------------------------------------
        
        # 1) Definir cuál dataframe se mostrará como "conversación"
           
        df_conv_base = pd.DataFrame()
        titulo_top = ""
        titulo_all = ""
        
        cols_conv = [
            "tipo",
            "Autor", "Fecha", "Likes", "Retweets", "Interacción",
            "Sentimiento",
            "Replies", "Sentimiento_replies",
            "Ubicación inferida", "Confianza",
            "Texto", "Link"
        ]
        
        # ✅ Si hay conversación calculada (originales + quotes), usamos esa como fuente
        if df_conversacion is not None and (not df_conversacion.empty):
            df_conv_base = df_conversacion.copy()
        
            # Títulos dinámicos según selección
            if incl_originales and incl_quotes:
                titulo_top = "1) 🔥 Top 10 — Conversación (Originales + Quotes)"
                titulo_all = "2) 📄 Ver TODA la conversación (Originales + Quotes)"
            elif incl_originales and (not incl_quotes):
                titulo_top = "1) 🔥 Top 10 — Posts originales (no RT)"
                titulo_all = "2) 📄 Ver TODOS los posts originales (no RT)"
            elif (not incl_originales) and incl_quotes:
                titulo_top = "1) 🔥 Top 10 — Retweets con cita (Quotes)"
                titulo_all = "2) 📄 Ver TODOS los retweets con cita (Quotes)"
            else:
                # Por coherencia (aunque normalmente no llega acá)
                titulo_top = "1) 🔥 Top 10 — Conversación"
                titulo_all = "2) 📄 Ver TODA la conversación"
        
        # Renderizar
        if df_conv_base is None or df_conv_base.empty:
            st.info("No se muestran tablas de 'Conversación' porque no hay datos (Originales/Quotes) con los filtros actuales.")
        else:
            # ---------------------------------------------------------
            # Enriquecer conversación con Replies agregados (si existe)
            # ---------------------------------------------------------
            # Blindaje: puede no existir df_replies_conv_agg si replies está apagado o falló la búsqueda
            if "incl_replies" not in locals():
                incl_replies = st.session_state.get("incl_replies", False)
        
            tiene_replies_conv_agg = ("df_replies_conv_agg" in locals()) and (df_replies_conv_agg is not None) and (not df_replies_conv_agg.empty)
        
            if incl_replies and tiene_replies_conv_agg:
                # df_replies_conv_agg: scope=CONV, target_id = tweet_id
                tmp_rep = df_replies_conv_agg.rename(columns={"target_id": "tweet_id"}).copy()
                tmp_rep["tweet_id"] = tmp_rep["tweet_id"].astype(str)
        
                # Blindaje: aseguramos tweet_id
                if "tweet_id" not in df_conv_base.columns:
                    df_conv_base["tweet_id"] = ""
        
                df_conv_base["tweet_id"] = df_conv_base["tweet_id"].astype(str)
        
                df_conv_base = df_conv_base.merge(
                    tmp_rep[["tweet_id", "Replies", "Sentimiento_replies"]],
                    on="tweet_id",
                    how="left"
                )
        
                df_conv_base["Replies"] = pd.to_numeric(df_conv_base["Replies"], errors="coerce").fillna(0).astype(int)
                df_conv_base["Sentimiento_replies"] = df_conv_base["Sentimiento_replies"].fillna("N/A")
            else:
                # columnas por consistencia
                df_conv_base["Replies"] = 0
                df_conv_base["Sentimiento_replies"] = "N/A"

            # ✅ Rank por Score = Interacción + (W_REPLIES * Replies)
            if "Interacción" not in df_conv_base.columns:
                df_conv_base["Interacción"] = (
                    pd.to_numeric(df_conv_base.get("Likes", 0), errors="coerce").fillna(0) +
                    pd.to_numeric(df_conv_base.get("Retweets", 0), errors="coerce").fillna(0)
                )
            
            df_conv_base["Replies"] = pd.to_numeric(df_conv_base.get("Replies", 0), errors="coerce").fillna(0).astype(int)
            df_conv_base["Score_ranking"] = df_conv_base["Interacción"] + (W_REPLIES * df_conv_base["Replies"])
            
            df_conv_rank = df_conv_base.sort_values("Score_ranking", ascending=False).copy()
        
            # Si por cualquier motivo quedaron títulos vacíos, ponemos fallback
            if not titulo_top:
                titulo_top = "1) 🔥 Top 10 — Conversación"
            if not titulo_all:
                titulo_all = "2) 📄 Ver TODA la conversación"
        
            # TABLA 1) TOP 10
            render_table(
                df_conv_rank,
                titulo_top,
                cols=cols_conv,
                top=10
            )

            # 👇 UX1: leer replies por fila (Top 10 conversación)
            if incl_replies and (df_replies is not None) and (not df_replies.empty):
                st.markdown("#### 💬 Leer replies — TOP 10 (Conversación)")
                df_conv_top10 = df_conv_rank.head(10).copy()
                render_replies_expanders_top10(
                    df_top10=df_conv_top10,
                    df_replies=df_replies,
                    scope="CONV",
                    id_col="tweet_id",
                    title_prefix="💬 Replies (conv)"
                )
        
            # TABLA 2) TODOS
            with st.expander(titulo_all):
                render_table(
                    df_conv_rank,
                    titulo_all,
                    cols=cols_conv,
                    top=None
                )    
        # ------------------------------------------------------------
        # TABLA 3) TOP 10 — Amplificación (muestra el TWEET ORIGINAL)
        # Ranking: Ampl_total (RT puros + Quotes) en el rango
        # ------------------------------------------------------------
        
        cols_top_amp = [
                "Autor",
                "Fechaua",
                "Ampl_total",
                "RT_puros_en_rango",
                "Likesta",
                "Sentimiento_dominante",
                "Replies", "Sentimiento_replies",
                "Ubicación_dominante", "Confianza_dominante",
                "Texto_original",
                "Link"
        ]
        
        if incl_retweets:
            st.warning("DEBUG AMPLIFICACION")
            
                try:
                    st.write("Cantidad df_rt_puros:", len(df_rt_puros))
                except Exception as e:
                    st.write("ERROR df_rt_puros:", e)
                
                try:
                    st.write("Columnas df_rt_puros:")
                    st.write(df_rt_puros.columns.tolist())
                except Exception as e:
                    st.write("ERROR columnas:", e)
                
                try:
                    st.write("Primeras filas df_rt_puros:")
                    st.dataframe(df_rt_puros.head(5))
                except Exception as e:
                    st.write("ERROR preview:", e)
    
                 try:
                    st.write("Cantidad df_amplificacion:", len(df_amplificacion) if df_amplificacion is not None else "None")
                except Exception as e:
                    st.write("ERROR df_amplificacion:", e)
                
            if df_amplificacion is not None and not df_amplificacion.empty:
                # ranking base (luego re-rankearemos por Score_ranking)
                df_amp_rank = df_amplificacion.copy()
            else:
                df_amp_rank = pd.DataFrame()
        
            if df_amp_rank.empty:
                st.info("No se encontraron datos de amplificación para mostrar.")
            else:
                # ─────────────────────────────────────────────
                # 1) Enriquecer amplificación con Replies agregados (si existe)
                # ─────────────────────────────────────────────
                if incl_replies and (df_replies_amp_agg is not None) and (not df_replies_amp_agg.empty):
                    tmp_rep2 = df_replies_amp_agg.rename(columns={"target_id": "original_id"}).copy()
                    tmp_rep2["original_id"] = tmp_rep2["original_id"].astype(str)
        
                    df_amp_rank["original_id"] = df_amp_rank["original_id"].astype(str)
                    df_amp_rank = df_amp_rank.merge(
                        tmp_rep2[["original_id", "Replies", "Sentimiento_replies"]],
                        on="original_id",
                        how="left"
                    )
                    df_amp_rank["Replies"] = pd.to_numeric(df_amp_rank["Replies"], errors="coerce").fillna(0).astype(int)
                    df_amp_rank["Sentimiento_replies"] = df_amp_rank["Sentimiento_replies"].fillna("N/A")
                else:
                    df_amp_rank["Replies"] = 0
                    df_amp_rank["Sentimiento_replies"] = "N/A"
        
                # ─────────────────────────────────────────────
                # 2) Blindajes de columnas clave (Ampl_total, Fechaua, URL_original)
                # ─────────────────────────────────────────────
                # Ampl_total
                if "Ampl_total" not in df_amp_rank.columns:
                    rt_col = "RT_puros_en_rango" if "RT_puros_en_rango" in df_amp_rank.columns else None
                    q_col  = "Quotes_en_rango" if "Quotes_en_rango" in df_amp_rank.columns else None
        
                    if rt_col and q_col:
                        df_amp_rank["Ampl_total"] = (
                            pd.to_numeric(df_amp_rank[rt_col], errors="coerce").fillna(0) +
                            pd.to_numeric(df_amp_rank[q_col], errors="coerce").fillna(0)
                        )
                    elif rt_col:
                        df_amp_rank["Ampl_total"] = pd.to_numeric(df_amp_rank[rt_col], errors="coerce").fillna(0)
                    else:
                        df_amp_rank["Ampl_total"] = 0
        
                df_amp_rank["Ampl_total"] = pd.to_numeric(df_amp_rank["Ampl_total"], errors="coerce").fillna(0)
        
                # Fechaua (si no existe)
                if "Fechaua" not in df_amp_rank.columns:
                    if "Fecha_ultima_amplificacion" in df_amp_rank.columns:
                        _f = pd.to_datetime(df_amp_rank["Fecha_ultima_amplificacion"], errors="coerce", utc=True)
                        df_amp_rank["Fechaua"] = _f.dt.tz_convert(None)
                    else:
                        df_amp_rank["Fechaua"] = ""
        
                # URL_original (para que Link no quede vacío)
                if "URL_original" not in df_amp_rank.columns:
                    if "original_id" in df_amp_rank.columns:
                        df_amp_rank["URL_original"] = df_amp_rank["original_id"].astype(str).apply(
                            lambda oid: f"https://x.com/i/web/status/{oid}"
                        )
                    else:
                        df_amp_rank["URL_original"] = ""
        
                # ─────────────────────────────────────────────
                # 3) Rank por Score = Ampl_total + (W_REPLIES * Replies)
                # ─────────────────────────────────────────────
                df_amp_rank["Replies"] = pd.to_numeric(df_amp_rank.get("Replies", 0), errors="coerce").fillna(0).astype(int)
                df_amp_rank["Score_ranking"] = df_amp_rank["Ampl_total"] + (W_REPLIES * df_amp_rank["Replies"])
                df_amp_rank = df_amp_rank.sort_values("Score_ranking", ascending=False).copy()
        
                # ─────────────────────────────────────────────
                # 4) Render tabla TOP 10
                # ─────────────────────────────────────────────
              
                render_table(
                    df_amp_rank,
                    "3) 📣 Top 10 — Amplificación (muestra el tweet ORIGINAL amplificado)",
                    cols=cols_top_amp,
                    top=10
                )
        
                # ─────────────────────────────────────────────
                # 5) UX1: leer replies por fila (Top 10 amplificación)
                # ─────────────────────────────────────────────
                if incl_replies and (df_replies is not None) and (not df_replies.empty):
                    st.markdown("#### 💬 Leer replies — TOP 10 (Amplificación)")
                    df_amp_top10 = df_amp_rank.head(10).copy()
                    render_replies_expanders_top10(
                        df_top10=df_amp_top10,
                        df_replies=df_replies,
                        scope="AMP",
                        id_col="original_id",
                        title_prefix="💬 Replies (amp)"
                    )
            
            # ------------------------------------------------------------
            # TABLA 4) TODOS — Amplificación (muestra el TWEET ORIGINAL)
            # ------------------------------------------------------------
            with st.expander("4) 📄 Ver TODA la amplificación (tweet ORIGINAL agregado)"):
                render_table(
                    df_amp_rank,
                    "4) 📄 Toda la amplificación (tweet ORIGINAL agregado)",
                    cols=cols_top_amp,
                    top=None
                )

            if df_amp_rank is not None and not df_amp_rank.empty:
                with st.expander("4) 📄 Ver TODA la amplificación (tweet ORIGINAL agregado)"):
                    render_table(
                        df_amp_rank,
                        "4) 📄 Toda la amplificación (tweet ORIGINAL agregado)",
                        cols=cols_top_amp,
                        top=None
                    )
            else:
                st.info("No hay datos para mostrar en la tabla 4 de amplificación.")
        else:
            st.info("No se muestra 'Amplificación' porque no está seleccionado 'RT puros'.")
        
        st.caption(
            "Nota: En Amplificación, se muestra el tweet ORIGINAL una sola vez por fila. "
            "Los RT puros y quotes se contabilizan en columnas (RT_puros_en_rango, Quotes_en_rango, Ampl_total). "
            "El botón 'Abrir' siempre abre el tweet ORIGINAL."
        )

        # ─────────────────────────────
        # Preparar recursos para PDF (NO consume cuota)
        # ─────────────────────────────
        report_kpis = {
            "Conversación (posts)": n_conversacion,
            "Temp. conversación": temp_conv,
            "% Neg (conv)": f"{pct_neg_conv}%",
            "% Pos (conv)": f"{pct_pos_conv}%",
            "Interacción (conv)": interaccion_conversacion,
            "Narrativa #1 (conv)": narrativa_conv_1,
            "Amplificación (RT puros)": total_ampl if incl_retweets else 0,
            "Temp. amplificación": temp_amp if incl_retweets else "N/A",
            "% Neg (amp)": f"{pct_neg_amp}%" if incl_retweets else "0%",
            "% Pos (amp)": f"{pct_pos_amp}%" if incl_retweets else "0%",
            "Likesta (amp)": likes_total_amp if incl_retweets else 0,
            "Narrativa #1 (amp)": narrativa_amp_1 if incl_retweets else "N/A",
        }
        
        if incl_replies:
            report_kpis.update({
                "Replies (conv)": replies_conv_total,
                "Temp. replies (conv)": temp_replies_conv,
                "% Neg (replies conv)": f"{pct_neg_replies_conv}%",
                "Replies (amp)": replies_amp_total,
                "Temp. replies (amp)": temp_replies_amp,
                "% Neg (replies amp)": f"{pct_neg_replies_amp}%",
            })
        
        report_nota = (
            "Advertencia metodológica: señal temprana basada en publicaciones públicas de X; "
            "sentimiento automatizado (IA/fallback) y ubicación inferida desde perfil/bio. En la presente versión análisis generado mediante IA Gemini"
            "No representa a toda la población. "
            "Quotes cuentan como conversación; RT puros solo amplificación."
        )
         
        # ─────────────────────────────
        # Exportar figuras Plotly a PNG (para PDF)
        # ─────────────────────────────
        # figs_png = {
        #    "fig_vol": _plotly_to_png_bytes(fig_vol),
        #    "fig_sent_conv": _plotly_to_png_bytes(fig_sent_conv),
        #    "fig_sent_amp": _plotly_to_png_bytes(fig_sent_amp),
        #    "fig_terms_conv": _plotly_to_png_bytes(fig_terms),
        #    "fig_terms_amp": _plotly_to_png_bytes(fig_terms2),
        #    "fig_rep_conv": _plotly_to_png_bytes(fig_rep_conv),
        #    "fig_rep_amp": _plotly_to_png_bytes(fig_rep_amp),
        #}
        
        # Filtrar solo las figuras que realmente existen
        #figs_png = {k: v for k, v in figs_png.items() if v is not None}

        # ─────────────────────────────
        # Guardar resultados en session_state (CLAVE para que no se borre al cambiar selects)
        # ─────────────────────────────
        # if st.session_state.get("LAST_PNG_ERROR"):
        #     st.warning("Export PNG falló: " + st.session_state["LAST_PNG_ERROR"])

        _save_results(
            DF_CONV_RANK=df_conv_rank if "df_conv_rank" in locals() else pd.DataFrame(),
            DF_AMP_RANK=df_amp_rank if "df_amp_rank" in locals() else pd.DataFrame(),
            DF_REPLIES=df_replies if "df_replies" in locals() else pd.DataFrame(),
            DF_REPLIES_CONV_AGG=df_replies_conv_agg if "df_replies_conv_agg" in locals() else pd.DataFrame(),
            DF_REPLIES_AMP_AGG=df_replies_amp_agg if "df_replies_amp_agg" in locals() else pd.DataFrame(),
            COLS_CONV=cols_conv,
            COLS_TOP_AMP=cols_top_amp,

            # ✅ NUEVO: para PDF
            REPORT_KPIS=report_kpis,
            REPORT_ALERTAS=alertas,
            REPORT_RESUMEN_MD=bullets_ia if bullets_ia else "",
            REPORT_NOTA_METODO=report_nota,
            
            # ✅ NUEVO: guardamos las FIGURAS PLOTLY (NO PNG) para exportarlas recién al generar PDF
            REPORT_FIGS_PLOTLY={
                "fig_vol": fig_vol if "fig_vol" in locals() else None,
                "fig_sent_conv": fig_sent_conv if "fig_sent_conv" in locals() else None,
                "fig_sent_amp": fig_sent_amp if "fig_sent_amp" in locals() else None,
                "fig_terms_conv": fig_terms if "fig_terms" in locals() else None,
                "fig_terms_amp": fig_terms2 if "fig_terms2" in locals() else None,
                "fig_rep_conv": fig_rep_conv if "fig_rep_conv" in locals() else None,
                "fig_rep_amp": fig_rep_amp if "fig_rep_amp" in locals() else None,
            },
            
            # (opcional: guardar query/time_range explícito)
            query=query,
            time_range=time_range,
        )
        st.session_state["SKIP_PERSISTENT_RENDER_ONCE"] = True

# ─────────────────────────────
# Render persistente: si ya hay resultados, se muestran aunque cambies selects
# ─────────────────────────────
if st.session_state.get("HAS_RESULTS", False):
    
    # ✅ 0) SIEMPRE mostrar descarga PDF (inclusive en el run posterior a "Buscar en X")
    render_pdf_controls()
    st.divider()

    # ✅ 1) Evitar duplicar SOLO el “cuerpo persistente” cuando vienes de "Buscar en X"
    #     OJO: ya NO usamos st.stop() antes del PDF.
    if st.session_state.get("SKIP_PERSISTENT_RENDER_ONCE", False):
        st.session_state["SKIP_PERSISTENT_RENDER_ONCE"] = False
        # En este run ya se mostró todo dentro del botón, así que aquí solo dejamos el PDF visible.
    else:
        # ✅ 2) En reruns (por filtros), re-renderizamos TODO lo previo a las tablas
        render_persisted_header_kpis_alertas_resumen()
        render_persisted_visuals()
        st.divider()
        render_persisted_tables_and_replies()
