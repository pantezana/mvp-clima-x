import streamlit as st
import tweepy
import re
import pandas as pd
import requests
import time
import plotly.express as px
import json
from datetime import datetime, timedelta

st.set_page_config(page_title="MVP Clima en X", layout="wide")
st.title("🖥️ MVP – Clima del Tema en X")

bearer_token = st.secrets["X_BEARER_TOKEN"]
client = tweepy.Client(bearer_token=bearer_token)

query = st.text_input("Palabras clave / hashtags")

time_range = st.selectbox(
    "Rango temporal",
    ["24 horas", "48 horas","72 horas", "7 días", "30 días"]
)

debug_gemini = False

# Límite de publicaciones a consultar (control de cuota)
limite_opcion = st.selectbox(
    "Límite de publicaciones a consultar (control de cuota X)",
    ["50", "100", "200", "500", "1000", "Sin límite (hasta donde llegue X)"],
    index=2  # 200 por defecto (ajústalo si quieres)
)

max_posts = None if "Sin límite" in limite_opcion else int(limite_opcion)

st.markdown("### 🎛️ Tipo de contenido a analizar")

c1, c2, c3 = st.columns(3)
with c1:
    incluir_originales = st.checkbox("Posts originales", value=True)
with c2:
    incluir_retweets = st.checkbox("Retweets (RT puros)", value=True)
with c3:
    incluir_quotes = st.checkbox("Retweets con cita (quote)", value=True)

# Regla simple de validación (neófito-friendly)
if not (incluir_originales or incluir_retweets or incluir_quotes):
    st.warning("Selecciona al menos un tipo de contenido (Originales, RT puros o Quotes).")

# Nota de uso (educativa)
st.caption(
    "Tip: Si eliges solo 'Posts originales', tu análisis no se llenará de retweets repetidos. "
    "Si incluyes RT/Quotes, verás también 'Amplificación' (qué post se está difundiendo)."
)

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

# ─────────────────────────────
# Selector de modelo de sentimiento (Hugging Face)
# ─────────────────────────────
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
Incluyen volumen, distribución de sentimiento estimada, términos dominantes y ejemplos de posts con más interacción.

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

# --- Preparación de texto
            
# Stopwords básicas en español (MVP)
STOPWORDS = set([
    "de","la","que","el","en","y","a","los","del","se","las","por","un","para","con",
    "no","una","su","al","lo","como","más","pero","sus","le","ya","o","este","sí",
    "porque","esta","entre","cuando","muy","sin","sobre"
])
            
def limpiar_texto(texto):
    palabras = re.findall(r"\b[a-záéíóúñ]+\b", (texto or "").lower())
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


if st.button("Buscar en X"):
    now = time.time()
    if now - st.session_state["last_search_ts"] < 20:
        st.warning("Espera 20 segundos entre búsquedas para evitar límites de X.")
        st.stop()
    st.session_state["last_search_ts"] = now

    if not query:
        st.warning("Ingresa una palabra clave")
    else:
        start_time = get_start_time(time_range).isoformat("T") + "Z"

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
                max_posts=max_posts,
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
        
        st.markdown("### 🙂 Sentimiento (sin duplicar por retweets)")
        
        # ---------------------------------------------------------
        # 4.1) Definir “conversación”:
        # - Conversación incluye: originales + quotes (porque quotes sí aportan comentario nuevo)
        # - RT puros NO entran a conversación (son amplificación pura y repiten texto)
        # ---------------------------------------------------------
        df_conversacion = pd.concat([df_originales, df_quotes], ignore_index=True)
        
        if df_conversacion.empty:
            st.warning("No hay 'conversación' (originales + quotes) en el rango seleccionado.")
            st.stop()
        
        # ---------------------------------------------------------
        # 4.2) Sentimiento por fila SOLO en conversación (originales + quotes)
        #     (acá sí tiene sentido por fila porque el texto cambia)
        # ---------------------------------------------------------
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
        #     Incluye:
        #       - RT puros + Quotes (en rango)
        #       - Sentimiento_dominante ponderado por (RT_puros + Quotes)  ✅ (tu decisión)
        #       - Fechaua = Fecha_última_amplificación (tu decisión)
        #       - Likesta = Likes_total_amplificación (tu decisión)
        #     Nota: aquí además guardamos Ubicación/Confianza como “dominante” (modo)
        # ---------------------------------------------------------
        df_amplificacion = pd.DataFrame()
        
        if (not df_rt_agregado.empty) or (not df_quotes_agregado.empty):
            # Base = unión por original_id
            base = df_rt_agregado.copy()
            if base.empty:
                base = pd.DataFrame({"original_id": df_quotes_agregado["original_id"].astype(str)})
        
            base["original_id"] = base["original_id"].astype(str)
        
            base = base.merge(df_quotes_agregado, on="original_id", how="outer")
        
            # Rellenos
            for c in ["RT_puros_en_rango", "Likes_total_amplificacion", "Retweets_total_amplificacion"]:
                if c not in base.columns:
                    base[c] = 0
                base[c] = pd.to_numeric(base[c], errors="coerce").fillna(0)
        
            for c in ["Quotes_en_rango", "Likes_total_quotes", "Retweets_total_quotes"]:
                if c not in base.columns:
                    base[c] = 0
                base[c] = pd.to_numeric(base[c], errors="coerce").fillna(0)
        
            # Amplificación total (en rango)
            base["Ampl_total"] = base["RT_puros_en_rango"] + base["Quotes_en_rango"]
        
            # Fecha última amplificación (recomendado)
            # - Preferimos max entre (Fecha_ultima_amplificacion, Fecha_ultima_quote)
 
            if "Fecha_ultima_amplificacion" in base.columns:
                base["Fecha_ultima_amplificacion"] = pd.to_datetime(base["Fecha_ultima_amplificacion"], errors="coerce")
            else:
                base["Fecha_ultima_amplificacion"] = pd.NaT
            
            if "Fecha_ultima_quote" in base.columns:
                base["Fecha_ultima_quote"] = pd.to_datetime(base["Fecha_ultima_quote"], errors="coerce")
            else:
                base["Fecha_ultima_quote"] = pd.NaT
            
            # 1) Aseguramos datetime
            f1 = pd.to_datetime(base["Fecha_ultima_amplificacion"], errors="coerce", utc=True)
            f2 = pd.to_datetime(base["Fecha_ultima_quote"], errors="coerce", utc=True)
            
            # 2) Quitamos la zona horaria (dejamos "naive") para poder comparar sin error
            f1 = f1.dt.tz_convert(None)
            f2 = f2.dt.tz_convert(None)
            
            # 3) Elegimos la más reciente entre ambas
            base["Fechaua"] = f1.where(f1 >= f2, f2)
            base["Fechaua"] = base["Fechaua"].fillna(f1).fillna(f2)

            st.write("Tipos:", base["Fecha_ultima_amplificacion"].dtype, base["Fecha_ultima_quote"].dtype)
            
            # Likes totales amplificación (RT + quote)
            base["Likesta"] = base["Likes_total_amplificacion"] + base["Likes_total_quotes"]
        
            # Retweets totales amplificación (RT + quote) (métrica adicional)
            base["Retweets"] = base["Retweets_total_amplificacion"] + base["Retweets_total_quotes"]
        
            # ---------------------------
            # Sentimiento dominante ponderado (RT_puros + quotes)
            # ---------------------------
            # 1) Sentimiento del original (ya calculado 1 vez) -> viene de df_rt_agregado
            # 2) Sentimiento de quotes: se calcula por fila en df_conversacion (quotes están ahí)
            #
            # Ponderación:
            # - Peso RT = RT_puros_en_rango (todos repiten el mismo sentimiento del original)
            # - Peso quotes = se usa sentimiento de cada quote y se suma 1 por quote
            #
            # Resultado: Sentimiento_dominante = el que tenga mayor peso total.
            # Score_dominante: (peso_ganador / peso_total) aprox.
            sentiment_map = {}
        
            # A) Contribución RT (sentimiento_original) con peso RT_puros_en_rango
            if not df_rt_agregado.empty:
                for _, row in df_rt_agregado.iterrows():
                    oid = str(row.get("original_id"))
                    sent = row.get("Sentimiento_original")
                    w = float(row.get("RT_puros_en_rango", 0) or 0)
                    if not oid:
                        continue
                    sentiment_map.setdefault(oid, {"Positivo": 0.0, "Neutral": 0.0, "Negativo": 0.0})
                    if sent in sentiment_map[oid]:
                        sentiment_map[oid][sent] += w
        
            # B) Contribución Quotes (cada quote suma 1 con su sentimiento)
            if not df_quotes.empty:
                # Necesitamos el sentimiento de cada quote (ya está en df_conversacion; filtramos solo tipo Quote)
                df_quotes_sent = df_conversacion[df_conversacion["tipo"] == "Quote"].copy()
                if not df_quotes_sent.empty:
                    for _, r in df_quotes_sent.iterrows():
                        oid = str(r.get("original_id"))
                        sent = r.get("Sentimiento")
                        if not oid:
                            continue
                        sentiment_map.setdefault(oid, {"Positivo": 0.0, "Neutral": 0.0, "Negativo": 0.0})
                        if sent in sentiment_map[oid]:
                            sentiment_map[oid][sent] += 1.0  # cada quote pesa 1
        
            # Resolver dominante
            dominantes = []
            for _, row in base.iterrows():
                oid = str(row.get("original_id"))
                weights = sentiment_map.get(oid, {"Positivo": 0.0, "Neutral": 0.0, "Negativo": 0.0})
                total_w = sum(weights.values()) if weights else 0.0
                if total_w <= 0:
                    dominantes.append(("Neutral", None))
                    continue
                dom = max(weights.items(), key=lambda kv: kv[1])[0]
                score_dom = round(float(weights[dom] / total_w), 3) if total_w else None
                dominantes.append((dom, score_dom))
        
            base["Sentimiento_dominante"] = [d[0] for d in dominantes]
            base["Score_sent_dominante"] = [d[1] for d in dominantes]
        
            # ---------------------------
            # Ubicación/Confianza “dominante” (modo) tomado de retweets+quotes
            # ---------------------------
            # Usamos TODAS las filas de df_raw donde original_id = X (RT o Quote)
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
                # Tomamos solo amplificaciones (RT+Quote) para inferir ubicación del “público que amplifica”
                amp_rows = df_raw[df_raw["tipo"].isin(["RT", "Quote"]) & df_raw["original_id"].notna()].copy()
        
                ubis = []
                confs = []
                for oid in base["original_id"].astype(str).tolist():
                    g = amp_rows[amp_rows["original_id"].astype(str) == str(oid)]
                    ubis.append(modo_safe(g["Ubicación inferida"]) if not g.empty else None)
                    confs.append(modo_safe(g["Confianza"]) if not g.empty else None)
        
                base["Ubicación_dominante"] = ubis
                base["Confianza_dominante"] = confs
        
            # ---------------------------
            # Link “Abrir” al tweet original:
            # - Si el original está en df_originales: usamos su URL (ideal)
            # - Si no: construimos URL genérica por id (X igual abre por id si existe)
            # ---------------------------
            url_por_original_id = {}
            if not df_originales.empty:
                for _id, _url in zip(df_originales["tweet_id"].tolist(), df_originales["URL"].tolist()):
                    if _id:
                        url_por_original_id[str(_id)] = _url
        
            base["URL_original"] = base["original_id"].astype(str).apply(lambda oid: url_por_original_id.get(oid, f"https://x.com/i/web/status/{oid}"))
        
            # Texto del original:
            # - Si lo tenemos en df_rt_agregado (Texto_base_original), úsalo
            # - Si no, vacío
            if "Texto_base_original" not in base.columns:
                base["Texto_base_original"] = ""

            base["Texto_original"] = base["Texto_base_original"]
        
            df_amplificacion = base.copy()
        
        # ---------------------------------------------------------
        # 4.6) Mensajes de control (para neófitos)
        # ---------------------------------------------------------
        st.caption(
            f"Sentimiento conversación calculado en {len(df_conversacion)} fila(s) (Originales + Quotes). "
            f"RT puros se agregan por tweet original para NO inflar el sentimiento."
        )
        
        st.caption(
            f"Método de sentimiento (conversación): {metodo_sent_conv}. "
            f"En amplificación: dominante ponderado por (RT_puros + Quotes)."
        )
        
        # Guardamos dfs clave para PARTE 5/6 (KPIs + tablas + gráficos)
        st.session_state["df_conversacion_rows"] = int(len(df_conversacion))
        st.session_state["df_amplificacion_rows"] = int(len(df_amplificacion)) if df_amplificacion is not None else 0

                        
        # =========================
        # PARTE 5 — KPIs + Alertas + Resumen ejecutivo (Gemini) + Tablero visual (sin inflar)
        # =========================
        # ✅ DÓNDE PEGAR:
        # Pega este bloque JUSTO DESPUÉS de la PARTE 4
        # (después de construir df_conversacion y df_amplificacion)
        # y ANTES de mostrar las 4 tablas (eso será PARTE 6).
        
        # ---------------------------------------------------------
        # 5.1) KPIs base (separados: Conversación vs Amplificación)
        # ---------------------------------------------------------
        st.markdown("## 🧾 Panel ejecutivo (mejorado)")
        
        # Asegurar tipos
        df_conversacion["Fecha"] = pd.to_datetime(df_conversacion["Fecha"], errors="coerce")
        df_conversacion["Likes"] = pd.to_numeric(df_conversacion["Likes"], errors="coerce").fillna(0)
        df_conversacion["Retweets"] = pd.to_numeric(df_conversacion["Retweets"], errors="coerce").fillna(0)
        df_conversacion["Interacción"] = pd.to_numeric(df_conversacion.get("Interacción", 0), errors="coerce").fillna(0)
        
        conv_total = int(len(df_conversacion))
        
        # % sentimiento SOLO conversación (no RT puros)
        pct_pos = round((df_conversacion["Sentimiento"] == "Positivo").mean() * 100, 1) if conv_total else 0
        pct_neu = round((df_conversacion["Sentimiento"] == "Neutral").mean() * 100, 1) if conv_total else 0
        pct_neg = round((df_conversacion["Sentimiento"] == "Negativo").mean() * 100, 1) if conv_total else 0
        
        conv_interaccion_total = int(df_conversacion["Interacción"].sum()) if conv_total else 0
        conv_interaccion_prom = round(df_conversacion["Interacción"].mean(), 2) if conv_total else 0
        
        # Narrativas dominantes (top términos) SOLO conversación
        todas_palabras = []
        for t in df_conversacion["Texto"].tolist():
            todas_palabras.extend(limpiar_texto(t))
        
        top_terminos = pd.Series(todas_palabras).value_counts().head(15) if len(todas_palabras) else pd.Series([], dtype=int)
        top_terminos_list = top_terminos.index.tolist()
        narrativa_1 = top_terminos_list[0] if len(top_terminos_list) else "N/A"
        
        # Top autor conversación (por Interacción)
        top_post_conv = df_conversacion.sort_values("Interacción", ascending=False).head(1)
        top_autor_conv = str(top_post_conv.iloc[0].get("Autor", "N/A")) if len(top_post_conv) else "N/A"
        
        # Temperatura (solo conversación)
        if pct_neg >= 40:
            temperatura = "🔴 Riesgo reputacional"
        elif pct_pos >= 60 and pct_neg < 25:
            temperatura = "🟢 Clima favorable"
        else:
            temperatura = "🟡 Mixto / neutro"
        
        # Amplificación (agregada por tweet original)
        amp_total_filas = int(len(df_amplificacion)) if df_amplificacion is not None else 0
        amp_total_eventos = 0
        amp_likes_total = 0
        amp_rt_puros_total = 0
        amp_quotes_total = 0
        
        if df_amplificacion is not None and not df_amplificacion.empty:
            amp_total_eventos = int(df_amplificacion["Ampl_total"].sum())
            amp_likes_total = int(df_amplificacion["Likesta"].sum())
            amp_rt_puros_total = int(df_amplificacion["RT_puros_en_rango"].sum())
            amp_quotes_total = int(df_amplificacion["Quotes_en_rango"].sum())
        
        # KPI layout (neófito-friendly)
        k1, k2, k3, k4, k5, k6 = st.columns(6)
        k1.metric("Conversación (posts)", f"{conv_total}")
        k2.metric("Temperatura", temperatura)
        k3.metric("% Negativo (conv.)", f"{pct_neg}%")
        k4.metric("Interacción (conv.)", f"{conv_interaccion_total}")
        k5.metric("Amplificación (eventos)", f"{amp_total_eventos}")
        k6.metric("Narrativa #1", narrativa_1)
        
        st.caption(
            f"Conversación: Pos {pct_pos}% | Neu {pct_neu}% | Neg {pct_neg}%. "
            f"Interacción promedio/post: {conv_interaccion_prom}. "
            f"Amplificación: RT puros={amp_rt_puros_total} | Quotes={amp_quotes_total} | Total={amp_total_eventos}."
        )
        
        # ---------------------------------------------------------
        # 5.2) Alertas (actualizadas con amplificación)
        # ---------------------------------------------------------
        alertas = []
        
        # Alerta reputacional (conv.)
        if pct_neg >= 40 and conv_total >= 10:
            alertas.append("⚠️ Conversación con componente negativo alto. Preparar mensaje de contención y aclaración con datos verificables.")
        elif pct_neg >= 30 and conv_total >= 10:
            alertas.append("🟡 Conversación con componente negativo relevante. Vigilar eventos gatillo y cuentas amplificadoras.")
        
        # Alerta por amplificación
        if amp_total_eventos >= 300:
            alertas.append("📣 Alta amplificación detectada (RT + Quotes). Revisar el TOP amplificados y activar monitoreo continuo.")
        elif amp_total_eventos >= 100:
            alertas.append("📢 Amplificación moderada. Confirmar si proviene de pocos posts “faro” o es dispersa.")
        
        # Alerta por muestra pequeña (conv.)
        if conv_total < 5:
            alertas.append("ℹ️ Muestra pequeña en conversación. Interpretar con cautela (señal temprana, no representativa).")
        
        if alertas:
            st.markdown("### 🚨 Alertas")
            for a in alertas:
                st.warning(a)
        
        st.caption(f"Método de sentimiento (conversación): {metodo_sent_conv}. En RT puros: sentimiento 1 vez por original (no se duplica).")
        
        # ---------------------------------------------------------
        # 5.3) INSUMOS para Gemini (incluye conversación + TOP amplificados)
        # ---------------------------------------------------------
        # Ejemplos top conversación (evita mandar todo)
        ejemplos_conv = (
            df_conversacion.sort_values("Interacción", ascending=False)
            .head(8)["Texto"]
            .apply(lambda t: (t[:240] + "…") if isinstance(t, str) and len(t) > 240 else t)
            .tolist()
        )
        
        # Ejemplos top amplificados (texto del tweet original amplificado)
        ejemplos_amp = []
        if df_amplificacion is not None and not df_amplificacion.empty:
            top_amp = df_amplificacion.sort_values("Ampl_total", ascending=False).head(8)
            for _, r in top_amp.iterrows():
                txt = r.get("Texto_base_original", "") or ""
                txt = (txt[:240] + "…") if isinstance(txt, str) and len(txt) > 240 else txt
                if txt:
                    ejemplos_amp.append(txt)
        
        payload = {
            "query": query,
            "time_range": time_range,
        
            "conversacion": {
                "volumen_posts": int(conv_total),
                "sentimiento_pct": {"positivo": pct_pos, "neutral": pct_neu, "negativo": pct_neg},
                "top_terminos": top_terminos_list[:10],
                "ejemplos_top_interaccion": ejemplos_conv,
            },
        
            "amplificacion": {
                "eventos_total": int(amp_total_eventos),
                "rt_puros_total": int(amp_rt_puros_total),
                "quotes_total": int(amp_quotes_total),
                "likes_total": int(amp_likes_total),
                "ejemplos_top_amplificados": ejemplos_amp[:8],
                "nota": "Amplificación agregada por tweet original (una fila por post original)."
            },
        
            "temperatura": temperatura,
            "nota_ubicacion": "Ubicación inferida desde perfil/bio; no es geolocalización exacta."
        }
        
        # Generar resumen (Gemini)
        st.markdown("## ⭐ Resumen ejecutivo")
        
        bullets_ia, gemini_status = resumen_ejecutivo_gemini(payload, debug=debug_gemini)
        
        if bullets_ia:
            st.caption(f"Generado con IA (Gemini). Estado: {gemini_status}")
            st.markdown(bullets_ia)
        else:
            st.caption(f"IA no disponible o falló. Estado: {gemini_status}. Mostrando resumen por reglas.")
        
            # Resumen por reglas (sin repetir números)
            narrativa_txt = ", ".join(top_terminos_list[:6]) if top_terminos_list else "sin términos dominantes claros"
            if pct_neg >= 40:
                riesgo_txt = "Riesgo reputacional alto: conversación con tono negativo predominante."
            elif pct_neg >= 30:
                riesgo_txt = "Riesgo reputacional moderado: negativos relevantes que pueden escalar con un evento gatillo."
            else:
                riesgo_txt = "Riesgo reputacional bajo en el periodo observado, sin señales fuertes de escalamiento."
        
            if amp_total_eventos >= 300:
                amp_txt = "Amplificación alta: pocos posts pueden estar actuando como “faro” y concentrando difusión."
            elif amp_total_eventos >= 100:
                amp_txt = "Amplificación moderada: revisar top amplificados para entender qué está empujando la conversación."
            else:
                amp_txt = "Amplificación baja o normal: difusión acotada en el periodo."
        
            oportunidad_txt = (
                "Oportunidad: publicar aclaración breve con datos verificables y enlace a información completa, "
                "y preparar respuestas estándar para preguntas recurrentes."
            )
        
            st.markdown(f"**Narrativa:** Se observa una conversación centrada en {narrativa_txt}.")
            st.markdown(f"**Riesgos:** {riesgo_txt} Además, {amp_txt}.")
            st.markdown(f"**Oportunidades:** {oportunidad_txt}")
        
        # Advertencia metodológica (1 sola vez)
        st.caption(
            "Advertencia metodológica: señal temprana basada en publicaciones públicas de X; sentimiento automatizado "
            "(IA/fallback) y ubicación inferida desde perfil/bio (no geolocalización exacta). No representa a toda la población."
        )
        
        # ---------------------------------------------------------
        # 5.4) Tablero visual (Plotly) — SOLO conversación para sentimiento
        # ---------------------------------------------------------
        st.markdown("## 📊 Tablero visual (mejorado)")
        
        if df_conversacion["Fecha"].isna().all():
            st.warning("No se pudo interpretar fechas para graficar tendencia.")
        else:
            df_conversacion["Día"] = df_conversacion["Fecha"].dt.date.astype(str)
        
            # 1) Volumen conversación por día
            vol_por_dia = df_conversacion.groupby("Día").size().reset_index(name="Volumen")
            fig_vol = px.line(vol_por_dia, x="Día", y="Volumen", markers=True, title="📈 Volumen de conversación por día (Originales + Quotes)")
            st.plotly_chart(fig_vol, use_container_width=True)
        
            # 2) Sentimiento (donut) — conversación
            sent_counts = df_conversacion["Sentimiento"].value_counts().reset_index()
            sent_counts.columns = ["Sentimiento", "Cantidad"]
            fig_sent = px.pie(sent_counts, names="Sentimiento", values="Cantidad", hole=0.45, title="🧁 Distribución de sentimiento (conversación)")
            st.plotly_chart(fig_sent, use_container_width=True)
            st.caption(f"Método de sentimiento (conversación): {metodo_sent_conv}.")
        
            # 3) Sentimiento por día (apilado) — conversación
            sent_por_dia = df_conversacion.groupby(["Día", "Sentimiento"]).size().reset_index(name="Cantidad")
            fig_sent_dia = px.bar(
                sent_por_dia, x="Día", y="Cantidad", color="Sentimiento",
                barmode="stack", title="📆 Sentimiento por día (conversación)"
            )
            st.plotly_chart(fig_sent_dia, use_container_width=True)
        
            # 4) Top términos — conversación
            if not top_terminos.empty:
                top_terminos_df = top_terminos.reset_index()
                top_terminos_df.columns = ["Término", "Frecuencia"]
                fig_terms = px.bar(
                    top_terminos_df, x="Frecuencia", y="Término", orientation="h",
                    title="🏷️ Top términos dominantes (conversación, limpio de stopwords)"
                )
                st.plotly_chart(fig_terms, use_container_width=True)
        
        # ✅ Guardamos payload por si luego quieres exportar
        st.session_state["payload_gemini"] = payload


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
        total_quotes = int(df_amplificacion["Quotes_en_rango"].sum()) if (df_amplificacion is not None and not df_amplificacion.empty) else 0
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
        
        pct_pos_amp, pct_neu_amp, pct_neg_amp = distribucion_amp_ponderada(df_amplificacion)
        
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
        def top_terminos_de_textos(lista_textos: list[str], top_n: int = 15):
            all_words = []
            for t in (lista_textos or []):
                all_words.extend(limpiar_texto(t))
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
            top_row = df_conversacion.sort_values("Interacción", ascending=False).head(1)
            top_autor = str(top_row.iloc[0].get("Autor", "N/A")) if len(top_row) else "N/A"
        else:
            top_autor = "N/A"
        
        # -----------------------------
        # 6) Mostrar KPIs (separados)
        # -----------------------------
        k1, k2, k3, k10, k12, k13 = st.columns(5)
        k1.metric("Conversación (posts)", f"{n_conversacion}")
        k2.metric("Temp. conversación", temp_conv)
        k3.metric("% Neg (conv)", f"{pct_neg_conv}%")
        k13.metric("% Pos (conv)", f"{pct_pos_conv}%")
        k10.metric("Interacción (conv)", f"{interaccion_conversacion}")
        k12.metric("Narrativa #1 (conv)", narrativa_conv_1)
        
        k4, k5, k6, k8, k9, k14 = st.columns(5)
        k4.metric("Amplificación total", f"{total_ampl}")
        k5.metric("Temp. amplificación", temp_amp)
        k6.metric("% Neg (amp)", f"{pct_neg_amp}%")
        k14.metric("% Pos (amp)", f"{pct_pos_amp}%")
        k8.metric("Quotes", f"{n_quotes}")
        k9.metric("RT puros", f"{n_rt_puros}")
    
        st.caption(
            f"Conv: Pos {pct_pos_conv}% | Neu {pct_neu_conv}% | Neg {pct_neg_conv}% — "
            f"Amp (ponderado): Pos {pct_pos_amp}% | Neu {pct_neu_amp}% | Neg {pct_neg_amp}%."
        )
        
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
            "nota_ubicacion": "Ubicación inferida desde perfil/bio; no es geolocalización exacta."
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
            if df_amplificacion is not None and not df_amplificacion.empty:
                # armamos una tabla con pesos para el donut
                tmp = df_amplificacion.copy()
                tmp["peso"] = pd.to_numeric(tmp["RT_puros_en_rango"], errors="coerce").fillna(0) + pd.to_numeric(tmp["Quotes_en_rango"], errors="coerce").fillna(0)
                sent_w = tmp.groupby("Sentimiento_dominante")["peso"].sum().reset_index()
                sent_w.columns = ["Sentimiento", "Peso"]
                fig_sent_amp = px.pie(sent_w, names="Sentimiento", values="Peso", hole=0.45, title="🧁 Sentimiento — Amplificación (ponderado)")
                st.plotly_chart(fig_sent_amp, use_container_width=True)
            else:
                st.info("Sin datos de amplificación.")
        
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
        
        # ------------------------------------------------------------
        # TABLA 1) TOP 10 — Tweets originales (no RT) dentro del rango
        # Ranking sugerido: Interacción = Likes + Retweets (del original)
        # ------------------------------------------------------------
        if not df_originales.empty:
            df_originales_rank = df_originales.sort_values("Interacción", ascending=False).copy()
        else:
            df_originales_rank = df_originales.copy()
        
        # Asegura URL para originales (ya la tienes como "URL" en PARTE 3)
        # Columnas: igual que tus tablas actuales + Abrir
        cols_top_originales = [
            "Autor", "Fecha", "Likes", "Retweets", "Interacción",
            "Sentimiento", "Ubicación inferida", "Confianza",
            "Texto", "Link"
        ]
        
        # Importante: df_originales puede no tener "Sentimiento" si en PARTE 4 solo lo calculaste en df_conversacion.
        # En ese caso, lo traemos desde df_conversacion (que incluye originales).
        if "Sentimiento" not in df_originales_rank.columns:
            if not df_conversacion.empty:
                sent_map = df_conversacion.set_index("tweet_id")["Sentimiento"].to_dict()
                df_originales_rank["Sentimiento"] = df_originales_rank["tweet_id"].map(sent_map)
        
        render_table(
            df_originales_rank,
            "1) 🔥 Top 10 — Posts originales (no RT)",
            cols=cols_top_originales,
            top=10
        )
        
        # ------------------------------------------------------------
        # TABLA 2) TODOS — Tweets originales (no RT) dentro del rango
        # ------------------------------------------------------------
        with st.expander("2) 📄 Ver TODOS los posts originales (no RT)"):
            render_table(
                df_originales_rank,  # ya rankeado; si prefieres por fecha, cambia aquí
                "2) 📄 Todos — Posts originales (no RT)",
                cols=cols_top_originales,
                top=None
            )
        
        # ------------------------------------------------------------
        # TABLA 3) TOP 10 — Amplificación (muestra el TWEET ORIGINAL)
        # Ranking: Ampl_total (RT puros + Quotes) en el rango
        # ------------------------------------------------------------
        if not df_amplificacion.empty:
            df_amp_rank = df_amplificacion.sort_values("Ampl_total", ascending=False).copy()
        else:
            df_amp_rank = df_amplificacion.copy()
        
        cols_top_amp = [
            "Fechaua",
            "Ampl_total", "RT_puros_en_rango", "Quotes_en_rango",
            "Likesta",
            "Sentimiento_dominante",
            "Ubicación_dominante", "Confianza_dominante",
            "Texto_original",
            "Link"
        ]
        
        render_table(
            df_amp_rank,
            "3) 📣 Top 10 — Amplificación (muestra el tweet ORIGINAL amplificado)",
            cols=cols_top_amp,
            top=10
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
        
        st.caption(
            "Nota: En Amplificación, se muestra el tweet ORIGINAL una sola vez por fila. "
            "Los RT puros y quotes se contabilizan en columnas (RT_puros_en_rango, Quotes_en_rango, Ampl_total). "
            "El botón 'Abrir' siempre abre el tweet ORIGINAL."
        )
