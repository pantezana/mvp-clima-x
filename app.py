import streamlit as st
import tweepy
import re
import pandas as pd
import requests
import time
import plotly.express as px
import json
import ast
from datetime import datetime, timedelta

st.set_page_config(page_title="MVP Clima en X", layout="wide")
st.title("🖥️ MVP – Clima del Tema en X")

bearer_token = st.secrets["X_BEARER_TOKEN"]
client = tweepy.Client(bearer_token=bearer_token)

st.success("Conectado a X correctamente ✅")

query = st.text_input("Palabras clave / hashtags")

time_range = st.selectbox(
    "Rango temporal",
    ["24 horas", "48 horas","72 horas", "7 días", "30 días"]
)

# Límite de publicaciones a consultar (control de cuota)
limite_opcion = st.selectbox(
    "Límite de publicaciones a consultar (control de cuota X)",
    ["50", "100", "200", "500", "1000", "Sin límite (hasta donde llegue X)"],
    index=2  # 200 por defecto (ajústalo si quieres)
)

max_posts = None if "Sin límite" in limite_opcion else int(limite_opcion)

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

def _extract_gemini_text(data: dict) -> str:
    """Concatena todas las parts text."""
    try:
        cand = data.get("candidates", [])[0]
        parts = cand.get("content", {}).get("parts", [])
        texts = []
        for p in parts:
            t = p.get("text", "")
            if t:
                texts.append(t)
        return "\n".join(texts).strip()
    except Exception:
        return ""

def _strip_code_fences(text: str) -> str:
    """Quita ```json ... ``` o ``` ... ``` si existen."""
    text = text.strip()
    # Quitar bloque inicial ```xxx
    text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
    # Quitar cierre ```
    text = re.sub(r"\s*```$", "", text)
    return text.strip()

def _try_parse_json(text: str):
    """Intenta parsear JSON incluso si viene envuelto en texto."""
    if not text:
        return None

    text = _strip_code_fences(text)

    # 1) Intento directo
    try:
        return json.loads(text)
    except Exception:
        pass

    # 2) Intento: extraer el primer bloque { ... }
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass

    return None

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
            tweets_data, users_by_id = fetch_tweets_paginado(
                client=client,
                query=query,
                start_time=start_time,
                max_posts=max_posts,
                tweet_fields=["created_at", "public_metrics", "author_id"],
                expansions=["author_id"],
                user_fields=["username", "name", "location", "description"]
            )

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
        
        if tweets_data:
            data = []
            for t in tweets_data:
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
                    "Likes": (t.public_metrics or {}).get("like_count", 0),
                    "Retweets": (t.public_metrics or {}).get("retweet_count", 0),
                    "Ubicación inferida": ubicacion,
                    "Confianza": confianza,
                    "Fuente ubic.": fuente
                })
        
            df = pd.DataFrame(data)
        else:
            st.warning("No se encontraron publicaciones para ese criterio o rango seleccionado")
            st.stop()


        st.markdown("## 🧠 ANALISIS Y RESULTADOS")
            
        # --- Preparación de texto
            
        # Stopwords básicas en español (MVP)
        stopwords = set([
            "de","la","que","el","en","y","a","los","del","se","las","por","un","para","con",
            "no","una","su","al","lo","como","más","pero","sus","le","ya","o","este","sí",
            "porque","esta","entre","cuando","muy","sin","sobre"
        ])
            
        def limpiar_texto(texto):
            palabras = re.findall(r"\b[a-záéíóúñ]+\b", texto)
            return [p for p in palabras if p not in stopwords and len(p) > 3]

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
            
        # 1) Intentamos con Hugging Face (IA)
        sent_hf = []
        score_hf = []
            
        for txt in df["Texto"].tolist():
            s, sc = sentimiento_hf(txt)
            sent_hf.append(s)
            score_hf.append(sc)
            
        df["Sentimiento_HF"] = sent_hf
        df["Score_HF"] = score_hf
            
        # 2) Si Hugging Face falla, usamos el plan B (léxico)
        df["Sentimiento_Lex"] = df["Texto"].apply(calcular_sentimiento)
            
        # 3) Sentimiento final:
        # - Si HF dio respuesta: usamos HF
        # - Si HF no dio: usamos Lex
        df["Sentimiento"] = df["Sentimiento_HF"].fillna(df["Sentimiento_Lex"])

        # Informar método usado
        metodo_sent = "IA (Hugging Face)" if df["Sentimiento_HF"].notna().any() else "Léxico (fallback)"
                        
        # ============================================================
        # ✅ BLOQUE UNIFICADO (KPI + Resumen ejecutivo + Gráficos + Tabla)            
        # - df armado con columnas: Texto, Fecha, Likes, Retweets, Autor, URL, Ubicación inferida...
        # - df["Sentimiento"] ya calculado (HF + fallback)      
        # ============================================================
            
        # Asegurar tipos
        df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")
        df["Likes"] = pd.to_numeric(df["Likes"], errors="coerce").fillna(0)
        df["Retweets"] = pd.to_numeric(df["Retweets"], errors="coerce").fillna(0)
        df["Interacción"] = df["Likes"] + df["Retweets"]
            
        total = len(df)
        pct_pos = round((df["Sentimiento"] == "Positivo").mean() * 100, 1) if total else 0
        pct_neu = round((df["Sentimiento"] == "Neutral").mean() * 100, 1) if total else 0
        pct_neg = round((df["Sentimiento"] == "Negativo").mean() * 100, 1) if total else 0
        interaccion_total = int(df["Interacción"].sum()) if total else 0
        interaccion_prom = round(df["Interacción"].mean(), 2) if total else 0
            
        # Narrativas dominantes (top términos)
        todas_palabras = []
        for t in df["Texto"].str.lower().tolist():
            todas_palabras.extend(limpiar_texto(t))
        top_terminos = pd.Series(todas_palabras).value_counts().head(15)
        top_terminos_list = top_terminos.index.tolist()
        narrativa_1 = top_terminos_list[0] if len(top_terminos_list) else "N/A"
            
        # Top post influyente
        top_post = df.sort_values("Interacción", ascending=False).head(1)
        if len(top_post) > 0:
            top_autor = str(top_post.iloc[0].get("Autor", "N/A"))
            top_int = int(top_post.iloc[0].get("Interacción", 0))
        else:
            top_autor, top_int = "N/A", 0
            
        # Temperatura (semáforo simple)
        if pct_neg >= 40:
            temperatura = "🔴 Riesgo reputacional"
        elif pct_pos >= 60 and pct_neg < 25:
            temperatura = "🟢 Clima favorable"
        else:
            temperatura = "🟡 Mixto / neutro"

        # Armamos insumos compactos (evita enviar 50 textos completos)
 
        ejemplos = (
            df.sort_values("Interacción", ascending=False)
                .head(10)["Texto"]
                .apply(lambda t: (t[:240] + "…") if isinstance(t, str) and len(t) > 240 else t)
                .tolist()
        )

        payload = {
            "query": query,
            "time_range": time_range,
            "volumen": int(total),
            "sentimiento_pct": {"positivo": pct_pos, "neutral": pct_neu, "negativo": pct_neg},
            "temperatura": temperatura,
            "top_terminos": top_terminos_list[:10],
            "ejemplos_top_interaccion": ejemplos,
            "nota_ubicacion": "Ubicación inferida desde perfil/bio; no es geolocalización exacta."
        }

        # ✅ Regla simple: si hay muy pocos posts, Gemini suele dar salida pobre.
        # En ese caso saltamos directo al resumen por reglas.
       
        bullets_ia, gemini_status = resumen_ejecutivo_gemini(payload, debug=debug_gemini)

        # ─────────────────────────────
        # 🔥 TOP POSTS + DETALLE (compacto)
        # ─────────────────────────────
            
        # Top 10 posts por interacción
        top_posts = df.sort_values("Interacción", ascending=False).head(10).copy()
        top_posts["Link"] = top_posts["URL"].apply(lambda u: f'<a href="{u}" target="_blank">Abrir</a>' if u else "")
            
        st.markdown("### 🔥 Top 10 posts por interacción (Likes + Retweets)")
        st.markdown(
            top_posts[["Autor", "Fecha", "Likes", "Retweets", "Interacción", "Texto", "Link"]]
            .to_html(escape=False, index=False),
            unsafe_allow_html=True
        )
                
        # Tabla completa en expander (optimiza espacio)
        with st.expander("📄 Ver tabla completa de resultados (detalle)"):
            df_full = df.copy()
            df_full["Link"] = df_full["URL"].apply(lambda u: f'<a href="{u}" target="_blank">Abrir</a>' if u else "")
            st.markdown(
                df_full[["Autor", "Fecha", "Likes", "Retweets", "Sentimiento", "Ubicación inferida", "Confianza", "Texto", "Link"]]
                .to_html(escape=False, index=False),
                unsafe_allow_html=True
            )
         st.caption("Nota: la ubicación NO es exacta; es una inferencia basada en 'location' del perfil y/o bio. Úsala solo como aproximación.")
            
        # ─────────────────────────────
        # 🧮 PANEL EJECUTIVO (KPI + Alertas)
        # ─────────────────────────────
        st.markdown("## 🧾 Panel ejecutivo")
            
        k1, k2, k3, k4, k5, k6 = st.columns(6)
        k1.metric("Volumen", f"{total}")
        k2.metric("Temperatura", temperatura)
        k3.metric("% Negativo", f"{pct_neg}%")
        k4.metric("Interacción", f"{interaccion_total}")
        k5.metric("Top autor", top_autor)
        k6.metric("Narrativa #1", narrativa_1)
            
        st.caption(
            f"Detalle rápido: Pos {pct_pos}% | Neu {pct_neu}% | Neg {pct_neg}%. "
            f"Interacción promedio/post: {interaccion_prom}."
        )
            
        # Alertas (reglas MVP)
        alertas = []
        if pct_neg >= 40:
            alertas.append("⚠️ Alto componente negativo. Priorizar aclaraciones con datos verificables y mensajes de contención.")
        if interaccion_total >= 500 and total >= 10:
            alertas.append("📣 Alta interacción total: posible amplificación/viralización. Vigilar fuentes y evolución del volumen.")
        if alertas:
            st.markdown("### 🚨 Alertas")
            for a in alertas:
                st.warning(a)
        st.caption(f"Método de sentimiento: {metodo_sent}. Score HF (0–1) es confianza aproximada cuando hay IA.")
        # ─────────────────────────────
        # 🧠 RESUMEN EJECUTIVO (sin repetir números)
        # ─────────────────────────────
        st.markdown("## ⭐ Resumen ejecutivo")

        if bullets_ia:
            st.caption("Generado con IA (Gemini). Si falla, se usa el resumen por reglas.")
            st.markdown(bullets_ia)      
        else:
            st.caption("IA no disponible o falló. Mostrando resumen por reglas.")
                                    
            # Riesgos / oportunidades (reglas simples, sin repetir métricas)
            riesgo_bullets = []
            if pct_neg >= 40:
                riesgo_bullets.append("Riesgo reputacional alto: conversación con tono negativo predominante.")
            elif pct_neg >= 30:
                riesgo_bullets.append("Riesgo reputacional moderado: presencia relevante de negativos que puede escalar con eventos gatillo.")
            else:
                riesgo_bullets.append("Riesgo reputacional bajo en el periodo observado, sin señales fuertes de escalamiento.")
                
            oportunidad_bullets = []
            if pct_pos > pct_neg:
                oportunidad_bullets.append("Clima con espacio para reforzar narrativa: responder con información clara, oportuna y verificable.")
            else:
                oportunidad_bullets.append("Oportunidad de aclaración: reducir ambigüedad con FAQ, cifras y vocería consistente.")
                
            # Mensajes sugeridos (framing informativo, no propaganda)
            mensajes = [
                "Mensaje sugerido: 'Compartimos información verificable y actualizada sobre el tema, con fuentes y fechas claras.'",
                "Mensaje sugerido: 'Si tienes dudas, revisa este resumen: qué se sabe, qué no se sabe aún y próximos hitos.'",
            ]
            if pct_neg >= 30:
                mensajes.append("Mensaje sugerido: 'Entendemos la preocupación. Aclaramos los puntos críticos y cómo se atenderán.'")
                
            # Qué monitorear mañana (operativo)
            monitoreo = [
                "Monitorear si aparece un nuevo hashtag o término dominante (cambio de agenda).",
                "Monitorear si sube la proporción de negativos o se concentra en una narrativa específica.",
                "Monitorear cuentas/post con alta interacción (posibles amplificadores).",
                "Monitorear señales regionales (ubicación inferida) solo como indicio, no como dato duro.",
            ]
                
            # Construir bullets (8–12)
            bullets = []
            bullets.append(f"Se detecta una conversación con narrativa dominante alrededor de: {', '.join(top_terminos_list[:6]) if top_terminos_list else 'sin términos dominantes claros'}.")
            bullets.extend(riesgo_bullets)
            bullets.extend(oportunidad_bullets)
            bullets.extend(mensajes[:2])
            bullets.append("Acción táctica: preparar 3 respuestas estándar (datos, procesos, próximos pasos) y mantener consistencia.")
            bullets.append("Acción táctica: si el volumen aumenta, publicar una aclaración breve + enlace a información completa.")
            bullets.extend(monitoreo[:3])
                
            # Mostrar en pantalla (máximo 12)
            for b in bullets[:12]:
                st.markdown(f"- {b}")
            
            # Advertencia metodológica (una sola vez, corta)
            st.caption(
                "Advertencia metodológica: señal temprana basada en publicaciones públicas de X; sentimiento automatizado (IA/fallback) "
                "y ubicación inferida desde perfil/bio (no geolocalización exacta). No representa a toda la población."
            )
            
            # ─────────────────────────────
            # 📊 TABLERO VISUAL (Plotly)
            # ─────────────────────────────
            st.markdown("## 📊 Tablero visual")
            
            if df["Fecha"].isna().all():
                st.warning("No se pudo interpretar fechas para graficar tendencia.")
            else:
                df["Día"] = df["Fecha"].dt.date.astype(str)
            
                # 1) Volumen por día
                vol_por_dia = df.groupby("Día").size().reset_index(name="Volumen")
                fig_vol = px.line(vol_por_dia, x="Día", y="Volumen", markers=True, title="📈 Volumen de publicaciones por día")
                st.plotly_chart(fig_vol, use_container_width=True)
            
                # 2) Sentimiento (donut)
                sent_counts = df["Sentimiento"].value_counts().reset_index()
                sent_counts.columns = ["Sentimiento", "Cantidad"]
                fig_sent = px.pie(sent_counts, names="Sentimiento", values="Cantidad", hole=0.45, title="🧁 Distribución de sentimiento")
                st.plotly_chart(fig_sent, use_container_width=True)
                st.caption(f"Método de sentimiento: {metodo_sent}. Score HF (0–1) es confianza aproximada cuando hay IA.")
            
                # 3) Sentimiento por día (apilado)
                sent_por_dia = df.groupby(["Día", "Sentimiento"]).size().reset_index(name="Cantidad")
                fig_sent_dia = px.bar(
                    sent_por_dia, x="Día", y="Cantidad", color="Sentimiento",
                    barmode="stack", title="📆 Sentimiento por día (barras apiladas)"
                )
                st.plotly_chart(fig_sent_dia, use_container_width=True)
            
                # 4) Top términos
                top_terminos_df = top_terminos.reset_index()
                top_terminos_df.columns = ["Término", "Frecuencia"]
                fig_terms = px.bar(
                    top_terminos_df, x="Frecuencia", y="Término", orientation="h",
                    title="🏷️ Top términos dominantes (limpio de stopwords)"
                )
                st.plotly_chart(fig_terms, use_container_width=True)
            
         





