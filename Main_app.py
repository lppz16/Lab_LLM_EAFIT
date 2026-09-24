import html
import re
import time

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from groq import Groq
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import (
    cosine_similarity,
    euclidean_distances,
    manhattan_distances,
)

st.set_page_config(page_title="Laboratorio LLM con Groq", page_icon="🧪", layout="wide")

# ----------------------------------------------------------------------------
# Constantes
# ----------------------------------------------------------------------------
MODELOS_FALLBACK = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
]
EXCLUIR_EN_CHAT = ("whisper", "tts", "guard", "orpheus", "playai")

EMB_MODELS = {
    "paraphrase-multilingual-MiniLM-L12-v2 (multilingüe, 384d)": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "paraphrase-multilingual-mpnet-base-v2 (multilingüe, 768d)": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "all-MiniLM-L6-v2 (inglés, 384d, ligero)": "sentence-transformers/all-MiniLM-L6-v2",
}
ENCODINGS = ["o200k_harmony", "o200k_base", "cl100k_base"]


# ----------------------------------------------------------------------------
# Funciones auxiliares
# ----------------------------------------------------------------------------
@st.cache_data(ttl=600, show_spinner=False)
def listar_modelos(api_key: str) -> pd.DataFrame:
    client = Groq(api_key=api_key)
    filas = []
    for m in client.models.list().data:
        d = m.model_dump()
        filas.append(
            {
                "id": d.get("id"),
                "propietario": d.get("owned_by"),
                "activo": d.get("active"),
                "context_window": d.get("context_window"),
                "max_completion_tokens": d.get("max_completion_tokens"),
            }
        )
    return pd.DataFrame(filas).sort_values("id").reset_index(drop=True)


@st.cache_resource(show_spinner=False)
def get_encoding(nombre: str):
    import tiktoken

    return tiktoken.get_encoding(nombre)


@st.cache_resource(show_spinner="Cargando modelo de embeddings (la primera vez se descarga)…")
def cargar_embedder(nombre: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(nombre)


@st.cache_data(show_spinner="Calculando embeddings…")
def calcular_embeddings(textos: tuple, nombre: str) -> np.ndarray:
    modelo = cargar_embedder(nombre)
    return modelo.encode(list(textos), show_progress_bar=False)


def jaccard(a: str, b: str) -> float:
    A = set(re.findall(r"\w+", a.lower()))
    B = set(re.findall(r"\w+", b.lower()))
    return len(A & B) / len(A | B) if (A | B) else 0.0


def lineas(texto: str) -> list:
    return [l.strip() for l in texto.split("\n") if l.strip()]


def chat_once(client, model, messages, params):
    t0 = time.time()
    r = client.chat.completions.create(model=model, messages=messages, **params)
    return r.choices[0].message.content or "", r.usage, time.time() - t0


def stream_chat(client, model, messages, params, meta):
    t0 = time.time()
    stream = client.chat.completions.create(
        model=model, messages=messages, stream=True, **params
    )
    for chunk in stream:
        xg = getattr(chunk, "x_groq", None)
        if xg is not None and getattr(xg, "usage", None) is not None:
            meta["usage"] = xg.usage
        if chunk.choices:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content
    meta["tiempo"] = time.time() - t0


# ----------------------------------------------------------------------------
# Sidebar: API key + parámetros
# ----------------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Configuración")
    api_key = st.text_input("Groq API Key", type="password", placeholder="gsk_...")
    st.caption("Consíguela en console.groq.com. No se guarda en ningún lado.")

    df_modelos = None
    if api_key:
        try:
            df_modelos = listar_modelos(api_key)
        except Exception as e:
            st.error(f"No se pudo listar modelos: {e}")

    if df_modelos is not None:
        ids = [
            i for i in df_modelos["id"].tolist()
            if not any(x in i.lower() for x in EXCLUIR_EN_CHAT)
        ]
    else:
        ids = MODELOS_FALLBACK
    ids = sorted(ids, key=lambda i: (0 if "gpt-oss" in i else 1, i))

    st.subheader("Modelo y parámetros")
    modelo = st.selectbox("Modelo", ids, help="Los GPT (gpt-oss) aparecen primero.")
    temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.05)
    top_p = st.slider("Top-p", 0.0, 1.0, 1.0, 0.05)
    st.caption("Recomendación: ajusta temperature o top-p, no ambos a la vez.")
    max_tokens = st.slider("Máx. tokens de salida", 64, 8192, 1024, 64)
    seed = st.number_input("Seed (-1 = aleatoria)", value=-1, step=1)
    stop = st.text_input("Stop sequence (opcional)")
    reasoning = st.selectbox(
        "Reasoning effort (solo gpt-oss)", ["low", "medium", "high"], index=1
    )

    st.subheader("Embeddings")
    emb_label = st.selectbox("Modelo de embeddings", list(EMB_MODELS.keys()))
    emb_name = EMB_MODELS[emb_label]
    st.caption("Groq no ofrece endpoint de embeddings, por eso se calculan localmente.")

client = Groq(api_key=api_key) if api_key else None


def params_for(model: str) -> dict:
    p = {
        "temperature": temperature,
        "top_p": top_p,
        "max_completion_tokens": max_tokens,
    }
    if seed >= 0:
        p["seed"] = int(seed)
    if stop:
        p["stop"] = [stop]
    if "gpt-oss" in model:
        p["reasoning_effort"] = reasoning
    return p


def requiere_key() -> bool:
    if client is None:
        st.warning("Ingresa tu Groq API Key en la barra lateral para usar esta sección.")
        return False
    return True


# ----------------------------------------------------------------------------
# UI principal
# ----------------------------------------------------------------------------
st.title("🧪 Laboratorio de LLMs con Groq")
st.caption("Generación de texto, tokens, Bag of Words, similitud y embeddings.")

(
    tab_gen, tab_comp, tab_modelos, tab_tok, tab_bow, tab_sim, tab_emb,
) = st.tabs(
    [
        "💬 Generación",
        "🔀 Comparar modelos/temperatura",
        "🤖 Modelos",
        "🔤 Tokens e IDs",
        "🎒 Bag of Words",
        "📏 Similitud",
        "🧬 Embeddings",
    ]
)

# ------------------------------ Generación ----------------------------------
with tab_gen:
    st.subheader(f"Generación de texto · {modelo}")
    system_prompt = st.text_area(
        "System prompt", "Eres un asistente útil que responde en español.", height=80
    )
    user_prompt = st.text_area(
        "Prompt", "Explica qué es un token en un LLM con una analogía simple.", height=120
    )
    if st.button("Generar", type="primary", key="btn_gen"):
        if requiere_key():
            messages = []
            if system_prompt.strip():
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": user_prompt})
            meta = {}
            try:
                with st.chat_message("assistant"):
                    st.write_stream(
                        stream_chat(client, modelo, messages, params_for(modelo), meta)
                    )
                c1, c2, c3, c4 = st.columns(4)
                usage = meta.get("usage")
                c1.metric("Tiempo (s)", f"{meta.get('tiempo', 0):.2f}")
                c2.metric("Tokens prompt", getattr(usage, "prompt_tokens", "—"))
                c3.metric("Tokens salida", getattr(usage, "completion_tokens", "—"))
                c4.metric("Tokens total", getattr(usage, "total_tokens", "—"))
                with st.expander("Parámetros enviados"):
                    st.json(params_for(modelo))
            except Exception as e:
                st.error(f"Error en la API: {e}")

# ------------------------------ Comparación ---------------------------------
with tab_comp:
    st.subheader("Mismo prompt, varios modelos y temperaturas")
    modelos_sel = st.multiselect("Modelos", ids, default=ids[:2])
    temps_sel = st.multiselect(
        "Temperaturas", [0.0, 0.2, 0.5, 0.7, 1.0, 1.5, 2.0], default=[0.2, 1.0]
    )
    prompt_comp = st.text_area(
        "Prompt a comparar",
        "Escribe un eslogan creativo para una cafetería en Medellín.",
        key="prompt_comp",
    )
    if st.button("Ejecutar comparación", type="primary", key="btn_comp"):
        if requiere_key():
            combos = [(m, t) for m in modelos_sel for t in temps_sel]
            if not combos:
                st.info("Selecciona al menos un modelo y una temperatura.")
            elif len(combos) > 12:
                st.warning("Máximo 12 combinaciones por ejecución.")
            else:
                resultados = []
                barra = st.progress(0.0)
                for i, (m, t) in enumerate(combos, 1):
                    p = params_for(m)
                    p["temperature"] = t
                    try:
                        txt, usage, dt = chat_once(
                            client, m, [{"role": "user", "content": prompt_comp}], p
                        )
                        resultados.append(
                            {
                                "modelo": m,
                                "temperature": t,
                                "tiempo_s": round(dt, 2),
                                "tokens_salida": getattr(usage, "completion_tokens", None),
                                "respuesta": txt,
                            }
                        )
                    except Exception as e:
                        resultados.append(
                            {
                                "modelo": m,
                                "temperature": t,
                                "tiempo_s": None,
                                "tokens_salida": None,
                                "respuesta": f"ERROR: {e}",
                            }
                        )
                    barra.progress(i / len(combos))
                df_res = pd.DataFrame(resultados)
                st.dataframe(df_res.drop(columns=["respuesta"]))
                for r in resultados:
                    with st.expander(f"{r['modelo']} · T={r['temperature']}"):
                        st.write(r["respuesta"])

# ------------------------------ Modelos -------------------------------------
with tab_modelos:
    st.subheader("Modelos disponibles en Groq")
    if df_modelos is None:
        st.info(
            "Ingresa tu API key para ver la lista real de modelos de tu cuenta. "
            "Mientras tanto, estos son algunos habituales:"
        )
        st.write(MODELOS_FALLBACK)
    else:
        solo_gpt = st.checkbox("Mostrar solo GPT (gpt-oss)")
        dfm = (
            df_modelos[df_modelos["id"].str.contains("gpt-oss")]
            if solo_gpt
            else df_modelos
        )
        st.dataframe(dfm)
        if "context_window" in dfm and dfm["context_window"].notna().any():
            st.markdown("**Ventana de contexto por modelo**")
            st.bar_chart(dfm.set_index("id")["context_window"])

# ------------------------------ Tokens --------------------------------------
with tab_tok:
    st.subheader("Tokens y Token IDs")
    st.caption(
        "Se usa tiktoken. `o200k_harmony` es el tokenizador de los modelos gpt-oss."
    )
    texto_tok = st.text_area(
        "Texto", "Hola, ¿cómo estás? Los tokenizadores dividen el texto en subpalabras.",
        key="texto_tok",
    )
    enc_name = st.selectbox("Tokenizador", ENCODINGS, key="enc_name")
    try:
        enc = get_encoding(enc_name)
        ids_tok = enc.encode(texto_tok, disallowed_special=())
        bytes_tok = [enc.decode_single_token_bytes(i) for i in ids_tok]
        partes = [b.decode("utf-8", errors="replace") for b in bytes_tok]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Caracteres", len(texto_tok))
        c2.metric("Palabras", len(texto_tok.split()))
        c3.metric("Tokens", len(ids_tok))
        c4.metric(
            "Caracteres / token",
            f"{len(texto_tok) / len(ids_tok):.2f}" if ids_tok else "—",
        )

        colores = ["#ffd6d6", "#d6e8ff", "#d6ffd9", "#fff2b3", "#e6d6ff"]
        spans = []
        for i, p in enumerate(partes):
            txt = html.escape(p).replace("\n", "↵<br>").replace(" ", "&nbsp;")
            spans.append(
                f'<span style="background:{colores[i % 5]};color:#111;padding:2px 1px;'
                f'border-radius:3px;margin:1px;display:inline-block">{txt}</span>'
            )
        st.markdown("**Tokens coloreados**")
        st.markdown("".join(spans), unsafe_allow_html=True)

        st.markdown("**Tabla de tokens**")
        st.dataframe(
            pd.DataFrame(
                {
                    "pos": range(len(ids_tok)),
                    "token_id": ids_tok,
                    "token": [repr(p) for p in partes],
                    "bytes": [len(b) for b in bytes_tok],
                }
            )
        )
        st.markdown("**Lista de Token IDs**")
        st.code(str(ids_tok))

        st.markdown("**Comparación entre tokenizadores**")
        filas = []
        for n in ENCODINGS:
            try:
                filas.append(
                    {
                        "tokenizador": n,
                        "tokens": len(get_encoding(n).encode(texto_tok, disallowed_special=())),
                    }
                )
            except Exception:
                filas.append({"tokenizador": n, "tokens": "no disponible"})
        st.dataframe(pd.DataFrame(filas))

        st.markdown("**Decodificar IDs → texto**")
        ids_txt = st.text_input("IDs separados por coma", ", ".join(map(str, ids_tok[:5])))
        try:
            nums = [int(x) for x in ids_txt.split(",") if x.strip()]
            st.write(repr(enc.decode(nums)))
        except Exception as e:
            st.warning(f"No se pudo decodificar: {e}")
    except Exception as e:
        st.error(f"No se pudo cargar el tokenizador '{enc_name}': {e}")

# ------------------------------ Bag of Words --------------------------------
with tab_bow:
    st.subheader("Bag of Words")
    docs_bow = lineas(
        st.text_area(
            "Documentos (uno por línea)",
            "El gato duerme en el sofá\nEl perro juega en el jardín\nEl gato y el perro son amigos",
            height=120,
            key="docs_bow",
        )
    )
    c1, c2, c3 = st.columns(3)
    modo = c1.radio("Representación", ["Conteo (BoW)", "Binario", "TF-IDF"])
    ngram = c2.selectbox("N-gramas", ["Unigramas", "Unigramas + bigramas"])
    lower = c3.checkbox("Minúsculas", value=True)
    ng = (1, 1) if ngram == "Unigramas" else (1, 2)

    if docs_bow:
        try:
            if modo == "TF-IDF":
                vec = TfidfVectorizer(ngram_range=ng, lowercase=lower)
            else:
                vec = CountVectorizer(
                    ngram_range=ng, lowercase=lower, binary=(modo == "Binario")
                )
            X = vec.fit_transform(docs_bow)
            df_bow = pd.DataFrame(
                X.toarray(),
                columns=vec.get_feature_names_out(),
                index=[f"Doc {i+1}" for i in range(len(docs_bow))],
            )
            st.metric("Tamaño del vocabulario", df_bow.shape[1])
            st.dataframe(df_bow)
            st.markdown("**Términos más frecuentes (suma sobre documentos)**")
            st.bar_chart(df_bow.sum().sort_values(ascending=False).head(15))
        except ValueError as e:
            st.error(f"No se pudo vectorizar: {e}")

# ------------------------------ Similitud -----------------------------------
with tab_sim:
    st.subheader("Métricas de similitud entre dos textos")
    ta = st.text_area("Texto A", "El gato duerme en el sofá de la sala", key="sim_a")
    tb = st.text_area("Texto B", "Un gatito descansa sobre el sillón de la sala", key="sim_b")
    metodo = st.radio(
        "Representación vectorial",
        ["Bag of Words", "TF-IDF", "Embeddings"],
        horizontal=True,
    )
    try:
        if metodo == "Embeddings":
            X = calcular_embeddings((ta, tb), emb_name)
        else:
            vec = CountVectorizer() if metodo == "Bag of Words" else TfidfVectorizer()
            X = vec.fit_transform([ta, tb]).toarray()

        cos = cosine_similarity(X[:1], X[1:])[0, 0]
        euc = euclidean_distances(X[:1], X[1:])[0, 0]
        man = manhattan_distances(X[:1], X[1:])[0, 0]
        dot = float(np.dot(X[0], X[1]))
        jac = jaccard(ta, tb)

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Coseno", f"{cos:.4f}")
        c2.metric("Euclidiana", f"{euc:.4f}")
        c3.metric("Manhattan", f"{man:.4f}")
        c4.metric("Producto punto", f"{dot:.4f}")
        c5.metric("Jaccard (palabras)", f"{jac:.4f}")
        st.caption(
            "Coseno y Jaccard: más alto = más similar. Euclidiana y Manhattan son "
            "distancias: más bajo = más similar. Prueba con Bag of Words vs Embeddings "
            "para ver que los embeddings capturan significado aunque las palabras difieran."
        )
    except Exception as e:
        st.error(f"No se pudo calcular la similitud: {e}")

# ------------------------------ Embeddings ----------------------------------
with tab_emb:
    st.subheader("Embeddings")
    docs_emb = lineas(
        st.text_area(
            "Textos (uno por línea)",
            "El gato duerme en el sofá\nEl perro corre por el parque\n"
            "La inflación subió en el último trimestre\n"
            "Los precios al consumidor aumentaron\n"
            "El banco central subió las tasas de interés",
            height=140,
            key="docs_emb",
        )
    )
    if st.toggle("Calcular embeddings", key="emb_on"):
        if len(docs_emb) < 2:
            st.info("Ingresa al menos 2 textos.")
        else:
            E = calcular_embeddings(tuple(docs_emb), emb_name)
            etiquetas = [f"D{i+1}" for i in range(len(docs_emb))]
            st.metric("Dimensión del vector", E.shape[1])

            st.markdown("**Primeras 8 dimensiones de cada embedding**")
            st.dataframe(
                pd.DataFrame(
                    E[:, :8],
                    index=etiquetas,
                    columns=[f"dim_{i}" for i in range(8)],
                )
            )

            st.markdown("**Matriz de similitud coseno**")
            sim = cosine_similarity(E)
            fig = px.imshow(
                sim, x=etiquetas, y=etiquetas, text_auto=".2f",
                color_continuous_scale="Blues", zmin=0, zmax=1,
            )
            st.plotly_chart(fig)

            if len(docs_emb) >= 3:
                st.markdown("**Proyección 2D (PCA)**")
                pts = PCA(n_components=2).fit_transform(E)
                fig2 = px.scatter(
                    x=pts[:, 0], y=pts[:, 1], text=etiquetas, hover_name=docs_emb,
                )
                fig2.update_traces(textposition="top center", marker=dict(size=12))
                st.plotly_chart(fig2)

            st.markdown("**Búsqueda semántica**")
            consulta = st.text_input("Consulta", "aumento de precios en la economía")
            if consulta.strip():
                q = calcular_embeddings((consulta,), emb_name)[0]
                sims = cosine_similarity(q.reshape(1, -1), E)[0]
                ranking = (
                    pd.DataFrame({"texto": docs_emb, "similitud_coseno": sims})
                    .sort_values("similitud_coseno", ascending=False)
                    .reset_index(drop=True)
                )
                st.dataframe(ranking)
    else:
        st.caption("Activa el interruptor para calcular (la primera vez descarga el modelo).")
