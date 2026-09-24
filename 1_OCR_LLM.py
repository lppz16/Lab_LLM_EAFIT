import importlib
import json
import re
import time
from collections import Counter

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from openai import BadRequestError, OpenAI
from PIL import Image, ImageDraw, ImageOps
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="OCR + LLM", page_icon="📷", layout="wide")

# ----------------------------------------------------------------------------
# Constantes
# ----------------------------------------------------------------------------
PROVEEDORES = {
    "OpenAI": {
        "base_url": None,
        "fallback": ["gpt-4o-mini", "gpt-4.1-mini", "gpt-4o", "gpt-4.1"],
        "ayuda": "platform.openai.com/api-keys",
    },
    "Groq (gpt-oss)": {
        "base_url": "https://api.groq.com/openai/v1",
        "fallback": ["openai/gpt-oss-120b", "openai/gpt-oss-20b"],
        "ayuda": "console.groq.com/keys",
    },
}
EXCLUIR = (
    "whisper", "tts", "guard", "orpheus", "playai", "embedding", "moderation",
    "transcribe", "audio", "realtime", "image", "dall", "search", "instruct",
    "codex", "computer-use",
)
EMB_MODELS = {
    "paraphrase-multilingual-MiniLM-L12-v2 (multilingüe, ligero)": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "paraphrase-multilingual-mpnet-base-v2 (multilingüe, mejor)": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
}
IDIOMAS_OCR = {
    "Español": "es", "Inglés": "en", "Portugués": "pt",
    "Francés": "fr", "Italiano": "it", "Alemán": "de",
}
MODELOS_SPACY = {"es": "es_core_news_sm", "en": "en_core_web_sm"}
DESCARTABLES = (
    "temperature", "top_p", "seed", "presence_penalty",
    "frequency_penalty", "reasoning_effort",
)
CONECTORES = {
    "es": [
        "sin embargo", "por lo tanto", "además", "en consecuencia", "por ejemplo",
        "es decir", "no obstante", "asimismo", "por otro lado", "finalmente",
        "en conclusión", "por consiguiente", "de este modo", "en primer lugar",
        "por otra parte", "en resumen", "dado que", "debido a", "por eso",
    ],
    "en": [
        "however", "therefore", "moreover", "furthermore", "for example",
        "in addition", "consequently", "finally", "in conclusion", "thus",
        "as a result", "on the other hand", "in particular", "for instance",
        "in summary", "because", "since", "hence",
    ],
}
ESTILOS = {
    "Formal": (
        "Registro formal y profesional: prosa continua, tono objetivo e impersonal, "
        "vocabulario culto, sin coloquialismos ni contracciones. Estructura con "
        "introducción, desarrollo y conclusión. Evita listas y tablas."
    ),
    "Técnica": (
        "Enfoque técnico: terminología precisa, definición de conceptos clave, detalle "
        "de funcionamiento, ejemplos o pasos cuando aplique, y supuestos o limitaciones. "
        "Puedes usar subtítulos y listas con viñetas."
    ),
}
EXTENSIONES = {
    "Breve": "entre 150 y 250 palabras",
    "Media": "entre 300 y 500 palabras",
    "Extensa": "entre 600 y 900 palabras",
}
MAX_LADO = 2400


# ----------------------------------------------------------------------------
# LLM (OpenAI y Groq comparten la misma API compatible)
# ----------------------------------------------------------------------------
def crear_cliente(proveedor: str, api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key, base_url=PROVEEDORES[proveedor]["base_url"])


@st.cache_data(ttl=600, show_spinner=False)
def listar_modelos(proveedor: str, api_key: str) -> list:
    cli = crear_cliente(proveedor, api_key)
    ids = [m.id for m in cli.models.list().data]
    ids = [i for i in ids if not any(x in i.lower() for x in EXCLUIR)]
    if proveedor == "OpenAI":
        ids = [i for i in ids if i.startswith(("gpt", "o1", "o3", "o4"))]
    return sorted(ids, key=lambda i: (0 if "gpt" in i else 1, i))


def llamar_llm(cli, modelo, mensajes, params):
    """Llama al modelo; si rechaza algún parámetro (p. ej. temperature en modelos
    de razonamiento), lo quita y reintenta."""
    p = dict(params)
    quitados = []
    for _ in range(6):
        try:
            t0 = time.time()
            r = cli.chat.completions.create(model=modelo, messages=mensajes, **p)
            return r, p, quitados, time.time() - t0
        except BadRequestError as e:
            msg = str(e).lower()
            cand = [k for k in p if k in DESCARTABLES and k in msg]
            if not cand:
                raise
            for k in cand:
                p.pop(k)
                quitados.append(k)
    raise RuntimeError("No se pudo completar la solicitud con esos parámetros.")


def construir_mensajes(texto, estilo, extension, idioma, extra):
    sistema = (
        "Eres un asistente que recibe texto extraído de una imagen mediante OCR "
        "(puede contener errores de reconocimiento). Tu tarea es AMPLIAR ese "
        "contenido: explicarlo, contextualizarlo y organizarlo con claridad.\n"
        "Reglas:\n"
        "1. No inventes datos, cifras, nombres ni citas que no estén en el texto. "
        "Si una palabra parece un error del OCR, márcala con [?].\n"
        "2. Distingue lo que dice el texto de lo que agregas como contexto general.\n"
        f"3. Estilo: {ESTILOS[estilo]}\n"
        f"4. Extensión: {EXTENSIONES[extension]}.\n"
        f"5. Responde en {idioma}."
    )
    if extra.strip():
        sistema += f"\n6. Instrucción adicional del usuario: {extra.strip()}"
    usuario = (
        f'Texto extraído por OCR:\n"""\n{texto}\n"""\n\nAmplía este contenido.'
    )
    return [
        {"role": "system", "content": sistema},
        {"role": "user", "content": usuario},
    ]


# ----------------------------------------------------------------------------
# OCR
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Cargando EasyOCR (la primera vez descarga modelos)…")
def cargar_ocr(idiomas: tuple):
    import easyocr

    return easyocr.Reader(list(idiomas), gpu=False)


def preprocesar(img, gris, contraste, escala):
    im = img.convert("RGB")
    if escala != 1.0:
        im = im.resize((int(im.width * escala), int(im.height * escala)), Image.LANCZOS)
    if gris:
        im = ImageOps.grayscale(im).convert("RGB")
    if contraste:
        im = ImageOps.autocontrast(im)
    return im


def ejecutar_ocr(img, idiomas):
    reader = cargar_ocr(tuple(idiomas))
    return reader.readtext(np.array(img))  # [(bbox, texto, confianza), ...]


def dibujar_cajas(img, res):
    im = img.copy()
    d = ImageDraw.Draw(im)
    ancho = max(2, im.width // 400)
    for bbox, _, conf in res:
        pts = [(float(x), float(y)) for x, y in bbox]
        color = (0, 180, 0) if conf >= 0.6 else (230, 140, 0)
        d.line(pts + [pts[0]], fill=color, width=ancho)
    return im


# ----------------------------------------------------------------------------
# Procesamiento de texto
# ----------------------------------------------------------------------------
def limpiar_markdown(t: str) -> str:
    t = re.sub(r"```.*?```", " ", t, flags=re.S)
    t = re.sub(r"^\s*[-*_=|: ]{3,}\s*$", "", t, flags=re.M)
    t = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", t)
    t = re.sub(r"^\s{0,3}#{1,6}\s*", "", t, flags=re.M)
    t = re.sub(r"^\s*(?:[-*+•]|\d+[.)])\s+", "", t, flags=re.M)
    t = re.sub(r"[*_`>|]", "", t)
    return re.sub(r"[ \t]+", " ", t).strip()


def dividir_oraciones(t: str) -> list:
    partes = re.split(r"(?<=[.!?…])\s+|\n+", t)
    return [p.strip() for p in partes if len(p.strip().split()) >= 2]


def contar_silabas(palabra: str, lang: str) -> int:
    if lang == "es":
        total = 0
        for g in re.findall(r"[aeiouáéíóúü]+", palabra):
            n = 1
            for a, b in zip(g, g[1:]):
                if (a in "aeoáéó" and b in "aeoáéó") or a in "íú" or b in "íú":
                    n += 1
            total += n
        return max(total, 1)
    n = len(re.findall(r"[aeiouy]+", palabra))
    if palabra.endswith("e") and n > 1:
        n -= 1
    return max(n, 1)


def etiqueta_legibilidad(v: float) -> str:
    if v > 80:
        return "muy fácil"
    if v > 65:
        return "bastante fácil"
    if v > 55:
        return "normal"
    if v > 40:
        return "algo difícil"
    return "difícil"


# ----------------------------------------------------------------------------
# NLP: modelos cacheados
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Cargando spaCy…")
def cargar_spacy(lang: str):
    try:
        import spacy
    except Exception:
        return None
    nombre = MODELOS_SPACY[lang]
    try:
        return spacy.load(nombre)
    except OSError:
        try:
            from spacy.cli import download

            download(nombre)
            importlib.invalidate_caches()
            return spacy.load(nombre)
        except (Exception, SystemExit):
            return None


@st.cache_resource(show_spinner="Cargando modelo de embeddings…")
def cargar_embedder(nombre: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(nombre)


def embeber(oraciones: list, nombre: str) -> np.ndarray:
    return cargar_embedder(nombre).encode(
        oraciones, show_progress_bar=False, normalize_embeddings=True
    )


# ----------------------------------------------------------------------------
# Métricas
# ----------------------------------------------------------------------------
def medidas_texto(t: str, lang: str) -> dict:
    palabras = re.findall(r"\w+", t.lower())
    oraciones = dividir_oraciones(t)
    parrafos = [p for p in re.split(r"\n\s*\n", t) if p.strip()]
    n = max(len(palabras), 1)
    s = max(len(oraciones), 1)
    unicas = len(set(palabras))
    silabas = sum(contar_silabas(p, lang) for p in palabras)
    spw, wps = silabas / n, n / s
    if lang == "es":
        legib = 206.835 - 62.3 * spw - wps
        formula = "Szigriszt-Pazos (INFLESZ)"
    else:
        legib = 206.835 - 1.015 * wps - 84.6 * spw
        formula = "Flesch Reading Ease"
    top = Counter(p for p in palabras if len(p) > 3).most_common(10)
    return {
        "caracteres": len(t),
        "palabras": len(palabras),
        "oraciones": len(oraciones),
        "parrafos": max(len(parrafos), 1),
        "palabras_unicas": unicas,
        "ttr": unicas / n,
        "guiraud": unicas / np.sqrt(n),
        "long_media_palabra": float(np.mean([len(p) for p in palabras])) if palabras else 0.0,
        "palabras_por_oracion": wps,
        "silabas_por_palabra": spw,
        "legibilidad": float(legib),
        "formula_legibilidad": formula,
        "etiqueta_legibilidad": etiqueta_legibilidad(legib),
        "top_palabras": [[w, c] for w, c in top],
    }


def prof_token(t) -> int:
    d = 0
    while t.head.i != t.i and d < 100:
        t = t.head
        d += 1
    return d


def metricas_sintaxis(doc):
    oraciones = [
        s for s in doc.sents
        if len([t for t in s if not t.is_punct and not t.is_space]) >= 2
    ]
    if not oraciones:
        return None
    prof = [max(prof_token(t) for t in s) for s in oraciones]
    con_verbo = [any(t.pos_ in ("VERB", "AUX") for t in s) for s in oraciones]
    tokens = [t for t in doc if not t.is_punct and not t.is_space]
    contenido = [t for t in tokens if t.pos_ in ("NOUN", "VERB", "ADJ", "ADV", "PROPN")]
    sub = sum(1 for t in tokens if t.dep_.startswith(("acl", "advcl", "ccomp", "xcomp", "csubj")))
    pos = Counter(t.pos_ for t in tokens)
    prof_media = float(np.mean(prof))
    if 2 <= prof_media <= 7:
        prof_score = 1.0
    else:
        prof_score = max(0.0, 1 - min(abs(prof_media - 2), abs(prof_media - 7)) / 4)
    frac_verbo = float(np.mean(con_verbo))
    return {
        "oraciones_analizadas": len(oraciones),
        "tokens_por_oracion": len(tokens) / len(oraciones),
        "profundidad_media": prof_media,
        "profundidad_max": int(max(prof)),
        "pct_oraciones_con_verbo": frac_verbo * 100,
        "densidad_lexica": len(contenido) / max(len(tokens), 1),
        "subordinadas_por_oracion": sub / len(oraciones),
        "pos": dict(pos),
        "score": 100 * (0.6 * frac_verbo + 0.4 * prof_score),
    }


def gramatica_lt(texto: str, lang: str) -> dict:
    texto = texto[:18000]
    r = requests.post(
        "https://api.languagetool.org/v2/check",
        data={"text": texto, "language": "es" if lang == "es" else "en-US"},
        timeout=30,
    )
    r.raise_for_status()
    hallazgos = []
    for m in r.json().get("matches", []):
        regla = m.get("rule", {})
        cat = regla.get("category", {})
        ctx = m.get("context", {})
        off, ln = ctx.get("offset", 0), ctx.get("length", 0)
        hallazgos.append(
            {
                "categoria": cat.get("name", ""),
                "cat_id": cat.get("id", ""),
                "mensaje": m.get("message", ""),
                "fragmento": ctx.get("text", "")[off: off + ln],
                "sugerencia": (m.get("replacements") or [{}])[0].get("value", ""),
            }
        )
    criticos = [h for h in hallazgos if h["cat_id"] not in ("STYLE", "TYPOGRAPHY")]
    n_pal = max(len(re.findall(r"\w+", texto)), 1)
    por100 = 100 * len(criticos) / n_pal
    return {
        "errores": len(criticos),
        "avisos_estilo": len(hallazgos) - len(criticos),
        "errores_por_100_palabras": por100,
        "por_categoria": dict(Counter(h["categoria"] for h in hallazgos)),
        "hallazgos": hallazgos,
        "score": max(0.0, 100 - 5 * por100),
    }


def terminos(texto: str, nlp) -> set:
    if nlp is not None:
        return {
            t.lemma_.lower() for t in nlp(texto)
            if t.pos_ in ("NOUN", "PROPN", "NUM") and not t.is_stop and len(t.text) > 2
        }
    return {w for w in re.findall(r"\w+", texto.lower()) if len(w) > 4}


def metricas_semanticas(original, generado, emb_name, nlp):
    so = dividir_oraciones(limpiar_markdown(original)) or [original.strip()]
    sg = dividir_oraciones(limpiar_markdown(generado))
    if not sg or not original.strip():
        return None
    Eo, Eg = embeber(so, emb_name), embeber(sg, emb_name)
    sim_global = float(cosine_similarity(Eo.mean(0, keepdims=True), Eg.mean(0, keepdims=True))[0, 0])
    M = cosine_similarity(Eo, Eg)
    cobertura = M.max(axis=1)
    aporte_nuevo = float(np.mean(M.max(axis=0) < 0.5))
    To, Tg = terminos(limpiar_markdown(original), nlp), terminos(limpiar_markdown(generado), nlp)
    cob_terminos = len(To & Tg) / len(To) if To else None
    idx = M.argmax(axis=1)
    tabla = [
        {
            "linea_ocr": so[i],
            "oracion_mas_similar": sg[idx[i]],
            "similitud": round(float(M[i, idx[i]]), 3),
        }
        for i in range(len(so))
    ]
    cob_media = float(cobertura.mean())
    return {
        "similitud_global": sim_global,
        "cobertura_media": cob_media,
        "pct_lineas_cubiertas": float(np.mean(cobertura >= 0.5)) * 100,
        "pct_aporte_nuevo": aporte_nuevo * 100,
        "cobertura_terminos_clave": cob_terminos,
        "tabla": tabla,
        "score": 100 * float(np.clip(0.5 * sim_global + 0.5 * cob_media, 0, 1)),
    }


def metricas_coherencia(generado, emb_name, lang):
    sg = dividir_oraciones(limpiar_markdown(generado))
    if len(sg) < 3:
        return None
    E = embeber(sg, emb_name)
    local = np.array([float(E[i] @ E[i + 1]) for i in range(len(E) - 1)])
    c = E.mean(0)
    c = c / np.linalg.norm(c)
    glob = E @ c
    conj = [{w for w in re.findall(r"\w+", s.lower()) if len(w) > 3} for s in sg]
    lex = [len(a & b) / len(a | b) if (a | b) else 0.0 for a, b in zip(conj, conj[1:])]
    txt = generado.lower()
    n_conec = sum(len(re.findall(r"\b" + re.escape(c_) + r"\b", txt)) for c_ in CONECTORES[lang])
    n_pal = max(len(re.findall(r"\w+", txt)), 1)
    return {
        "coherencia_local": float(local.mean()),
        "coherencia_global": float(glob.mean()),
        "cohesion_lexica": float(np.mean(lex)),
        "conectores_por_100_palabras": 100 * n_conec / n_pal,
        "similitud_consecutivas": local.tolist(),
        "score": 100 * float(np.clip(0.5 * local.mean() + 0.5 * glob.mean(), 0, 1)),
    }


def juez_llm(cli, modelo, original, generado, estilo):
    prompt = (
        "Evalúa la RESPUESTA generada a partir del TEXTO ORIGINAL (extraído por OCR). "
        "Responde SOLO con un JSON válido, sin texto adicional, con enteros de 1 a 10 "
        "y un comentario breve:\n"
        '{"coherencia": n, "fluidez": n, "fidelidad": n, "completitud": n, '
        '"ajuste_estilo": n, "comentario": "..."}\n\n'
        "coherencia = ideas bien conectadas; fluidez = redacción natural; "
        "fidelidad = no inventa datos respecto al original; completitud = cubre lo del "
        f"original y lo amplía; ajuste_estilo = cumple el estilo pedido ({estilo}).\n\n"
        f'TEXTO ORIGINAL:\n"""\n{original}\n"""\n\nRESPUESTA:\n"""\n{generado}\n"""'
    )
    r, _, _, _ = llamar_llm(
        cli, modelo, [{"role": "user", "content": prompt}],
        {"temperature": 0, "max_completion_tokens": 2500},
    )
    txt = r.choices[0].message.content or ""
    m = re.search(r"\{.*\}", txt, re.S)
    if not m:
        raise ValueError("El juez no devolvió un JSON válido.")
    d = json.loads(m.group(0))
    out = {
        k: float(d.get(k, 0))
        for k in ("coherencia", "fluidez", "fidelidad", "completitud", "ajuste_estilo")
    }
    out["comentario"] = str(d.get("comentario", ""))
    return out


def seguro(fn, *args):
    try:
        return fn(*args)
    except Exception as e:
        return {"error": str(e)}


def ok(m) -> bool:
    return isinstance(m, dict) and "error" not in m


def aviso(m, etiqueta):
    if m is None:
        st.info(f"{etiqueta}: sin datos suficientes o desactivado.")
    elif isinstance(m, dict) and "error" in m:
        st.warning(f"{etiqueta}: {m['error']}")


def fmt(v, d=2):
    return "—" if v is None else f"{v:.{d}f}"


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Configuración")
    proveedor = st.selectbox("Proveedor del LLM", list(PROVEEDORES))
    api_key = st.text_input("API Key", type="password", placeholder="sk-... / gsk_...")
    st.caption(f"Consíguela en {PROVEEDORES[proveedor]['ayuda']}. No se guarda.")

    ids = PROVEEDORES[proveedor]["fallback"]
    if api_key:
        try:
            ids = listar_modelos(proveedor, api_key) or ids
        except Exception as e:
            st.error(f"No se pudo listar modelos: {e}")
    modelo_sel = st.selectbox("Modelo", ids)
    modelo_custom = st.text_input("Modelo personalizado (opcional)")
    modelo = modelo_custom.strip() or modelo_sel

    st.subheader("Parámetros del LLM")
    temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.05)
    top_p = st.slider("Top-p", 0.0, 1.0, 1.0, 0.05)
    max_tokens = st.slider("Máx. tokens de salida", 256, 8192, 2048, 128)
    presence = st.slider("Presence penalty", -2.0, 2.0, 0.0, 0.1)
    frecuencia = st.slider("Frequency penalty", -2.0, 2.0, 0.0, 0.1)
    seed = st.number_input("Seed (-1 = aleatoria)", min_value=-1, value=-1, step=1)
    reasoning = st.selectbox(
        "Reasoning effort (modelos de razonamiento)",
        ["auto (no enviar)", "low", "medium", "high"],
    )
    st.caption("Si el modelo no admite un parámetro, se omite automáticamente.")

    st.subheader("Estilo de la respuesta")
    estilo = st.radio("Tipo de respuesta", list(ESTILOS), horizontal=True)
    extension = st.select_slider("Nivel de ampliación", list(EXTENSIONES), value="Media")
    idioma = st.selectbox("Idioma de la respuesta", ["Español", "English"])
    extra = st.text_area("Instrucción adicional (opcional)", height=70)

    st.subheader("OCR")
    idiomas_sel = st.multiselect(
        "Idiomas del texto en la imagen", list(IDIOMAS_OCR), default=["Español", "Inglés"]
    )
    idiomas_ocr = [IDIOMAS_OCR[i] for i in idiomas_sel]
    gris = st.checkbox("Escala de grises", value=False)
    contraste = st.checkbox("Autocontraste", value=True)
    escala = st.select_slider("Escalado de la imagen", [1.0, 1.5, 2.0], value=1.0)

    st.subheader("Métricas")
    emb_label = st.selectbox("Modelo de embeddings", list(EMB_MODELS))
    emb_name = EMB_MODELS[emb_label]
    usar_lt = st.checkbox("Gramática con LanguageTool (API pública)", value=True)
    st.caption("Envía el texto generado a api.languagetool.org.")
    usar_juez = st.checkbox("Evaluación con LLM-juez (1 llamada extra)", value=False)


def construir_params() -> dict:
    p = {"temperature": temperature, "top_p": top_p, "max_completion_tokens": max_tokens}
    if presence != 0:
        p["presence_penalty"] = presence
    if frecuencia != 0:
        p["frequency_penalty"] = frecuencia
    if seed >= 0:
        p["seed"] = int(seed)
    if reasoning != "auto (no enviar)":
        p["reasoning_effort"] = reasoning
    return p


# ----------------------------------------------------------------------------
# Página principal
# ----------------------------------------------------------------------------
st.title("📷 OCR + LLM: de la imagen a una respuesta ampliada")
st.caption("Sube una imagen, extrae el texto con OCR, amplíalo con un LLM y evalúa la calidad del resultado.")

origen = st.radio("Origen de la imagen", ["Subir archivo", "Cámara"], horizontal=True)
if origen == "Subir archivo":
    archivo = st.file_uploader(
        "Imagen", type=["png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"]
    )
else:
    archivo = st.camera_input("Toma una foto")

if archivo is None:
    st.info("Sube o toma una imagen para comenzar.")
    st.stop()

clave_img = f"{archivo.name}-{archivo.size}"
if st.session_state.get("clave_img") != clave_img:
    for k in ("res_ocr", "img_ocr", "texto_ocr", "gen", "metricas"):
        st.session_state.pop(k, None)
    st.session_state["clave_img"] = clave_img

img = ImageOps.exif_transpose(Image.open(archivo))
if max(img.size) > MAX_LADO:
    img.thumbnail((MAX_LADO, MAX_LADO))
img_proc = preprocesar(img, gris, contraste, escala)

# ------------------------------ 1 y 2: OCR ----------------------------------
col_img, col_txt = st.columns(2)
with col_img:
    st.subheader("1 · Imagen y OCR")
    if st.button("🔍 Extraer texto (OCR)", type="primary"):
        if not idiomas_ocr:
            st.warning("Selecciona al menos un idioma de OCR.")
        else:
            try:
                with st.spinner("Ejecutando OCR…"):
                    res = ejecutar_ocr(img_proc, idiomas_ocr)
                st.session_state["res_ocr"] = res
                st.session_state["img_ocr"] = img_proc
                st.session_state["texto_ocr"] = "\n".join(t for _, t, _ in res)
                st.session_state.pop("gen", None)
                st.session_state.pop("metricas", None)
            except Exception as e:
                st.error(f"Error en el OCR: {e}")
    res = st.session_state.get("res_ocr")
    if res:
        st.image(
            dibujar_cajas(st.session_state["img_ocr"], res),
            caption="Verde: confianza ≥ 60% · Naranja: confianza baja",
        )
    else:
        st.image(img_proc, caption="Vista previa (con el preprocesamiento elegido)")

with col_txt:
    st.subheader("2 · Texto extraído (editable)")
    texto_ocr = st.text_area(
        "Texto OCR", key="texto_ocr", height=320, label_visibility="collapsed",
        placeholder="Aquí aparecerá el texto detectado. Puedes corregirlo antes de enviarlo al LLM.",
    )
    if res:
        confs = [float(c) for _, _, c in res]
        c1, c2, c3 = st.columns(3)
        c1.metric("Líneas detectadas", len(res))
        c2.metric("Confianza media", f"{np.mean(confs) * 100:.1f}%")
        c3.metric("Palabras", len(texto_ocr.split()))
        with st.expander("Detalle por línea"):
            st.dataframe(
                pd.DataFrame(
                    {"texto": [t for _, t, _ in res], "confianza": [round(float(c), 3) for _, _, c in res]}
                )
            )

# ------------------------------ 3: LLM --------------------------------------
st.divider()
st.subheader(f"3 · Ampliación con LLM · {modelo}")

if st.button("✨ Ampliar con LLM", type="primary"):
    if not api_key:
        st.warning("Ingresa tu API key en la barra lateral.")
    elif not texto_ocr.strip():
        st.warning("Primero extrae (o escribe) el texto a ampliar.")
    else:
        try:
            cli = crear_cliente(proveedor, api_key)
            mensajes = construir_mensajes(texto_ocr, estilo, extension, idioma, extra)
            with st.spinner("Generando respuesta…"):
                r, efectivos, quitados, dt = llamar_llm(cli, modelo, mensajes, construir_params())
            texto_gen = r.choices[0].message.content or ""
            if not texto_gen.strip():
                st.warning("La respuesta llegó vacía. Sube 'Máx. tokens' o baja el reasoning effort.")
            else:
                st.session_state["gen"] = {
                    "texto": texto_gen,
                    "original": texto_ocr,
                    "modelo": modelo,
                    "proveedor": proveedor,
                    "estilo": estilo,
                    "lang": "es" if idioma == "Español" else "en",
                    "tiempo": dt,
                    "prompt_tokens": getattr(r.usage, "prompt_tokens", None),
                    "completion_tokens": getattr(r.usage, "completion_tokens", None),
                    "params": efectivos,
                    "omitidos": quitados,
                }
                st.session_state.pop("metricas", None)
        except Exception as e:
            st.error(f"Error en la API: {e}")

gen = st.session_state.get("gen")
if gen:
    with st.container(border=True):
        st.markdown(gen["texto"])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Modelo", gen["modelo"])
    c2.metric("Tiempo (s)", f"{gen['tiempo']:.2f}")
    c3.metric("Tokens prompt", gen["prompt_tokens"] if gen["prompt_tokens"] is not None else "—")
    c4.metric("Tokens salida", gen["completion_tokens"] if gen["completion_tokens"] is not None else "—")
    if gen["omitidos"]:
        st.info("Parámetros no admitidos por este modelo (omitidos): " + ", ".join(gen["omitidos"]))
    with st.expander("Parámetros enviados"):
        st.json(gen["params"])
    st.download_button("⬇️ Descargar respuesta (.md)", gen["texto"], file_name="respuesta.md")

    # ------------------------------ 4: métricas -----------------------------
    st.divider()
    st.subheader("4 · Métricas del texto generado")

    if st.button("📊 Calcular métricas", type="primary"):
        lang = gen["lang"]
        limpio = limpiar_markdown(gen["texto"])
        with st.spinner("Calculando métricas (la primera vez carga modelos)…"):
            nlp = cargar_spacy(lang)
            doc = nlp(limpio) if nlp is not None else None
            M = {
                "texto": medidas_texto(limpio, lang),
                "texto_original": medidas_texto(limpiar_markdown(gen["original"]), lang),
                "sintaxis": seguro(metricas_sintaxis, doc) if doc is not None else {
                    "error": f"Modelo spaCy '{MODELOS_SPACY[lang]}' no disponible. "
                             f"Instálalo con: python -m spacy download {MODELOS_SPACY[lang]}"
                },
                "gramatica": seguro(gramatica_lt, limpio, lang) if usar_lt else None,
                "semantica": seguro(metricas_semanticas, gen["original"], gen["texto"], emb_name, nlp),
                "coherencia": seguro(metricas_coherencia, gen["texto"], emb_name, lang),
                "juez": None,
            }
            if usar_juez:
                if api_key:
                    M["juez"] = seguro(
                        juez_llm, crear_cliente(gen["proveedor"], api_key),
                        gen["modelo"], gen["original"], gen["texto"], gen["estilo"],
                    )
                else:
                    M["juez"] = {"error": "Se necesita la API key para el LLM-juez."}
        st.session_state["metricas"] = M

    M = st.session_state.get("metricas")
    if M:
        # ---- Resumen ----
        puntajes = {}
        if ok(M.get("coherencia")):
            puntajes["Coherencia"] = M["coherencia"]["score"]
        if ok(M.get("semantica")):
            puntajes["Semántica"] = M["semantica"]["score"]
        if ok(M.get("sintaxis")):
            puntajes["Sintaxis"] = M["sintaxis"]["score"]
        if ok(M.get("gramatica")):
            puntajes["Gramática"] = M["gramatica"]["score"]
        puntajes["Legibilidad"] = float(np.clip(M["texto"]["legibilidad"], 0, 100))

        cr, cm = st.columns([1, 1])
        with cr:
            claves = list(puntajes)
            vals = [puntajes[k] for k in claves]
            fig = go.Figure(
                go.Scatterpolar(r=vals + vals[:1], theta=claves + claves[:1], fill="toself")
            )
            fig.update_layout(
                polar=dict(radialaxis=dict(range=[0, 100])),
                showlegend=False, margin=dict(t=30, b=30),
            )
            st.plotly_chart(fig)
        with cm:
            st.markdown("**Puntajes (0–100)**")
            cols = st.columns(len(puntajes))
            for col, (k, v) in zip(cols, puntajes.items()):
                col.metric(k, f"{v:.0f}")
            st.caption(
                "Son indicadores heurísticos para comparar corridas (otro modelo, "
                "temperatura o estilo), no una verdad absoluta."
            )
            st.download_button(
                "⬇️ Descargar métricas (.json)",
                json.dumps(
                    M, ensure_ascii=False, indent=2,
                    default=lambda o: o.item() if hasattr(o, "item") else str(o),
                ),
                file_name="metricas.json",
            )

        t_med, t_sin, t_gra, t_sem, t_coh, t_jue, t_info = st.tabs(
            ["📏 Medidas", "🧩 Sintaxis", "✍️ Gramática", "🧠 Semántica",
             "🔗 Coherencia", "⚖️ LLM-juez", "ℹ️ Cómo se calcula"]
        )

        # ---- Medidas ----
        with t_med:
            m, mo = M["texto"], M["texto_original"]
            a, b, c, d = st.columns(4)
            a.metric("Caracteres", m["caracteres"])
            b.metric("Palabras", m["palabras"])
            c.metric("Oraciones", m["oraciones"])
            d.metric("Párrafos", m["parrafos"])
            a, b, c, d = st.columns(4)
            a.metric("Palabras únicas", m["palabras_unicas"])
            b.metric("TTR (diversidad léxica)", fmt(m["ttr"], 3))
            c.metric("Índice de Guiraud", fmt(m["guiraud"], 2))
            d.metric("Long. media de palabra", fmt(m["long_media_palabra"], 2))
            a, b, c = st.columns(3)
            a.metric("Palabras por oración", fmt(m["palabras_por_oracion"], 1))
            b.metric("Sílabas por palabra", fmt(m["silabas_por_palabra"], 2))
            c.metric(m["formula_legibilidad"], fmt(m["legibilidad"], 1), m["etiqueta_legibilidad"], delta_color="off")

            st.markdown("**Texto OCR vs. respuesta**")
            comp = pd.DataFrame(
                {
                    "OCR": [mo["palabras"], mo["oraciones"], mo["palabras_unicas"], round(mo["ttr"], 3)],
                    "Respuesta": [m["palabras"], m["oraciones"], m["palabras_unicas"], round(m["ttr"], 3)],
                },
                index=["palabras", "oraciones", "palabras únicas", "TTR"],
            )
            st.dataframe(comp)
            if mo["palabras"]:
                st.metric("Factor de ampliación", f"{m['palabras'] / mo['palabras']:.1f}×")
            if m["top_palabras"]:
                st.markdown("**Palabras más frecuentes (>3 letras)**")
                st.bar_chart(pd.Series({w: c_ for w, c_ in m["top_palabras"]}))

        # ---- Sintaxis ----
        with t_sin:
            s = M.get("sintaxis")
            if ok(s):
                a, b, c = st.columns(3)
                a.metric("Oraciones analizadas", s["oraciones_analizadas"])
                b.metric("Tokens por oración", fmt(s["tokens_por_oracion"], 1))
                c.metric("Densidad léxica", fmt(s["densidad_lexica"], 2))
                a, b, c = st.columns(3)
                a.metric("Profundidad media del árbol", fmt(s["profundidad_media"], 2))
                b.metric("Profundidad máxima", s["profundidad_max"])
                c.metric("Oraciones con verbo", f"{s['pct_oraciones_con_verbo']:.0f}%")
                st.metric("Subordinadas por oración", fmt(s["subordinadas_por_oracion"], 2))
                dpos = pd.DataFrame({"POS": list(s["pos"]), "n": list(s["pos"].values())}).sort_values("n", ascending=False)
                st.plotly_chart(px.bar(dpos, x="POS", y="n", title="Distribución de categorías gramaticales"))
            else:
                aviso(s, "Sintaxis")

        # ---- Gramática ----
        with t_gra:
            g = M.get("gramatica")
            if ok(g):
                a, b, c = st.columns(3)
                a.metric("Errores", g["errores"])
                b.metric("Errores / 100 palabras", fmt(g["errores_por_100_palabras"], 2))
                c.metric("Avisos de estilo", g["avisos_estilo"])
                if g["por_categoria"]:
                    st.bar_chart(pd.Series(g["por_categoria"]))
                if g["hallazgos"]:
                    st.dataframe(pd.DataFrame(g["hallazgos"]).drop(columns=["cat_id"]))
                else:
                    st.success("LanguageTool no encontró problemas.")
            else:
                aviso(g, "Gramática")

        # ---- Semántica ----
        with t_sem:
            sm = M.get("semantica")
            if ok(sm):
                a, b, c = st.columns(3)
                a.metric("Similitud global (coseno)", fmt(sm["similitud_global"], 3))
                b.metric("Cobertura semántica media", fmt(sm["cobertura_media"], 3))
                c.metric("Líneas OCR cubiertas (≥0.5)", f"{sm['pct_lineas_cubiertas']:.0f}%")
                a, b = st.columns(2)
                a.metric("Oraciones con aporte nuevo", f"{sm['pct_aporte_nuevo']:.0f}%")
                ct = sm["cobertura_terminos_clave"]
                b.metric("Términos clave conservados", "—" if ct is None else f"{ct * 100:.0f}%")
                st.markdown("**Mejor correspondencia de cada línea OCR en la respuesta**")
                st.dataframe(pd.DataFrame(sm["tabla"]))
            else:
                aviso(sm, "Semántica")

        # ---- Coherencia ----
        with t_coh:
            co = M.get("coherencia")
            if ok(co):
                a, b, c, d = st.columns(4)
                a.metric("Coherencia local", fmt(co["coherencia_local"], 3))
                b.metric("Coherencia global", fmt(co["coherencia_global"], 3))
                c.metric("Cohesión léxica", fmt(co["cohesion_lexica"], 3))
                d.metric("Conectores / 100 palabras", fmt(co["conectores_por_100_palabras"], 2))
                dfl = pd.DataFrame(
                    {"par": range(1, len(co["similitud_consecutivas"]) + 1),
                     "similitud": co["similitud_consecutivas"]}
                )
                st.plotly_chart(
                    px.line(dfl, x="par", y="similitud", markers=True,
                            title="Similitud entre oraciones consecutivas")
                )
            else:
                aviso(co, "Coherencia (requiere al menos 3 oraciones)")

        # ---- Juez ----
        with t_jue:
            j = M.get("juez")
            if ok(j):
                cols = st.columns(5)
                for col, k in zip(cols, ["coherencia", "fluidez", "fidelidad", "completitud", "ajuste_estilo"]):
                    col.metric(k.replace("_", " ").capitalize(), f"{j[k]:.0f}/10")
                st.write(j["comentario"])
            else:
                aviso(j, "LLM-juez")

        # ---- Explicación ----
        with t_info:
            st.markdown(
                """
- **Medidas:** conteos básicos, TTR y Guiraud (diversidad léxica), y legibilidad
  (Szigriszt-Pazos para español, Flesch para inglés; el conteo de sílabas es aproximado).
- **Sintaxis (spaCy):** profundidad del árbol de dependencias, % de oraciones con verbo,
  densidad léxica y subordinadas. Puntaje = 60% oraciones con verbo + 40% profundidad en rango 2–7.
- **Gramática (LanguageTool):** errores por 100 palabras, sin contar avisos de estilo/tipografía.
  Puntaje = 100 − 5 × errores por 100 palabras.
- **Semántica (embeddings):** coseno entre los centroides del OCR y la respuesta, y cobertura
  (para cada línea OCR, la oración más parecida en la respuesta). Puntaje = 50% similitud + 50% cobertura.
- **Coherencia (embeddings):** similitud entre oraciones consecutivas (local) y frente al tema
  general (global), más cohesión léxica y conectores discursivos. Puntaje = promedio local/global.
- **LLM-juez:** el mismo modelo califica de 1 a 10; tiene sesgos, úsalo como referencia.
- En estilo técnico es normal una legibilidad baja; compara siempre entre corridas similares.
"""
            )
