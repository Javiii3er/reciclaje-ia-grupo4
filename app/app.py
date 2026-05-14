"""
app.py — UI/UX Híbrida · Marvin + Orbitron · Sin emojis extra
Proyecto 01: Reciclaje Inteligente · Grupo #4 · UMG 2026
"""

import sys
import os
import time
import threading

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import streamlit as st
from PIL import Image
import cv2
import numpy as np
import queue as _queue
import av
import torch
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode

torch.set_num_threads(1)

from predict import predict_top3, predict_frame, load_model
from model import CLASS_THRESHOLDS, get_prediction_info

# ── CONFIGURACIÓN DE PÁGINA ──────────────────────────────────────────────────
st.set_page_config(
    page_title="EcoRecicla IA | Grupo #4 UMG",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700;900&family=Inter:wght@300;400;600;700;900&display=swap');

/* ── BASE ────────────────────────────────────────────────────────────────── */
html, body, [class*="st-"] {
    font-family: 'Inter', sans-serif;
    font-size: clamp(1.1rem, 1.3vw, 1.35rem);
}

.stApp {
    background-color: #0b1120;
    background-image:
        linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f2027 100%),
        linear-gradient(rgba(52,211,153,0.04) 1px, transparent 1px),
        linear-gradient(90deg, rgba(52,211,153,0.04) 1px, transparent 1px);
    background-size: 100% 100%, 48px 48px, 48px 48px;
    min-height: 100vh;
}

.block-container {
    padding-top: 0.15rem !important;
    padding-bottom: 0.4rem !important;
    padding-left: 1.5rem !important;
    padding-right: 1.5rem !important;
    max-width: 100% !important;
}
[data-testid="stHeader"] { height: 0 !important; min-height: 0 !important; }
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stMain"]::-webkit-scrollbar { display: none; }
[data-testid="stMain"] { scrollbar-width: none; }

/* ── NAVBAR ──────────────────────────────────────────────────────────────── */
.navbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: linear-gradient(90deg, #0f172a, #1e293b);
    padding: 0.55rem 1.4rem;
    border: 1px solid rgba(255,255,255,0.08);
    border-top: 2px solid #34d399;
    margin-bottom: 0.3rem;
    border-radius: 10px;
    box-shadow: 0 0 28px rgba(52,211,153,0.08);
}
.navbar-left {
    font-family: 'Orbitron', sans-serif;
    font-size: clamp(1.4rem, 1.8vw, 2rem);
    font-weight: 900;
    letter-spacing: 2px;
    color: #34d399;
    text-shadow: 0 0 18px rgba(52,211,153,0.5);
}
.navbar-center { display: flex; gap: 0.6rem; align-items: center; }
.navbar-center span {
    font-size: clamp(0.95rem, 1.1vw, 1.2rem);
    padding: 0.38rem 0.9rem;
    background: rgba(255,255,255,0.1);
    border-radius: 8px;
    color: #cbd5e1;
    white-space: nowrap;
}
.navbar-right {
    font-size: clamp(0.82rem, 0.92vw, 1rem);
    color: #475569;
    letter-spacing: 0.5px;
}

/* ── HERO ────────────────────────────────────────────────────────────────── */
.hero { text-align: center; padding: 0.3rem 1rem 0.25rem; }
.hero h1 {
    font-family: 'Orbitron', sans-serif;
    font-size: clamp(1.8rem, 2.8vw, 3.2rem);
    font-weight: 900;
    letter-spacing: 4px;
    background: linear-gradient(90deg, #34d399, #60a5fa, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    line-height: 1.1;
    display: inline-block;
    filter: drop-shadow(0 0 14px rgba(52,211,153,0.35));
}
.hero p {
    color: #94a3b8;
    font-size: clamp(0.88rem, 1vw, 1.08rem);
    margin-top: 0.2rem;
    letter-spacing: 0.5px;
}

/* ── SEPARADOR HERO ──────────────────────────────────────────────────────── */
.hero-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(52,211,153,0.3), rgba(96,165,250,0.3), transparent);
    margin: 0.3rem 0 0.35rem;
    border: none;
}

/* ── PANEL DE TRABAJO ────────────────────────────────────────────────────── */
.work-panel {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 16px;
    padding: 0.7rem 1.4rem 0.8rem;
    backdrop-filter: blur(12px);
    box-shadow: 0 4px 40px rgba(0,0,0,0.35);
    margin: 0.2rem 0 0.3rem;
}

/* ── SECTION TITLE ───────────────────────────────────────────────────────── */
.section-title {
    font-size: clamp(1.1rem, 1.3vw, 1.4rem);
    font-weight: 700;
    color: #e2e8f0;
    padding-bottom: 0.3rem;
    margin-bottom: 0.45rem;
    border-bottom: 1px solid rgba(255,255,255,0.1);
    letter-spacing: 0.3px;
}
.section-subtitle {
    font-size: clamp(0.85rem, 0.95vw, 1.05rem);
    color: #64748b;
    margin-bottom: 0.45rem;
}

/* ── RESULTADO ───────────────────────────────────────────────────────────── */
.result-box {
    border-radius: 12px;
    padding: 1rem 1.3rem;
    margin-top: 0.7rem;
    border-left: 5px solid;
}
.result-top  { background: rgba(52,211,153,0.12);  border-color: #34d399; }
.result-info { background: rgba(96,165,250,0.12);  border-color: #60a5fa; }
.result-warn { background: rgba(251,191,36,0.12);  border-color: #fbbf24; }
.result-box h3 {
    color: #f1f5f9;
    margin: 0 0 0.3rem;
    font-size: clamp(1.15rem, 1.45vw, 1.55rem);
    font-weight: 700;
    line-height: 1.25;
}
.result-box p {
    color: #cbd5e1;
    margin: 0;
    font-size: clamp(0.95rem, 1.05vw, 1.12rem);
    line-height: 1.55;
}
.result-box .conf-value {
    color: #f1f5f9;
    font-weight: 700;
    font-size: clamp(1.05rem, 1.2vw, 1.3rem);
    margin: 0.15rem 0 0;
    display: block;
}
.confidence-bar-bg {
    background: rgba(255,255,255,0.1);
    border-radius: 8px;
    height: 12px;
    margin-top: 0.5rem;
    overflow: hidden;
}
.confidence-bar {
    height: 100%;
    border-radius: 8px;
    background: linear-gradient(90deg, #34d399, #60a5fa);
    transition: width 0.5s ease;
}
.result-label {
    font-size: clamp(0.72rem, 0.78vw, 0.84rem);
    font-weight: 600;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 0.2rem;
}
.alt-list { margin: 0.5rem 0 0; padding: 0; list-style: none; }
.alt-list li {
    font-size: clamp(0.9rem, 1vw, 1.05rem);
    color: #94a3b8;
    padding: 0.2rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.05);
}
.alt-list li:last-child { border-bottom: none; }
.alt-list li strong { color: #cbd5e1; }

/* ── IMAGENES / VIDEO ────────────────────────────────────────────────────── */
[data-testid="stImage"] img {
    max-height: calc(100vh - 390px) !important;
    min-height: 200px;
    object-fit: contain;
    width: 100% !important;
    border-radius: 12px;
    border: 1px solid rgba(255,255,255,0.08);
}

/* Eliminar espacio gris interno del camera_input */
[data-testid="stCameraInput"] {
    background: transparent !important;
    padding: 0 !important;
}
[data-testid="stCameraInput"] > div,
[data-testid="stCameraInput"] > div > div {
    padding: 0 !important;
    margin: 0 !important;
    background: transparent !important;
    border: none !important;
    gap: 0 !important;
}
[data-testid="stCameraInput"] video,
[data-testid="stCameraInput"] canvas {
    border-radius: 12px !important;
    width: 100% !important;
    max-height: calc(100vh - 390px) !important;
    min-height: 200px;
    display: block;
    object-fit: cover;
    border: 1px solid rgba(255,255,255,0.08) !important;
}
[data-testid="stCameraInput"] button {
    margin-top: 0 !important;
    border-radius: 0 0 12px 12px !important;
}

/* ── BADGES AGG 164-2021 ─────────────────────────────────────────────────── */
.badges-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.3rem;
    margin-top: 0.3rem;
}
.sidebar-badge {
    display: block;
    padding: 0.32rem 0.8rem;
    border-radius: 999px;
    font-size: clamp(0.8rem, 0.88vw, 0.95rem);
    font-weight: 700;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.badge-blanco   { background: #e2e8f0; color: #1e293b !important; border: 1px solid #94a3b8; }
.badge-verde    { background: #065f46; color: #6ee7b7 !important; }
.badge-negro    { background: #111827; color: #9ca3af !important; border: 1px solid #374151; }

/* ── BOTONES ─────────────────────────────────────────────────────────────── */
button {
    min-height: 50px !important;
    font-size: clamp(0.92rem, 1vw, 1.1rem) !important;
    padding: 0.75rem 1.3rem !important;
    font-weight: 600 !important;
}

/* ── RADIO BUTTONS ───────────────────────────────────────────────────────── */
div[role="radiogroup"] {
    display: flex !important;
    flex-wrap: nowrap !important;
    gap: 0.6rem !important;
    justify-content: center !important;
}
div[role="radiogroup"] label {
    border-radius: 12px !important;
    padding: 0.5rem 1.2rem !important;
    font-weight: 600 !important;
    font-size: clamp(1rem, 1.2vw, 1.3rem) !important;
    transition: background 0.2s !important;
    color: #f1f5f9 !important;
    white-space: nowrap !important;
}
div[role="radiogroup"] label p { color: #f1f5f9 !important; margin: 0 !important; }

/* ── CAPTION / INFO ──────────────────────────────────────────────────────── */
[data-testid="stCaptionContainer"] p {
    font-size: clamp(0.88rem, 0.95vw, 1rem) !important;
    color: #64748b !important;
}
[data-testid="stInfo"] {
    font-size: clamp(0.9rem, 1vw, 1.05rem) !important;
}

/* ── TABS ────────────────────────────────────────────────────────────────── */
[data-testid="stTabs"] button {
    font-size: clamp(0.9rem, 1vw, 1.08rem) !important;
    font-weight: 600 !important;
    padding: 0.5rem 1rem !important;
}

/* ── EXPANDER ────────────────────────────────────────────────────────────── */
[data-testid="stExpander"] summary p {
    font-size: clamp(0.9rem, 1vw, 1.05rem) !important;
    font-weight: 600 !important;
}

/* ── SLIDER ──────────────────────────────────────────────────────────────── */
[data-testid="stSlider"] { padding: 0.8rem 0; }

/* ── FOOTER ──────────────────────────────────────────────────────────────── */
.footer-title {
    font-size: clamp(0.78rem, 0.86vw, 0.92rem);
    font-weight: 700;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 0.3rem;
}
.footer-divider {
    height: 1px;
    background: rgba(255,255,255,0.07);
    margin: 0.35rem 0;
    border: none;
}

/* ── RESULTADO: reducir espaciado interno ────────────────────────────────── */
.result-box { padding: 0.7rem 1.1rem; margin-top: 0.5rem; }
</style>
""", unsafe_allow_html=True)

# ── MODELO ────────────────────────────────────────────────────────────────────
DOUBT_THRESHOLD = 70.0

@st.cache_resource
def get_stable_model():
    path = os.path.join(os.path.dirname(__file__), '..', 'models', 'modelo_reciclaje.pth')
    if os.path.exists(path):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model  = load_model(path)
        model.to(device).eval()
        return model, device
    return None, 'cpu'

_INFER_SEM = threading.Semaphore(1)

def _es_dudoso(clase: str, confianza: float) -> bool:
    return confianza < CLASS_THRESHOLDS.get(clase, DOUBT_THRESHOLD)

def _titulo_clase(clase: str) -> str:
    info = get_prediction_info(clase)
    if info['categoria'] == 'Plástico':
        return f"Plástico — {clase.replace('_', ' ').title()}"
    return clase.replace('_', ' ').title()

# ── PROCESADOR WEBRTC ─────────────────────────────────────────────────────────
_COLOR_BGR = {
    'Blanco': (220, 220, 220),
    'Verde':  (0,   200,  80),
    'Negro':  (50,   50,  50),
}

class ReciclajeProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = None
        self._frame_count = 0
        self._lock = threading.Lock()
        self._last = {'clase': '...', 'confianza': 0, 'color': 'Gris', 'emoji': '', 'consejo': ''}
        self.result_queue = _queue.Queue(maxsize=1)

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        h, w = img.shape[:2]

        rect_size = min(400, min(h, w) - 20)
        x1, y1 = (w - rect_size) // 2, (h - rect_size) // 2
        x2, y2 = x1 + rect_size, y1 + rect_size

        self._frame_count += 1
        if self._frame_count % 15 == 0 and self.model is not None:
            try:
                roi = img[y1:y2, x1:x2]
                rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                pil = Image.fromarray(cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA))
                result = predict_frame(pil, self.model)
                with self._lock:
                    self._last = result
                try:
                    self.result_queue.put_nowait(result)
                except _queue.Full:
                    pass
            except Exception:
                pass

        with self._lock:
            last = dict(self._last)

        color_bgr = _COLOR_BGR.get(last.get('color', 'Gris'), (100, 100, 100))
        overlay = img.copy()
        cv2.rectangle(img,     (x1, y1),     (x2, y2),     (255, 255, 255), 2)
        cv2.rectangle(img,     (x1-1, y1-1), (x2+1, y2+1), color_bgr,       1)
        cv2.rectangle(overlay, (0, 0),        (w, 60),       color_bgr,      -1)
        cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)
        cv2.putText(img,
                    f"{last.get('clase','').upper()}  {last.get('confianza',0)}%",
                    (20, 42), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ── CARGA DEL MODELO ─────────────────────────────────────────────────────────
with st.spinner("Cargando Motor de IA..."):
    MODEL, DEVICE = get_stable_model()

# ── NAVBAR ───────────────────────────────────────────────────────────────────
device_label = "GPU · CUDA" if (MODEL is not None and str(DEVICE) != 'cpu') else "CPU"
model_status = "Activo" if MODEL is not None else "No cargado"

st.markdown(f"""
<div class="navbar">
    <div class="navbar-left">🌿 EcoRecicla IA</div>
    <div class="navbar-center">
        <span>Modelo: {model_status}</span>
        <span>{device_label}</span>
        <span>Acc. train: 99.44%</span>
    </div>
    <div class="navbar-right">UMG · Grupo 4 · 2026</div>
</div>
""", unsafe_allow_html=True)

# ── HERO ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🌿 EcoRecicla IA</h1>
  <p>Clasificación inteligente de residuos · MobileNetV2 · GPU Acelerado</p>
</div>
<hr class="hero-divider">
""", unsafe_allow_html=True)

# ── SELECTOR DE MODO ──────────────────────────────────────────────────────────
mode = st.radio(
    "Modo de clasificación",
    options=["CÁMARA — Tomar Foto", "VIDEO — Tiempo Real", "ARCHIVO — Imagen / Video"],
    horizontal=True,
    label_visibility="collapsed"
)

st.markdown('<div class="work-panel">', unsafe_allow_html=True)

# ── MODO CÁMARA ───────────────────────────────────────────────────────────────
if "CÁMARA" in mode:
    st.markdown('<div class="section-title">Cámara · Captura Instantánea</div>', unsafe_allow_html=True)
    col_v, col_r = st.columns([2, 1])

    with col_v:
        st.markdown('<div class="section-subtitle">Apunta al objeto y presiona Tomar Foto.</div>', unsafe_allow_html=True)
        camera_photo = st.camera_input("Captura de cámara", label_visibility="collapsed")

    with col_r:
        if camera_photo:
            img = Image.open(camera_photo)
            if MODEL is not None:
                with _INFER_SEM:
                    res, disclaimer = predict_top3(img, model=MODEL)
                top = res[0]
                es_dudoso  = _es_dudoso(top['clase'], top['confianza'])
                titulo_res = _titulo_clase(top['clase'])
                sub_res    = f"{top['confianza']}%"
                msg_duda   = ""

                if es_dudoso and len(res) > 1:
                    alt        = res[1]
                    titulo_res = f"¿{_titulo_clase(top['clase'])}?"
                    sub_res    = "Confianza baja"
                    msg_duda   = (f"Parece <strong>{top['clase'].replace('_',' ')}</strong>, "
                                  f"pero también podría ser "
                                  f"<strong>{alt['clase'].replace('_',' ')}</strong>.")

                color_box = "result-top" if not es_dudoso else "result-warn"
                st.markdown(f"""
<div class="result-box {color_box}">
    <div class="result-label">{"Clasificación" if not es_dudoso else "Incertidumbre"}</div>
    <h3>{titulo_res}</h3>
    <span class="conf-value">{sub_res} confianza</span>
    <div class="confidence-bar-bg"><div class="confidence-bar" style="width:{top['confianza']}%"></div></div>
</div>
<div class="result-box result-info">
    <div class="result-label">{"Recomendación" if es_dudoso else "Contenedor: " + top['color']}</div>
    <p>{msg_duda if es_dudoso else top['consejo']}</p>
</div>
""", unsafe_allow_html=True)

                if len(res) > 1:
                    items = "".join(
                        f"<li><strong>{r['clase'].replace('_',' ').title()}</strong> &nbsp;—&nbsp; {r['confianza']}%</li>"
                        for r in res[1:]
                    )
                    st.markdown(f"""
<div class="result-box" style="border-color:rgba(255,255,255,0.15);background:rgba(255,255,255,0.04);margin-top:0.6rem">
    <div class="result-label">Otras posibilidades</div>
    <ul class="alt-list">{items}</ul>
</div>""", unsafe_allow_html=True)

                st.markdown(f'<div class="result-box result-warn" style="margin-top:0.6rem"><p style="font-size:clamp(0.82rem,0.9vw,0.95rem)">{disclaimer}</p></div>', unsafe_allow_html=True)
            else:
                st.error("Modelo no disponible.")
        else:
            st.info("Esperando captura de imagen...")

# ── MODO VIDEO ────────────────────────────────────────────────────────────────
elif "VIDEO" in mode:
    st.markdown('<div class="section-title">Video · Detección en Tiempo Real</div>', unsafe_allow_html=True)
    col_v, col_r = st.columns([2, 1])

    with col_v:
        st.markdown('<div class="section-subtitle">Detección continua via WebRTC — funciona en cualquier navegador.</div>', unsafe_allow_html=True)
        ctx = webrtc_streamer(
            key="reciclaje-video",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=ReciclajeProcessor,
            async_processing=True,
            media_stream_constraints={"video": True, "audio": False},
        )

    with col_r:
        res_ph = st.empty()
        st.markdown('<hr class="footer-divider">', unsafe_allow_html=True)
        st.info("Coloca el residuo dentro del cuadro central para mejorar la precisión.")

    if MODEL is None:
        st.error("Modelo no disponible.")
    elif ctx.video_processor:
        ctx.video_processor.model = MODEL
        if ctx.state.playing:
            deadline = time.time() + 21600
            while ctx.state.playing and time.time() < deadline:
                proc = ctx.video_processor
                if proc is None:
                    time.sleep(0.03)
                    continue
                try:
                    rd        = proc.result_queue.get(timeout=0.1)
                    conf      = rd.get('confianza', 0)
                    clase     = rd.get('clase', '')
                    dudoso    = _es_dudoso(clase, conf)
                    titulo    = f"¿{_titulo_clase(clase)}? · Incertidumbre" if dudoso else _titulo_clase(clase)
                    color_box = "result-top" if not dudoso else "result-warn"
                    res_ph.markdown(f"""
<div class="result-box {color_box}" style="margin-top:0">
    <div class="result-label">Detección en tiempo real</div>
    <h3>{titulo}</h3>
    <span class="conf-value">{conf}% confianza</span>
    <div class="confidence-bar-bg"><div class="confidence-bar" style="width:{conf}%"></div></div>
</div>
<div class="result-box result-info" style="margin-top:0.5rem">
    <div class="result-label">{"Sugerencia" if dudoso else "Contenedor: " + rd.get('color','')}</div>
    <p>{"La IA tiene dudas. Intenta centrar mejor el objeto." if dudoso else rd.get('consejo','')}</p>
</div>
""", unsafe_allow_html=True)
                except _queue.Empty:
                    pass
                time.sleep(0.03)

# ── MODO ARCHIVO ──────────────────────────────────────────────────────────────
else:
    st.markdown('<div class="section-title">Archivo · Análisis Offline</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Sube una imagen o un video para que la IA lo analice.</div>', unsafe_allow_html=True)
    tab_img, tab_vid = st.tabs(["Imagen", "Video"])

    with tab_img:
        col_v, col_r = st.columns([2, 1])
        with col_v:
            uploaded_img = st.file_uploader(
                "Arrastra o selecciona una imagen",
                type=["jpg", "jpeg", "png", "webp"], key="img_upload"
            )
            if uploaded_img:
                img = Image.open(uploaded_img)
                st.image(img, use_column_width=True, caption="Imagen cargada")

        with col_r:
            if uploaded_img and MODEL is not None:
                with _INFER_SEM:
                    res, disclaimer = predict_top3(img, model=MODEL)
                top = res[0]
                es_dudoso  = _es_dudoso(top['clase'], top['confianza'])
                titulo_res = _titulo_clase(top['clase'])
                sub_res    = f"{top['confianza']}%"
                msg_duda   = ""

                if es_dudoso and len(res) > 1:
                    alt        = res[1]
                    titulo_res = f"¿{_titulo_clase(top['clase'])}?"
                    sub_res    = "Confianza baja"
                    msg_duda   = (f"Parece <strong>{top['clase'].replace('_',' ')}</strong>, "
                                  f"pero también podría ser "
                                  f"<strong>{alt['clase'].replace('_',' ')}</strong>.")

                color_box = "result-top" if not es_dudoso else "result-warn"
                st.markdown(f"""
<div class="result-box {color_box}">
    <div class="result-label">{"Clasificación" if not es_dudoso else "Incertidumbre"}</div>
    <h3>{titulo_res}</h3>
    <span class="conf-value">{sub_res} confianza</span>
    <div class="confidence-bar-bg"><div class="confidence-bar" style="width:{top['confianza']}%"></div></div>
</div>
<div class="result-box result-info" style="margin-top:0.5rem">
    <div class="result-label">{"Recomendación" if es_dudoso else "Contenedor: " + top['color']}</div>
    <p>{msg_duda if es_dudoso else top['consejo']}</p>
</div>
""", unsafe_allow_html=True)

                items = "".join(
                    f"<li><strong>{r['clase'].replace('_',' ').title()}</strong> &nbsp;—&nbsp; {r['confianza']}%</li>"
                    for r in res
                )
                st.markdown(f"""
<div class="result-box" style="border-color:rgba(255,255,255,0.15);background:rgba(255,255,255,0.04);margin-top:0.6rem">
    <div class="result-label">Top 3 resultados</div>
    <ul class="alt-list">{items}</ul>
</div>""", unsafe_allow_html=True)
                st.markdown(f'<div class="result-box result-warn" style="margin-top:0.6rem"><p style="font-size:clamp(0.82rem,0.9vw,0.95rem)">{disclaimer}</p></div>', unsafe_allow_html=True)

    with tab_vid:
        col_v, col_r = st.columns([2, 1])
        with col_v:
            uploaded_vid = st.file_uploader(
                "Arrastra o selecciona un video",
                type=["mp4", "avi", "mov", "mkv"], key="vid_upload"
            )
            if uploaded_vid:
                tmp_path = os.path.join(os.path.dirname(__file__), "_tmp_video.mp4")
                with open(tmp_path, "wb") as f:
                    f.write(uploaded_vid.read())
                st.video(tmp_path)

        with col_r:
            if uploaded_vid and MODEL is not None:
                st.info("Analizando fotogramas del video...")
                cap          = cv2.VideoCapture(tmp_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                sample_every = max(1, total_frames // 8)
                results_vid  = []
                frame_idx    = 0
                progress     = st.progress(0)

                while True:
                    ret, frame = cap.read()
                    if not ret: break
                    if frame_idx % sample_every == 0:
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        pil = Image.fromarray(cv2.resize(rgb, (224, 224)))
                        with _INFER_SEM:
                            r = predict_frame(pil, MODEL)
                        results_vid.append(r)
                        progress.progress(min(100, int(frame_idx / max(total_frames, 1) * 100)))
                    frame_idx += 1
                cap.release()
                progress.empty()

                if results_vid:
                    from collections import Counter
                    counter  = Counter(r['clase'] for r in results_vid)
                    top_clase, top_count = counter.most_common(1)[0]
                    avg_conf = sum(r['confianza'] for r in results_vid if r['clase'] == top_clase) / top_count
                    st.markdown(f"""
<div class="result-box result-top">
    <div class="result-label">Clase dominante</div>
    <h3>{_titulo_clase(top_clase)}</h3>
    <span class="conf-value">{avg_conf:.1f}% confianza · {top_count}/{len(results_vid)} muestras</span>
    <div class="confidence-bar-bg"><div class="confidence-bar" style="width:{avg_conf:.0f}%"></div></div>
</div>""", unsafe_allow_html=True)

                    items = "".join(
                        f"<li><strong>{c.replace('_',' ').title()}</strong> &nbsp;·&nbsp; {n} frames</li>"
                        for c, n in counter.most_common(3)
                    )
                    st.markdown(f"""
<div class="result-box" style="border-color:rgba(255,255,255,0.15);background:rgba(255,255,255,0.04);margin-top:0.6rem">
    <div class="result-label">Resumen de detecciones</div>
    <ul class="alt-list">{items}</ul>
</div>""", unsafe_allow_html=True)
                try:
                    os.remove(tmp_path)
                except Exception: pass

st.markdown('</div>', unsafe_allow_html=True)  # cierre work-panel

# ── PIE DE PÁGINA: GUÍA DE CONTENEDORES ──────────────────────────────────────
st.markdown('<hr class="footer-divider">', unsafe_allow_html=True)
st.markdown('<div class="footer-title">Guía de Contenedores — Normativa AGG 164-2021 · MARN Guatemala</div>', unsafe_allow_html=True)
st.markdown("""
<div class="badges-grid" style="grid-template-columns: repeat(3, 1fr);">
    <span class="sidebar-badge badge-blanco">⬜ Blanco — Reciclables (Plástico, Vidrio, Metal, Papel, Cartón, Tetra Pak)</span>
    <span class="sidebar-badge badge-verde">🟢 Verde — Orgánicos (restos de comida, jardín)</span>
    <span class="sidebar-badge badge-negro">⬛ Negro — No reciclable · Electrónicos (punto limpio)</span>
</div>
""", unsafe_allow_html=True)