"""
app.py — UI/UX Premium · Tarjetas en sidebar · Captura centrada
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
import torch

torch.set_num_threads(1)

from predict import predict_top3, predict_frame, load_model

# ── CONFIGURACIÓN DE PÁGINA ──────────────────────────────────────────────────
st.set_page_config(
    page_title="EcoRecicla IA | Grupo #4 UMG",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── CSS PREMIUM ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;900&display=swap');

html, body, [class*="st-"] {
    font-family: 'Inter', sans-serif;
    font-size: 1.05rem;
}

.stApp {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f2027 100%);
    min-height: 100vh;
}

.block-container {
    padding-top: 0.1rem !important;
    padding-bottom: 0 !important;
    padding-left: 1rem !important;
    padding-right: 1rem !important;
    max-width: 100% !important;
}
[data-testid="stHeader"] { height: 0 !important; min-height: 0 !important; }
#MainMenu, footer, header { visibility: hidden; }

[data-testid="stMain"]::-webkit-scrollbar { display: none; }
[data-testid="stMain"] { scrollbar-width: none; }

/* ── NAVBAR ────────────────────────────────────────────────────── */
.navbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: linear-gradient(90deg, #0f172a, #1e293b);
    padding: 0.6rem 1rem;
    border-bottom: 1px solid rgba(255,255,255,0.1);
    margin-bottom: 0.4rem;
    border-radius: 8px;
}
.navbar-left { font-size: 1.6rem; font-weight: 700; color: #34d399; }
.navbar-center { display: flex; gap: 0.6rem; }
.navbar-center span {
    font-size: 0.95rem;
    padding: 0.3rem 0.5rem;
    background: rgba(255,255,255,0.1);
    border-radius: 6px;
    color: #cbd5e1;
}

/* ── HERO ──────────────────────────────────────────────────────── */
.hero { text-align: center; padding: 0.3rem 1rem 0.4rem; }
.hero h1 {
    font-size: 2.8rem;
    font-weight: 900;
    background: linear-gradient(90deg, #34d399, #60a5fa, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    line-height: 1.15;
    display: inline-block;
}
.hero p { color: #94a3b8; font-size: 1.3rem; margin-top: 0.2rem; }

/* ── PANEL DE TRABAJO ──────────────────────────────────────────── */
.work-panel {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 20px;
    padding: 0.8rem 1.2rem;
    backdrop-filter: blur(12px);
    box-shadow: 0 4px 32px rgba(0,0,0,0.3);
    margin: 0.3rem 0 0.4rem;
}

/* ── RESULTADO ─────────────────────────────────────────────────── */
.result-box {
    border-radius: 14px;
    padding: 0.9rem 1.2rem;
    margin-top: 0.6rem;
    border-left: 5px solid;
}
.result-top  { background: rgba(52,211,153,0.12); border-color: #34d399; }
.result-info { background: rgba(96,165,250,0.12); border-color: #60a5fa; }
.result-warn { background: rgba(251,191,36,0.12);  border-color: #fbbf24; }
.result-box h3 { color: #f1f5f9; margin: 0 0 0.25rem; font-size: 1.4rem; }
.result-box p  { color: #cbd5e1; margin: 0; font-size: 1rem; }
.confidence-bar-bg {
    background: rgba(255,255,255,0.1);
    border-radius: 8px; height: 14px; margin-top: 0.4rem; overflow: hidden;
}
.confidence-bar {
    height: 100%; border-radius: 8px;
    background: linear-gradient(90deg, #34d399, #60a5fa);
    transition: width 0.5s ease;
}

/* ── IMÁGENES / VIDEO ──────────────────────────────────────────── */
[data-testid="stImage"] img {
    max-height: 450px !important;
    object-fit: contain;
    width: 100% !important;
    border-radius: 12px;
}
[data-testid="stCameraInput"] video {
    border-radius: 12px;
    width: 100% !important;
    max-height: 450px !important;
}

/* ── BADGES — colores según AGG 164-2021 ───────────────────────── */
.sidebar-badge {
    display: block;
    padding: 0.35rem 0.8rem;
    border-radius: 999px;
    font-size: 0.95rem;
    font-weight: 700;
    margin: 0.2rem 0;
}
.badge-amarillo { background: #854d0e; color: #fde68a !important; }
.badge-azul     { background: #1d4ed8; color: #bfdbfe !important; }
.badge-celeste  { background: #0e4f6b; color: #a5f3fc !important; }
.badge-gris     { background: #374151; color: #d1d5db !important; }
.badge-naranja  { background: #7c2d12; color: #fed7aa !important; }
.badge-verde    { background: #065f46; color: #6ee7b7 !important; }
.badge-negro    { background: #111827; color: #9ca3af !important; border: 1px solid #374151; }
.badge-rojo     { background: #7f1d1d; color: #fca5a5 !important; }

/* ── BOTONES ────────────────────────────────────────────────────── */
button { min-height: 50px !important; font-size: 1rem !important; padding: 0.8rem 1.2rem !important; }

/* ── RADIO BUTTONS ─────────────────────────────────────────────── */
div[role="radiogroup"] {
    display: flex !important; flex-wrap: nowrap !important;
    gap: 0.6rem !important; justify-content: center !important;
}
div[role="radiogroup"] label {
    border-radius: 12px !important; padding: 0.6rem 1rem !important;
    font-weight: 600 !important; font-size: 1.05rem !important;
    transition: background 0.2s !important; color: #f1f5f9 !important;
    white-space: nowrap !important;
}
div[role="radiogroup"] label p { color: #f1f5f9 !important; margin: 0 !important; }

/* ── SLIDERS ────────────────────────────────────────────────────── */
[data-testid="stSlider"] { padding: 1rem 0; }
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

# ── STREAM NATIVO (cv2 + st.empty) ───────────────────────────────────────────
def run_native_stream(model, placeholder_frame, placeholder_result, stop_event, fps_value=20):
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    # Aumentar resolución a 720p para vista más grande
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, fps_value)

    if not cap.isOpened():
        placeholder_result['error'] = 'No se pudo acceder a la cámara. Verifica permisos.'
        stop_event.set()
        return

    last_res  = {'emoji': '🔍', 'clase': 'Analizando...', 'confianza': 0, 'color': 'Gris'}
    frame_cnt = 0

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        h, w, _ = frame.shape
        # ── RETÍCULA DE DETECCIÓN (CENTRAL) ──
        # Definimos un cuadro de 400x400 en el centro del stream
        rect_size = 400
        x1 = (w - rect_size) // 2
        y1 = (h - rect_size) // 2
        x2 = x1 + rect_size
        y2 = y1 + rect_size

        frame_cnt += 1
        # Inferencia cada 15 fotogramas (~1.5 veces por segundo)
        if frame_cnt % 15 == 0 and model is not None:
            acquired = _INFER_SEM.acquire(blocking=True, timeout=0.10)
            if acquired:
                try:
                    # DETECTOR DE OBJETO: Solo enviamos a la IA lo que está dentro del cuadro
                    roi = frame[y1:y2, x1:x2]
                    rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                    resized = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)
                    with torch.no_grad():
                        last_res = predict_frame(Image.fromarray(resized), model)
                except Exception:
                    pass
                finally:
                    _INFER_SEM.release()

        # Colores BGR alineados con AGG 164-2021
        color_bgr = {
            'Amarillo': (0,  215, 255),
            'Azul':     (255,  80,   0),
            'Celeste':  (210, 180,   0),
            'Gris':     (160, 160, 160),
            'Naranja':  (0,  140, 255),
            'Verde':    (0,  200,  80),
            'Rojo':     (0,   60, 220),
            'Negro':    (50,  50,  50),
        }.get(last_res.get('color', 'Gris'), (100, 100, 100))

        # ── OVERLAY VISUAL ──
        overlay = frame.copy()
        
        # 1. Dibujar el cuadro del "Detector de Objetos"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
        cv2.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), color_bgr, 1) # Borde exterior de color

        # 2. Barra de información superior
        cv2.rectangle(overlay, (0, 0), (w, 60), color_bgr, -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        
        text = f"{last_res.get('emoji','')} {last_res.get('clase','').upper()} ({last_res.get('confianza',0)}%)"
        cv2.putText(frame, text, (20, 42), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2)

        placeholder_frame['img']   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        placeholder_result['data'] = last_res
        time.sleep(0.03)

    cap.release()

# ── CARGA DEL MODELO ─────────────────────────────────────────────────────────
with st.spinner("⚡ Cargando Motor de IA..."):
    MODEL, DEVICE = get_stable_model()

# ── TOP NAVBAR COMPACTA ──────────────────────────────────────────────────
device_label = "🟢 GPU · CUDA" if (MODEL is not None and str(DEVICE) != 'cpu') else "🟡 CPU"
model_status = "✅ Activo" if MODEL is not None else "❌ No cargado"

st.markdown(f"""
<div class="navbar">
    <div class="navbar-left">🌿 EcoRecicla IA</div>
    <div class="navbar-center">
        <span>🤖 {model_status}</span>
        <span>{device_label}</span>
        <span>📊 99.44% Accuracy</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Guía de contenedores — colores correctos según AGG 164-2021
st.markdown("**🗑️ Contenedores según AGG 164-2021**")
col_c1, col_c2, col_c3 = st.columns(3)
with col_c1:
    st.markdown("""
    <span class="sidebar-badge badge-amarillo" style="display:block;margin-bottom:0.2rem">🟡 Amarillo — Papel / Cartón</span>
    <span class="sidebar-badge badge-azul"     style="display:block;margin-bottom:0.2rem">🔵 Azul — Plástico</span>
    <span class="sidebar-badge badge-celeste"  style="display:block">🔷 Celeste — Vidrio</span>
    """, unsafe_allow_html=True)
with col_c2:
    st.markdown("""
    <span class="sidebar-badge badge-gris"    style="display:block;margin-bottom:0.2rem">⚫ Gris — Metal</span>
    <span class="sidebar-badge badge-naranja" style="display:block;margin-bottom:0.2rem">🟠 Naranja — Multicapa (Tetra Pak)</span>
    <span class="sidebar-badge badge-verde"   style="display:block">🟢 Verde — Orgánicos</span>
    """, unsafe_allow_html=True)
with col_c3:
    st.markdown("""
    <span class="sidebar-badge badge-negro" style="display:block;margin-bottom:0.2rem">⬛ Negro — No reciclable</span>
    <span class="sidebar-badge badge-rojo"  style="display:block">🔴 Rojo — Electrónicos</span>
    """, unsafe_allow_html=True)

st.divider()

# ── HERO ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🌿 EcoRecicla IA</h1>
  <p>Clasificación inteligente de residuos · MobileNetV2 · GPU Acelerado</p>
</div>
""", unsafe_allow_html=True)

# ── SELECTOR DE MODO (pantalla completa centrada) ─────────────────────────────
mode = st.radio(
    "",
    options=["📷  CÁMARA — Tomar Foto", "🎥  VIDEO — Tiempo Real", "📁  ARCHIVO — Subir Imagen/Video"],
    horizontal=True,
    label_visibility="collapsed"
)

# AUTO-STOP VIDEO: Si cambiamos de modo, liberamos la cámara inmediatamente para evitar conflictos
if "VIDEO" not in mode and st.session_state.get("cam_running"):
    st.session_state.stop_event.set()
    st.session_state.cam_running = False
    st.rerun() 

st.markdown('<div class="work-panel">', unsafe_allow_html=True)

# ── MODO CÁMARA ───────────────────────────────────────────────────────────────
if "CÁMARA" in mode:
    st.markdown("### 📷 Cámara · Captura Instantánea")
    st.caption("Apunta al objeto y toma la foto. La IA analizará el residuo al instante.")
    camera_photo = st.camera_input("", label_visibility="collapsed")
    if camera_photo:
        img = Image.open(camera_photo)
        if MODEL is not None:
            with _INFER_SEM:
                res, disclaimer = predict_top3(img, model=MODEL)
            top = res[0]
            # --- LÓGICA DE UMBRAL DE DUDA (FASE C) ---
            es_dudoso = top['confianza'] < DOUBT_THRESHOLD
            titulo_res = f"✅ {top['emoji']} {top['clase'].replace('_',' ').title()}"
            sub_res    = f"{top['confianza']}%"
            msg_duda   = ""
            
            if es_dudoso and len(res) > 1:
                alt = res[1]
                titulo_res = f"🤔 ¿{top['clase'].replace('_',' ').title()}?"
                sub_res    = "Incertidumbre detectada"
                msg_duda   = (f"Parece **{top['clase'].replace('_',' ')}**, pero también podría ser "
                              f"**{alt['clase'].replace('_',' ')}**. ¿Podrías acercar más la cámara?")

            # FIX 3: Resultados en dos columnas para reducir altura
            col_r1, col_r2 = st.columns(2)
            with col_r1:
                color_box = "result-top" if not es_dudoso else "result-warn"
                st.markdown(f"""
<div class="result-box {color_box}">
  <h3>{titulo_res}</h3>
  <p style="color:#f1f5f9;font-weight:700;font-size:1.1rem;margin:0.1rem 0">{sub_res}</p>
  <div class="confidence-bar-bg"><div class="confidence-bar" style="width:{top['confianza']}%"></div></div>
</div>
""", unsafe_allow_html=True)
            with col_r2:
                if es_dudoso:
                    st.markdown(f"""
<div class="result-box result-info">
  <h3>💡 Recomendación</h3>
  <p>{msg_duda}</p>
</div>
""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""
<div class="result-box result-info">
  <h3>🗑️ {top['color']}</h3>
  <p>{top['consejo']}</p>
</div>
""", unsafe_allow_html=True)
            if len(res) > 1:
                st.markdown("**🔬 Otras posibilidades:**")
                cols_alt = st.columns(len(res) - 1)
                for i, r in enumerate(res[1:]):
                    cols_alt[i].markdown(f"{r['emoji']} **{r['clase'].replace('_',' ').title()}** — {r['confianza']}%")
            st.markdown(f'<div class="result-box result-warn"><p>{disclaimer}</p></div>', unsafe_allow_html=True)
        else:
            st.error("⚠️ Modelo no disponible.")

# ── MODO VIDEO ────────────────────────────────────────────────────────────────
elif "VIDEO" in mode:
    st.markdown("### 🎥 Video · Detección en Tiempo Real")
    st.caption("Cámara nativa OpenCV — sin WebRTC. Presiona **INICIAR** para arrancar el stream con IA.")

    st.markdown("#### ⚙️ Configuración")
    col_fps, col_info = st.columns([2, 1])
    with col_fps:
        fps_value = st.slider(
            "📊 Fotogramas por segundo (FPS)",
            min_value=10,
            max_value=60,
            value=20,
            step=5,
            help="Mayor FPS = más precisión pero más CPU. Menor FPS = menos CPU."
        )
    with col_info:
        st.info(f"🎯 FPS: **{fps_value}**")
    st.divider()

    if MODEL is None:
        st.error("⚠️ Modelo no disponible.")
    else:
        col_btn1, col_btn2, _ = st.columns([1, 1, 3])
        start_btn = col_btn1.button("▶️  INICIAR", key="start_cam", use_container_width=True)
        stop_btn  = col_btn2.button("⏹️  DETENER", key="stop_cam",  use_container_width=True)

        if "cam_running" not in st.session_state:
            st.session_state.cam_running   = False
            st.session_state.stop_event    = threading.Event()
            st.session_state.frame_shared  = {}
            st.session_state.result_shared = {}

        if start_btn and not st.session_state.cam_running:
            st.session_state.stop_event    = threading.Event()
            st.session_state.frame_shared  = {}
            st.session_state.result_shared = {}
            st.session_state.cam_running   = True
            threading.Thread(
                target=run_native_stream,
                args=(MODEL, st.session_state.frame_shared,
                      st.session_state.result_shared, st.session_state.stop_event, fps_value),
                daemon=True
            ).start()

        if stop_btn and st.session_state.cam_running:
            st.session_state.stop_event.set()
            st.session_state.cam_running = False

        vid_ph = st.empty()
        res_ph = st.empty()

        if st.session_state.cam_running:
            st.info("🟢 Stream activo. Presiona **DETENER** para parar.")
            # Aumentamos el tiempo de vida del stream para la Expoferia (4 horas)
            deadline = time.time() + 14400 
            while st.session_state.cam_running and time.time() < deadline:
                fd = st.session_state.frame_shared.get("img")
                rd = st.session_state.result_shared.get("data")
                em = st.session_state.result_shared.get("error")
                if em:
                    vid_ph.error(f"⚠️ {em}")
                    st.session_state.cam_running = False
                    break
                if fd is not None:
                    vid_ph.image(fd, channels="RGB", use_column_width=True, caption="🎥 Stream en Vivo")
                if rd:
                    # --- LÓGICA DE UMBRAL DE DUDA (FASE C) ---
                    conf = rd.get('confianza', 0)
                    es_dudoso = conf < DOUBT_THRESHOLD
                    titulo_res = f"{rd.get('emoji','')} {rd.get('clase','').replace('_',' ').title()} · {conf}%"
                    color_box = "result-top" if not es_dudoso else "result-warn"
                    
                    if es_dudoso:
                        titulo_res = f"🤔 ¿{rd.get('clase','').replace('_',' ').title()}? · Incertidumbre"

                    res_ph.markdown(f"""
<div class="result-box {color_box}" style="margin-top:0.4rem">
  <h3>{titulo_res}</h3>
  <div class="confidence-bar-bg"><div class="confidence-bar" style="width:{conf}%"></div></div>
</div>
<div class="result-box result-info" style="margin-top:0.3rem">
  <h3>{"💡 Sugerencia" if es_dudoso else "🗑️ Contenedor: " + rd.get('color','')}</h3>
  <p>{"La IA tiene dudas. Intenta centrar mejor el objeto." if es_dudoso else ""}</p>
</div>
""", unsafe_allow_html=True)
                # Reducimos el sleep para mayor fluidez visual (~33 FPS max)
                time.sleep(0.03)
            if st.session_state.cam_running:
                st.warning("🔄 Presiona **INICIAR** de nuevo para continuar.")
                st.session_state.stop_event.set()
                st.session_state.cam_running = False

# ── MODO ARCHIVO ──────────────────────────────────────────────────────────────
else:
    st.markdown("### 📁 Archivo · Análisis Offline")
    st.caption("Sube una imagen (JPG, PNG, WEBP) o un video (MP4, AVI, MOV).")
    tab_img, tab_vid = st.tabs(["🖼️  Imagen", "🎬  Video"])

    with tab_img:
        uploaded_img = st.file_uploader(
            "Arrastra o selecciona una imagen",
            type=["jpg", "jpeg", "png", "webp"], key="img_upload"
        )
        if uploaded_img:
            img = Image.open(uploaded_img)
            # FIX 3: imagen + resultados en columnas lado a lado
            col_img, col_res = st.columns([1, 1])
            with col_img:
                st.image(img, use_column_width=True, caption="Imagen cargada")
            with col_res:
                if MODEL is not None:
                    with _INFER_SEM:
                        res, disclaimer = predict_top3(img, model=MODEL)
                    top = res[0]
                    # --- LÓGICA DE UMBRAL DE DUDA (FASE C) ---
                    es_dudoso = top['confianza'] < DOUBT_THRESHOLD
                    titulo_res = f"✅ {top['emoji']} {top['clase'].replace('_',' ').title()}"
                    sub_res    = f"{top['confianza']}%"
                    msg_duda   = ""
                    
                    if es_dudoso and len(res) > 1:
                        alt = res[1]
                        titulo_res = f"🤔 ¿{top['clase'].replace('_',' ').title()}?"
                        sub_res    = "Incertidumbre detectada"
                        msg_duda   = (f"Parece **{top['clase'].replace('_',' ')}**, pero también podría ser "
                                      f"**{alt['clase'].replace('_',' ')}**. ¿Podrías acercar más la cámara?")

                    color_box = "result-top" if not es_dudoso else "result-warn"
                    st.markdown(f"""
<div class="result-box {color_box}">
  <h3>{titulo_res}</h3>
  <p style="color:#f1f5f9;font-weight:700;font-size:1.05rem;margin:0.1rem 0">{sub_res}</p>
  <div class="confidence-bar-bg"><div class="confidence-bar" style="width:{top['confianza']}%"></div></div>
</div>
<div class="result-box result-info" style="margin-top:0.4rem">
  <h3>{"💡 Recomendación" if es_dudoso else "🗑️ " + top['color']}</h3>
  <p>{msg_duda if es_dudoso else top['consejo']}</p>
</div>
""", unsafe_allow_html=True)
                    st.markdown("**🔬 Top 3:**")
                    for r in res:
                        st.markdown(f"- {r['emoji']} **{r['clase'].replace('_',' ').title()}** — {r['confianza']}%")
                    st.markdown(f'<div class="result-box result-warn"><p>{disclaimer}</p></div>', unsafe_allow_html=True)
                else:
                    st.error("⚠️ Modelo no disponible.")

    with tab_vid:
        uploaded_vid = st.file_uploader(
            "Arrastra o selecciona un video",
            type=["mp4", "avi", "mov", "mkv"], key="vid_upload"
        )
        if uploaded_vid:
            tmp_path = os.path.join(os.path.dirname(__file__), "_tmp_video.mp4")
            with open(tmp_path, "wb") as f:
                f.write(uploaded_vid.read())
            st.video(tmp_path)
            st.info("🔬 Analizando fotogramas del video...")
            cap          = cv2.VideoCapture(tmp_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            sample_every = max(1, total_frames // 8)
            results_vid  = []
            frame_idx    = 0
            progress     = st.progress(0, text="Procesando...")
            if MODEL is not None:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if frame_idx % sample_every == 0:
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        pil = Image.fromarray(cv2.resize(rgb, (224, 224)))
                        with _INFER_SEM:
                            r = predict_frame(pil, MODEL)
                        results_vid.append(r)
                        done = min(100, int(frame_idx / max(total_frames, 1) * 100))
                        progress.progress(done, text=f"Fotograma {frame_idx}/{total_frames}")
                    frame_idx += 1
                cap.release()
                progress.empty()
                if results_vid:
                    from collections import Counter
                    counter  = Counter(r['clase'] for r in results_vid)
                    top_clase, top_count = counter.most_common(1)[0]
                    top_r    = next(r for r in results_vid if r['clase'] == top_clase)
                    avg_conf = sum(r['confianza'] for r in results_vid if r['clase'] == top_clase) / top_count
                    st.markdown(f"""
<div class="result-box result-top">
  <h3>✅ {top_r['emoji']} {top_clase.replace('_',' ').title()} — {avg_conf:.1f}%</h3>
  <p>{top_count}/{len(results_vid)} muestras detectadas</p>
  <div class="confidence-bar-bg"><div class="confidence-bar" style="width:{avg_conf:.0f}%"></div></div>
</div>
<div class="result-box result-info" style="margin-top:0.5rem">
  <h3>🗑️ Contenedor: {top_r['color']}</h3>
</div>
""", unsafe_allow_html=True)
                    st.markdown("**📊 Todas las detecciones:**")
                    for clase, cnt in counter.most_common():
                        emoji = next((r['emoji'] for r in results_vid if r['clase'] == clase), '❓')
                        st.markdown(f"- {emoji} **{clase.replace('_',' ').title()}** · {cnt} fotogramas")
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
            else:
                cap.release()
                st.error("⚠️ Modelo no disponible.")
