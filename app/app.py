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
    layout="wide"
)

# ── CSS PREMIUM ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;900&display=swap');

html, body, [class*="st-"] { font-family: 'Inter', sans-serif; }

/* Fondo general */
.stApp {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f2027 100%);
    min-height: 100vh;
}

/* Quitar padding excesivo de Streamlit */
.block-container {
    padding-top: 0.2rem !important;
    padding-bottom: 0 !important;
    max-width: 920px !important;
}
[data-testid="stHeader"] { height: 0 !important; min-height: 0 !important; }
#MainMenu, footer, header { visibility: hidden; }

/* Ocultar scrollbar de la página principal (pero no bloquear scroll de widgets) */
[data-testid="stMain"]::-webkit-scrollbar { display: none; }
[data-testid="stMain"] { scrollbar-width: none; }

/* ── HERO — compacto ──────────────────────────────────────── */
.hero {
    text-align: center;
    padding: 0.15rem 1rem 0.25rem;
}
.hero h1 {
    font-size: 1.9rem;
    font-weight: 900;
    color: #34d399;
    background: linear-gradient(90deg, #34d399, #60a5fa, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    line-height: 1.15;
    display: inline-block;
}
.hero p { color: #94a3b8; font-size: 0.87rem; margin-top: 0.1rem; }

/* ── PANEL DE TRABAJO (visual, sin height fija) ───────────────── */
.work-panel {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 20px;
    padding: 0.9rem 1.2rem;
    backdrop-filter: blur(12px);
    box-shadow: 0 4px 32px rgba(0,0,0,0.3);
    margin: 0.25rem 0 0.5rem;
}

/* ── RESULTADO ───────────────────────────────────────────────── */
.result-box {
    border-radius: 14px;
    padding: 0.7rem 1rem;
    margin-top: 0.5rem;
    border-left: 5px solid;
}
.result-top  { background: rgba(52,211,153,0.12); border-color: #34d399; }
.result-info { background: rgba(96,165,250,0.12); border-color: #60a5fa; }
.result-warn { background: rgba(251,191,36,0.12);  border-color: #fbbf24; }
.result-box h3 { color: #f1f5f9; margin: 0 0 0.2rem; font-size: 0.97rem; }
.result-box p  { color: #cbd5e1; margin: 0; font-size: 0.82rem; }
.confidence-bar-bg {
    background: rgba(255,255,255,0.1);
    border-radius: 8px; height: 7px; margin-top: 0.4rem; overflow: hidden;
}
.confidence-bar {
    height: 100%; border-radius: 8px;
    background: linear-gradient(90deg, #34d399, #60a5fa);
    transition: width 0.5s ease;
}

/* Imagen subida: bordes redondeados y tamaño controlado */
[data-testid="stImage"] img {
    max-height: 240px !important;
    object-fit: contain;
    width: 100% !important;
    border-radius: 12px;
}

/* Cámara: solo radio de borde, sin limitar altura para que funcione */
[data-testid="stCameraInput"] video {
    border-radius: 12px;
    width: 100% !important;
}

/* ── SIDEBAR — FIX 4: scroll interno ────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a, #1e293b);
    border-right: 1px solid rgba(255,255,255,0.08);
}
[data-testid="stSidebar"] > div:first-child {
    height: 100vh;
    overflow-y: auto;
    scrollbar-width: thin;
    scrollbar-color: rgba(100,116,139,0.4) transparent;
    padding-bottom: 1rem;
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }

.sidebar-badge {
    display: block;
    padding: 0.2rem 0.65rem;
    border-radius: 999px;
    font-size: 0.74rem;
    font-weight: 700;
    margin: 0.18rem 0;
}
.badge-azul    { background: #1d4ed8; color: #bfdbfe !important; }
.badge-verde   { background: #065f46; color: #6ee7b7 !important; }
.badge-gris    { background: #374151; color: #d1d5db !important; }
.badge-cafe    { background: #78350f; color: #fcd34d !important; }
.badge-negro   { background: #111827; color: #9ca3af !important; border: 1px solid #374151; }
.badge-rojo    { background: #7f1d1d; color: #fca5a5 !important; }
.badge-amarillo{ background: #78350f; color: #fde68a !important; }

/* ── RADIO BUTTONS ───────────────────────────────────────────── */
div[role="radiogroup"] {
    display: flex !important;
    flex-wrap: nowrap !important;
    gap: 0.5rem !important;
    justify-content: center !important;
}
div[role="radiogroup"] label {
    border-radius: 12px !important;
    padding: 0.4rem 0.7rem !important;
    font-weight: 600 !important;
    font-size: 0.80rem !important;
    transition: background 0.2s !important;
    color: #f1f5f9 !important;
    white-space: nowrap !important;
}
div[role="radiogroup"] label p {
    color: #f1f5f9 !important;
    margin: 0 !important;
}
</style>
""", unsafe_allow_html=True)

# ── MODELO ────────────────────────────────────────────────────────────────────
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
def run_native_stream(model, placeholder_frame, placeholder_result, stop_event):
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 20)

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

        frame_cnt += 1
        if frame_cnt % 20 == 0 and model is not None:
            acquired = _INFER_SEM.acquire(blocking=True, timeout=0.12)
            if acquired:
                try:
                    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    resized = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)
                    with torch.no_grad():
                        last_res = predict_frame(Image.fromarray(resized), model)
                except Exception:
                    pass
                finally:
                    _INFER_SEM.release()

        color_bgr = {
            'Amarillo': (0, 215, 255), 'Azul':  (255, 80,   0),
            'Verde':    (0, 200,  80), 'Gris':  (160, 160, 160),
            'Café':     (42,  85, 165), 'Rojo': (  0,  60, 220),
            'Negro':    (50,  50,  50)
        }.get(last_res.get('color', 'Gris'), (100, 100, 100))

        h, w, _ = frame.shape
        overlay  = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 52), color_bgr, -1)
        cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)
        text = f"{last_res.get('emoji','')} {last_res.get('clase','').upper()} ({last_res.get('confianza',0)}%)"
        cv2.putText(frame, text, (12, 36), cv2.FONT_HERSHEY_DUPLEX, 0.85, (255, 255, 255), 2)

        placeholder_frame['img']   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        placeholder_result['data'] = last_res
        time.sleep(0.04)

    cap.release()

# ── CARGA DEL MODELO ─────────────────────────────────────────────────────────
with st.spinner("⚡ Cargando Motor de IA..."):
    MODEL, DEVICE = get_stable_model()

# ── SIDEBAR: tarjetas de modo + estado + guía ────────────────────────────────
with st.sidebar:
    st.markdown("## 🌿 EcoRecicla IA")
    st.markdown("### Modos de entrada")
    st.markdown("""
<div style="border-radius:12px;padding:0.65rem 1rem;margin-bottom:0.45rem;
            background:linear-gradient(145deg,#064e3b,#065f46);border:1px solid #34d399;">
  <span style="font-size:1.3rem;">📷</span>
  <strong style="color:#f1f5f9;margin-left:0.4rem;">Cámara</strong><br>
  <span style="color:#6ee7b7;font-size:0.73rem;">Captura instantánea con IA</span>
</div>
<div style="border-radius:12px;padding:0.65rem 1rem;margin-bottom:0.45rem;
            background:linear-gradient(145deg,#1e3a5f,#1d4ed8);border:1px solid #60a5fa;">
  <span style="font-size:1.3rem;">🎥</span>
  <strong style="color:#f1f5f9;margin-left:0.4rem;">Video en Vivo</strong><br>
  <span style="color:#93c5fd;font-size:0.73rem;">Stream nativo · IA en tiempo real</span>
</div>
<div style="border-radius:12px;padding:0.65rem 1rem;margin-bottom:0.45rem;
            background:linear-gradient(145deg,#3b1f6b,#6d28d9);border:1px solid #a78bfa;">
  <span style="font-size:1.3rem;">📁</span>
  <strong style="color:#f1f5f9;margin-left:0.4rem;">Subir Archivo</strong><br>
  <span style="color:#c4b5fd;font-size:0.73rem;">Imagen o video · Análisis offline</span>
</div>
""", unsafe_allow_html=True)

    st.markdown("---")

    device_label = "🟢 GPU · CUDA" if (MODEL is not None and str(DEVICE) != 'cpu') else "🟡 CPU"
    model_status = "✅ Activo" if MODEL is not None else "❌ No cargado"
    st.markdown(f"""
<div style="background:rgba(255,255,255,0.05);border:1px solid rgba(255,255,255,0.1);
            border-radius:12px;padding:0.85rem;margin-bottom:0.8rem;">
  <p style="color:#94a3b8;font-size:0.68rem;margin:0 0 0.35rem;text-transform:uppercase;letter-spacing:.07em;">Estado del Sistema</p>
  <p style="color:#f1f5f9;font-weight:700;margin:0;font-size:0.88rem;">🤖 Motor IA · {model_status}</p>
  <p style="color:#60a5fa;margin:0.15rem 0 0;font-size:0.8rem;">⚡ {device_label}</p>
</div>
""", unsafe_allow_html=True)

    st.markdown("**Guía de Contenedores**")
    # FIX 5: Un solo bloque HTML en lugar de 7 llamadas separadas
    st.markdown("""
<div style="display:flex;flex-direction:column;gap:0.22rem;margin-top:0.3rem">
  <span class="sidebar-badge badge-azul">🔵 Azul — Plástico / Latas</span>
  <span class="sidebar-badge badge-verde">🟢 Verde — Vidrio</span>
  <span class="sidebar-badge badge-gris">⚫ Gris — Papel / Cartón</span>
  <span class="sidebar-badge badge-cafe">🟤 Café — Orgánicos</span>
  <span class="sidebar-badge badge-negro">⬛ Negro — No reciclable</span>
  <span class="sidebar-badge badge-rojo">🔴 Rojo — Electrónicos</span>
  <span class="sidebar-badge badge-amarillo">🟡 Amarillo — Metales</span>
</div>
""", unsafe_allow_html=True)
    st.markdown("---")
    st.caption("🚀 Grupo #4 · UMG 2026 · MobileNetV2")

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
            # FIX 3: Resultados en dos columnas para reducir altura
            col_r1, col_r2 = st.columns(2)
            with col_r1:
                st.markdown(f"""
<div class="result-box result-top">
  <h3>✅ {top['emoji']} {top['clase'].replace('_',' ').title()}</h3>
  <p style="color:#34d399;font-weight:700;font-size:1.1rem;margin:0.1rem 0">{top['confianza']}%</p>
  <div class="confidence-bar-bg"><div class="confidence-bar" style="width:{top['confianza']}%"></div></div>
</div>
""", unsafe_allow_html=True)
            with col_r2:
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
                      st.session_state.result_shared, st.session_state.stop_event),
                daemon=True
            ).start()

        if stop_btn and st.session_state.cam_running:
            st.session_state.stop_event.set()
            st.session_state.cam_running = False

        vid_ph = st.empty()
        res_ph = st.empty()

        if st.session_state.cam_running:
            st.info("🟢 Stream activo — IA cada 20 fotogramas. Presiona **DETENER** para parar.")
            deadline = time.time() + 30
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
                    res_ph.markdown(f"""
<div class="result-box result-top" style="margin-top:0.4rem">
  <h3>{rd.get('emoji','')} {rd.get('clase','').replace('_',' ').title()} · {rd.get('confianza',0)}%</h3>
  <div class="confidence-bar-bg"><div class="confidence-bar" style="width:{rd.get('confianza',0)}%"></div></div>
</div>
<div class="result-box result-info" style="margin-top:0.3rem">
  <h3>🗑️ Contenedor: {rd.get('color','')}</h3>
</div>
""", unsafe_allow_html=True)
                time.sleep(0.08)
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
                    st.markdown(f"""
<div class="result-box result-top">
  <h3>✅ {top['emoji']} {top['clase'].replace('_',' ').title()}</h3>
  <p style="color:#34d399;font-weight:700;font-size:1.05rem;margin:0.1rem 0">{top['confianza']}%</p>
  <div class="confidence-bar-bg"><div class="confidence-bar" style="width:{top['confianza']}%"></div></div>
</div>
<div class="result-box result-info" style="margin-top:0.4rem">
  <h3>🗑️ {top['color']}</h3><p>{top['consejo']}</p>
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
