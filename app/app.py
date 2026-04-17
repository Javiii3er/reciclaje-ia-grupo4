"""
app.py — Interfaz web con Streamlit para clasificación de desechos.
Proyecto 01: Reciclaje Inteligente · Grupo #4 · UMG 2026

Uso:
    streamlit run app/app.py
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import streamlit as st
from PIL import Image

from predict import predict_top3

# ── Configuración de la página ────────────────────────────────────────────────
st.set_page_config(
    page_title="Reciclaje Inteligente — Grupo #4",
    page_icon="♻️",
    layout="centered",
)

# ── Encabezado ────────────────────────────────────────────────────────────────
st.title("♻️ Reciclaje Inteligente")
st.markdown(
    "**Clasificación de Desechos con Visión Artificial**  \n"
    "Proyecto 01 · Grupo #4 · Curso 045 — Inteligencia Artificial · UMG 2026"
)
st.divider()

# ── Disclaimer ético ─────────────────────────────────────────────────────────
st.info(
    "⚠️ **Herramienta educativa.** Las recomendaciones son orientativas y pueden "
    "variar según las normativas de tu municipio."
)

# ── Entrada de imagen ─────────────────────────────────────────────────────────
st.subheader("📸 Carga una imagen del residuo")

input_mode = st.radio(
    "Selecciona el modo de entrada:",
    ["📁 Subir imagen", "📷 Usar cámara"],
    horizontal=True,
)

image = None

if input_mode == "📁 Subir imagen":
    uploaded = st.file_uploader(
        "Selecciona una imagen JPG o PNG",
        type=["jpg", "jpeg", "png"],
    )
    if uploaded:
        image = Image.open(uploaded)

else:
    captured = st.camera_input("Toma una foto del residuo")
    if captured:
        image = Image.open(captured)

# ── Predicción ────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'modelo_reciclaje.pth')

if image is not None:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(image, caption="Imagen cargada", use_column_width=True)

    with col2:
        if not os.path.exists(MODEL_PATH):
            st.error(
                "❌ Modelo no encontrado. Asegúrate de que el archivo "
                "`models/modelo_reciclaje.pth` esté disponible."
            )
        else:
            with st.spinner("Analizando imagen..."):
                try:
                    results, disclaimer = predict_top3(image, MODEL_PATH)
                    top = results[0]

                    # Resultado principal
                    st.success(f"**Clase detectada:** `{top['clase']}`")
                    st.metric("Confianza", f"{top['confianza']}%")

                    # Info del recipiente
                    st.markdown(f"### {top['emoji']} Recipiente: **{top['color']}**")
                    st.markdown(f"**Categoría:** {top['categoria']}")
                    st.info(f"💡 **Consejo:** {top['consejo']}")

                    # Barra de confianza
                    st.markdown("**Nivel de confianza:**")
                    st.progress(int(top['confianza']))

                    # Mostrar Top-3 si confianza < 70%
                    if top['confianza'] < 70:
                        st.warning(
                            "La confianza es menor al 70%. "
                            "Aquí están las 3 clases más probables:"
                        )
                        for i, r in enumerate(results, 1):
                            st.markdown(
                                f"**{i}.** `{r['clase']}` — "
                                f"{r['emoji']} {r['color']} — "
                                f"{r['confianza']}%"
                            )

                except Exception as e:
                    st.error(f"Error al procesar la imagen: {e}")

# ── Tabla de referencia ───────────────────────────────────────────────────────
st.divider()
with st.expander("📋 Ver tabla completa de clases y recipientes"):
    st.markdown("""
| Clase | Categoría | Recipiente | Consejo |
|-------|-----------|------------|---------|
| `lata` | Metal | 🟡 Amarillo | Enjuagar y aplastar |
| `botella_pet` | Plástico | 🔵 Azul | Quitar tapón y aplastar |
| `botella_vidrio` | Vidrio | 🟢 Verde | No romper, depositar entero |
| `papel` | Papel | ⚫ Gris | Mantener seco y limpio |
| `carton` | Papel | ⚫ Gris | Doblar y atar si es grande |
| `bolsa_plastica` | Plástico | 🔵 Azul | Vaciar completamente |
| `tetrapak` | Mixto | 🔵 Azul | Enjuagar y aplastar |
| `organico` | Orgánico | 🟤 Café | Para compostaje o biogás |
| `electronicos_pequenos` | Electrónico | 🔴 Rojo | Llevar a punto limpio |
| `no_reciclable` | Otros | ⬛ Negro | Basura general |
""")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Grupo #4 · Proyecto 01 — Reciclaje Inteligente · "
    "Curso 045 Inteligencia Artificial · UMG 2026"
)
