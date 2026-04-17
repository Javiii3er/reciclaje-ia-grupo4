# ♻️ Reciclaje Inteligente — Clasificación de Desechos con Visión Artificial

> Proyecto 01 — Curso 045: Inteligencia Artificial  
> Universidad Mariano Gálvez de Guatemala · Facultad de Ingeniería  
> Grupo #4 · Ciclo 2026

---

## 📋 Información del Curso

| Campo | Detalle |
|-------|---------|
| **Universidad** | Universidad Mariano Gálvez de Guatemala |
| **Facultad** | Ingeniería, Matemática y Ciencias Físicas |
| **Curso** | 045 — Inteligencia Artificial |
| **Catedrático** | Ing. MA. Carmelo Estuardo Mayén Monterroso |
| **Grupo** | #4 |
| **Ciclo** | 2026 |

---

## 👥 Integrantes del Equipo

| Nombre | Carné | Rol | Clave funcional |
|--------|-------|-----|-----------------|
| Javier José Luis Rivera Pérez | 1790-22-10552 | Líder / Desarrollador Principal | Javier-Líder-ML |
| Marvin Alexander Vásquez López | 1790-22-12802 | Desarrollador de Modelo IA | Marvin-Dev-IA |
| Karla Waleska Rodríguez Arévalo | 1790-22-9763 | Analista de Datos / Soporte EDA | Karla-Analista-EDA |
| Cesar Ulises Gonzáles Cardona | 1790-22-6044 | Desarrollador de Interfaz (App) | Cesar-Dev-App |
| Paolo Alexander Marroquín de la Cruz | 1790-22-8967 | Documentador Técnico / QA | Paolo-Doc-QA |

---

## 📌 Descripción del Proyecto

Sistema de inteligencia artificial capaz de **identificar el tipo de residuo sólido** a partir de una fotografía tomada por cámara o cargada por el usuario. El sistema devuelve:

- ✅ Clase del residuo detectado (ej. `botella_pet`, `lata`, `papel`)
- 🗑️ Color de recipiente donde debe depositarse
- 📊 Nivel de confianza del modelo (%)
- 💡 Consejo práctico de manejo

El modelo está construido con **Transfer Learning sobre MobileNetV2** preentrenado en ImageNet, fine-tuned para 10 clases de residuos comunes en el contexto guatemalteco.

> ⚠️ **Disclaimer ético:** Este sistema es una herramienta de apoyo educativo. Las recomendaciones son orientativas y pueden variar según el municipio. No reemplaza las normativas locales de reciclaje.

---

## 🗂️ Clases del modelo

| Clase | Categoría | Recipiente | Consejo |
|-------|-----------|------------|---------|
| `lata` | Metal | 🟡 Amarillo | Enjuagar y aplastar antes de depositar |
| `botella_pet` | Plástico | 🔵 Azul | Quitar tapón y aplastar |
| `botella_vidrio` | Vidrio | 🟢 Verde | No romper, depositar entero |
| `papel` | Papel | ⚫ Gris | Mantener seco y limpio |
| `carton` | Papel | ⚫ Gris | Doblar y atar si es grande |
| `bolsa_plastica` | Plástico | 🔵 Azul | Vaciar completamente |
| `tetrapak` | Mixto | 🔵 Azul | Enjuagar y aplastar |
| `organico` | Orgánico | 🟤 Café | Para compostaje o biogás |
| `electronicos_pequenos` | Electrónico | 🔴 Rojo | Llevar a punto limpio autorizado |
| `no_reciclable` | Otros | ⬛ Negro | Depositar en basura general |

---

## 🛠️ Tecnologías utilizadas

- **Python** 3.10+
- **PyTorch** + **TorchVision** — modelo MobileNetV2 y entrenamiento
- **Streamlit** — interfaz web interactiva
- **Pillow** — carga y manipulación de imágenes
- **NumPy** — operaciones matriciales
- **scikit-learn** — métricas de evaluación (F1-macro, matriz de confusión)
- **Matplotlib** + **Seaborn** — visualización de métricas
- **OpenCV** — preprocesamiento y captura desde cámara
- **Albumentations** — data augmentation avanzado

---

## ⚙️ Instalación y ejecución

### Requisitos previos
- Python 3.10 o superior
- Git

### Pasos

```bash
# 1. Clonar el repositorio
git clone https://github.com/TU_USUARIO/reciclaje-inteligente.git
cd reciclaje-inteligente

# 2. Crear entorno virtual
python -m venv venv

# 3. Activar entorno virtual
# En Windows:
.\venv\Scripts\activate
# En Linux / Mac:
source venv/bin/activate

# 4. Instalar dependencias
pip install -r requirements.txt

# 5. Ejecutar la aplicación
streamlit run app/app.py
```

---

## 📁 Estructura del repositorio

```
reciclaje-inteligente/
├── README.md                   ← Este archivo
├── requirements.txt            ← Dependencias con versiones exactas
├── .gitignore                  ← Archivos excluidos del repositorio
│
├── data/
│   ├── raw/                    ← Dataset original sin modificar
│   ├── processed/              ← Dataset tras preprocesamiento
│   └── README.md               ← Fuentes, licencias y descripción del dataset
│
├── notebooks/
│   ├── 01_EDA.ipynb            ← Análisis exploratorio de datos
│   ├── 02_train.ipynb          ← Entrenamiento del modelo
│   └── 03_evaluation.ipynb     ← Evaluación, métricas y matriz de confusión
│
├── src/
│   ├── preprocess.py           ← Funciones de preprocesamiento
│   ├── model.py                ← Definición, carga del modelo y COLOR_MAP
│   ├── train.py                ← Script de entrenamiento
│   └── predict.py              ← Función de predicción/inferencia
│
├── app/
│   └── app.py                  ← Interfaz Streamlit
│
├── models/                     ← Modelo entrenado (.pth) — ver nota abajo
│
└── docs/
    ├── minutas/                ← Minutas de reuniones semanales
    ├── manual_tecnico.pdf      ← Manual técnico del sistema
    ├── manual_usuario.pdf      ← Manual de usuario
    ├── DERCAS.pdf              ← Análisis y diseño del sistema
    ├── informe_final.pdf       ← Informe final con métricas
    └── demo_video_link.txt     ← Enlace al video de demostración
```

> 📦 **Nota sobre el modelo:** Si el archivo `.pth` supera 25 MB, se almacena en Google Drive. El enlace de descarga se publicará aquí una vez completado el entrenamiento.

---

## 📊 Resultados del modelo

> *Esta sección se actualizará al completar el entrenamiento (Semana 3 — mayo 2026).*

| Métrica | Valor |
|---------|-------|
| Accuracy global | — |
| F1-macro | — |
| Top-2 Accuracy | — |

---

## 🗓️ Hitos del proyecto

| Hito | Fecha |
|------|-------|
| ⭐ Expoferia — Demo funcional en vivo | 16 de mayo de 2026 |
| 🎓 Entrega final — Repositorio completo tag v1.0 | 30 de mayo de 2026 |

---

## 🎥 Video de demostración

> *Enlace disponible próximamente — semana del 25 al 29 de mayo de 2026.*

---

## 📄 Convención de commits

| Prefijo | Uso |
|---------|-----|
| `feat:` | Nueva funcionalidad |
| `fix:` | Corrección de error |
| `docs:` | Documentación |
| `data:` | Datos y dataset |
| `train:` | Entrenamiento del modelo |
| `app:` | Interfaz Streamlit |
| `eval:` | Evaluación y métricas |
