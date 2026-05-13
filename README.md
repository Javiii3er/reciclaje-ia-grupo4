# EcoRecicla IA — Clasificación de Desechos con Visión Artificial

> Proyecto 01 — Curso 045: Inteligencia Artificial
> Universidad Mariano Gálvez de Guatemala · Facultad de Ingeniería
> Grupo #4 · Ciclo 2026

---

## Información del Curso

| Campo | Detalle |
|-------|---------|
| **Universidad** | Universidad Mariano Gálvez de Guatemala |
| **Facultad** | Ingeniería en Sistemas de Información y Ciencias de la Computación |
| **Curso** | 045 — Inteligencia Artificial |
| **Catedrático** | Ing. MA. Carmelo Estuardo Mayén Monterroso |
| **Grupo** | #4 |
| **Ciclo** | 2026 |

---

## Integrantes del Equipo

| Nombre | Carné | Rol | Clave funcional |
|--------|-------|-----|-----------------|
| Javier José Luis Rivera Pérez | 1790-22-10552 | Líder / Desarrollador Principal | Javier-Líder-ML |
| Marvin Alexander Vásquez López | 1790-22-12802 | Desarrollador de Modelo IA | Marvin-Dev-IA |
| Karla Waleska Rodríguez Arévalo | 1790-22-9763 | Analista de Datos / Soporte EDA | Karla-Analista-EDA |
| Cesar Ulises Gonzáles Cardona | 1790-22-6044 | Desarrollador de Interfaz (App) | Cesar-Dev-App |
| Paolo Alexander Marroquín de la Cruz | 1790-22-8967 | Documentador Técnico / QA | Paolo-Doc-QA |

---

## Descripción del Proyecto

**EcoRecicla IA** es una aplicación que usa inteligencia artificial para identificar residuos desde una fotografía y decirte exactamente en qué recipiente depositarlos. Desarrollada con tecnología **MobileNetV2** y entrenada con más de 1,900 imágenes, clasifica 10 tipos de residuos en tiempo real y busca facilitar el reciclaje responsable en Guatemala, alineada con el **Acuerdo Gubernativo 164-2021**.

El sistema devuelve:

- Clase del residuo detectado (ej. `botella_pet`, `lata`, `papel`)
- Color de recipiente donde debe depositarse según AGG 164-2021
- Nivel de confianza del modelo (%)
- Consejo práctico de manejo
- Alerta de duda cuando la confianza es menor al 70%

> **Disclaimer ético:** Este sistema es una herramienta de apoyo educativo. Las recomendaciones son orientativas y pueden variar según las normativas de reciclaje de cada municipio. No reemplaza la consulta de normativas locales.

---

## Resultados del Modelo

| Métrica | Valor | Descripción |
|---------|-------|-------------|
| **Accuracy global** | **99.44%** | Imágenes correctamente clasificadas en el conjunto de entrenamiento |
| **Top-2 Accuracy** | **99.86%** | La clase correcta está entre las 2 más probables |
| **Confianza media** | **99.68%** | El modelo es muy seguro en sus predicciones |
| **Baseline (referencia)** | 27.66% | Accuracy de un modelo que adivina al azar |
| **Mejora sobre baseline** | +71.78 pp | El modelo es 71.78 puntos porcentuales mejor que adivinar |

> **Nota:** Las métricas corresponden al dataset de entrenamiento. Los resultados en producción con imágenes del mundo real pueden variar.

### Análisis de errores de confusión

El modelo es altamente preciso, pero presenta 3 errores recurrentes en el conjunto de prueba:

| # | Clase real | Confundida con | Veces | Razón |
|---|-----------|----------------|-------|-------|
| 1 | `papel` | `carton` | 8 | Similitud de textura en imágenes planas |
| 2 | `no_reciclable` | `papel` | 7 | Materiales de apariencia similar |
| 3 | `botella_pet` | `botella_vidrio` | 6 | Transparencia similar en ciertas condiciones de luz |

> Estos casos están documentados en el Informe PDF final y son la base del análisis de errores del proyecto.

---

## Clases del modelo

Colores alineados con el **Acuerdo Gubernativo 164-2021** (MARN Guatemala) y su Reforma AGG 184-2023.

| Clase | Categoría | Recipiente | Consejo |
|-------|-----------|------------|---------|
| `lata` | Metal | ⚫ Gris | Enjuagar y aplastar antes de depositar |
| `botella_pet` | Plástico | 🔵 Azul | Quitar tapón y aplastar |
| `botella_vidrio` | Vidrio | 🔷 Celeste | No romper, depositar entero |
| `papel` | Papel | 🟡 Amarillo | Mantener seco y limpio |
| `carton` | Papel | 🟡 Amarillo | Doblar y atar si es grande |
| `bolsa_plastica` | Plástico | 🔵 Azul | Vaciar completamente |
| `tetrapak` | Mixto | 🟠 Naranja | Enjuagar y aplastar |
| `organico` | Orgánico | 🟢 Verde | Para compostaje o biogás |
| `electronicos_pequenos` | Electrónico | 🔴 Rojo | Llevar a punto limpio autorizado |
| `no_reciclable` | Otros | ⬛ Negro | Depositar en basura general |

---

## Modos de la aplicación

La app **EcoRecicla IA** cuenta con tres modos de entrada:

| Modo | Descripción |
|------|-------------|
| **Foto instantánea** | Captura desde la cámara web del dispositivo |
| **Video en tiempo real** | Stream continuo con OpenCV — predicción cada 20 fotogramas |
| **Archivo** | Carga de imagen (JPG/PNG) o video (MP4/AVI/MOV) |

---

## Tecnologías utilizadas

| Librería | Versión | Uso |
|----------|---------|-----|
| `torch` | 2.2.2 | Entrenamiento y carga del modelo MobileNetV2 |
| `torchvision` | 0.17.2 | Transformaciones de imagen y modelos preentrenados |
| `streamlit` | 1.33.0 | Interfaz web interactiva |
| `Pillow` | 10.3.0 | Carga y manipulación de imágenes |
| `numpy` | 1.26.4 | Operaciones matriciales |
| `scikit-learn` | 1.4.2 | Métricas de evaluación (F1-macro, matriz de confusión) |
| `matplotlib` | 3.8.4 | Gráficas de entrenamiento |
| `seaborn` | 0.13.2 | Heatmap de matriz de confusión |
| `opencv-python` | 4.9.0.80 | Preprocesamiento y video en tiempo real |
| `albumentations` | 1.4.3 | Data augmentation avanzado |

---

## Instalación y ejecución

### Requisitos previos
- Python 3.10 o superior
- Git
- Cámara web (para modos foto y video)

### Pasos

```bash
# 1. Clonar el repositorio
git clone https://github.com/Javiii3er/reciclaje-ia-grupo4.git
cd reciclaje-ia-grupo4

# 2. Crear entorno virtual
python -m venv venv

# 3. Activar entorno virtual
# Windows:
.\venv\Scripts\activate
# Linux / Mac:
source venv/bin/activate

# 4. Instalar dependencias
pip install -r requirements.txt

# 5. Ejecutar la aplicación
streamlit run app/app.py
```

> **Nota sobre el modelo:** El archivo `models/modelo_reciclaje.pth` debe estar en la carpeta `models/`. Si supera 25 MB se almacena en Google Drive — ver enlace abajo.

---

## Estructura del repositorio

```
reciclaje-ia-grupo4/
├── README.md                       ← Este archivo
├── requirements.txt                ← Dependencias con versiones exactas
├── .gitignore
├── download_datasets.py            ← Descarga automática de datasets públicos
│
├── data/
│   ├── raw/                        ← Dataset organizado por clase
│   ├── processed/                  ← Splits train/val/test (70/15/15)
│   └── README.md                   ← Fuentes, licencias y distribución
│
├── notebooks/
│   ├── 01_EDA.ipynb                ← Análisis exploratorio con resultados
│   ├── 02_train.ipynb              ← Entrenamiento con curva de aprendizaje
│   └── 03_evaluation.ipynb         ← Métricas finales y matriz de confusión
│
├── src/
│   ├── model.py                    ← MobileNetV2 y COLOR_MAP (10 clases)
│   ├── preprocess.py               ← Transformaciones e augmentation
│   ├── train.py                    ← Entrenamiento con Early Stopping
│   ├── evaluate.py                 ← Evaluación y generación de métricas
│   ├── predict.py                  ← Inferencia top-3 y tiempo real
│   └── split_dataset.py            ← Split estratificado 70/15/15
│
├── app/
│   └── app.py                      ← EcoRecicla IA — Interfaz Streamlit
│
├── models/
│   └── modelo_reciclaje.pth        ← Modelo entrenado (ver nota arriba)
│
└── docs/
    ├── curva_aprendizaje.png        ← Curva loss/accuracy por época
    ├── matriz_confusion.png         ← Heatmap de matriz de confusión
    ├── minutas/                     ← Minutas de reuniones semanales
    ├── manual_tecnico.pdf           ← Manual técnico del sistema
    ├── manual_usuario.pdf           ← Manual de usuario
    ├── DERCAS.pdf                   ← Análisis y diseño del sistema
    ├── informe_final.pdf            ← Informe final con métricas
    └── demo_video_link.txt          ← Enlace al video de demostración
```

---

## Hitos del proyecto

| Hito | Fecha | Estado |
|------|-------|--------|
| Expoferia — Demo funcional en vivo | 16 de mayo de 2026 | 🟡 Próximo |
| Entrega final — Repositorio con tag v1.0 | 30 de mayo de 2026 | ⏳ Pendiente |

---

## Convención de commits

| Prefijo | Uso |
|---------|-----|
| `feat:` | Nueva funcionalidad |
| `fix:` | Corrección de error |
| `docs:` | Documentación |
| `data:` | Datos y dataset |
| `train:` | Entrenamiento del modelo |
| `app:` | Interfaz Streamlit |
| `eval:` | Evaluación y métricas |

---

## Video de demostración

> Enlace disponible la semana del 25 al 29 de mayo de 2026.

---

*Grupo #4 · Proyecto 01 — Reciclaje Inteligente · Curso 045 Inteligencia Artificial · UMG 2026*
