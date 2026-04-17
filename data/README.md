# 📦 Dataset — Reciclaje Inteligente

## Fuentes de datos utilizadas

| Nombre | Fuente | URL | Tamaño aprox. | Licencia |
|--------|--------|-----|---------------|----------|
| TrashNet | GitHub — garythung | https://github.com/garythung/trashnet | 2,527 imágenes, 6 clases | MIT |
| Garbage Classification | Kaggle | https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification | ~2,500 imágenes, 6 clases | CC BY 4.0 |
| Imágenes propias del equipo | Recolección manual | — | 500+ imágenes, 10 clases | Propiedad del equipo |

---

## Clases y distribución objetivo

| Clase | Fuente principal | Imágenes públicas | Imágenes propias | Total objetivo |
|-------|-----------------|-------------------|------------------|----------------|
| `lata` | TrashNet | ~400 | 50+ | 450+ |
| `botella_pet` | TrashNet + Kaggle | ~500 | 50+ | 550+ |
| `botella_vidrio` | TrashNet | ~400 | 50+ | 450+ |
| `papel` | TrashNet + Kaggle | ~500 | 50+ | 550+ |
| `carton` | TrashNet | ~400 | 50+ | 450+ |
| `bolsa_plastica` | Kaggle | ~300 | 50+ | 350+ |
| `tetrapak` | Imágenes propias | — | 50+ | 50+ |
| `organico` | Kaggle | ~300 | 50+ | 350+ |
| `electronicos_pequenos` | Imágenes propias | — | 50+ | 50+ |
| `no_reciclable` | Kaggle | ~300 | 50+ | 350+ |

---

## Proceso de limpieza

- Se eliminaron imágenes con resolución menor a 100×100 px.
- Se descartaron imágenes borrosas o con etiqueta incorrecta detectada visualmente.
- No se incluyen imágenes con rostros visibles (cumplimiento ético).
- Las imágenes propias fueron tomadas en distintos fondos, ángulos e iluminación.

---

## Estructura de carpetas

```
data/
├── raw/
│   ├── lata/
│   ├── botella_pet/
│   ├── botella_vidrio/
│   ├── papel/
│   ├── carton/
│   ├── bolsa_plastica/
│   ├── tetrapak/
│   ├── organico/
│   ├── electronicos_pequenos/
│   └── no_reciclable/
└── processed/
    ├── train/
    ├── val/
    └── test/
```

---

## Nota sobre imágenes propias

Las imágenes propias del equipo fueron recolectadas entre el 16 y 25 de abril de 2026. Cada integrante fotografió residuos reales en condiciones variadas (interiores, exteriores, diferentes ángulos) para mejorar la robustez del modelo ante condiciones del mundo real guatemalteco.
