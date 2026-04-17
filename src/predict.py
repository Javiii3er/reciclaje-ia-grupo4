"""
predict.py — Función de predicción/inferencia para la app Streamlit.
Proyecto 01: Reciclaje Inteligente · Grupo #4 · UMG 2026
"""

import torch
from PIL import Image
from typing import List, Tuple

from model import load_model, CLASSES, get_prediction_info, DISCLAIMER
from preprocess import get_inference_transforms


def predict_top3(
    image: Image.Image,
    model_path: str = 'models/modelo_reciclaje.pth'
) -> Tuple[List[dict], str]:
    """
    Realiza la predicción sobre una imagen PIL y retorna las 3 clases más probables.

    Args:
        image: Imagen PIL cargada (RGB).
        model_path: Ruta al modelo entrenado.

    Returns:
        Tuple con:
        - Lista de dicts con top-3 predicciones (clase, confianza, info del recipiente).
        - Disclaimer ético del sistema.
    """
    model = load_model(model_path)
    transform = get_inference_transforms()

    img_rgb = image.convert('RGB')
    tensor = transform(img_rgb).unsqueeze(0)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]

    top3_probs, top3_idx = probs.topk(3)

    results = []
    for prob, idx in zip(top3_probs.tolist(), top3_idx.tolist()):
        class_name = CLASSES[idx]
        info = get_prediction_info(class_name)
        results.append({
            'clase':      class_name,
            'confianza':  round(prob * 100, 1),
            'categoria':  info['categoria'],
            'color':      info['color'],
            'emoji':      info['emoji'],
            'consejo':    info['consejo'],
        })

    return results, DISCLAIMER
