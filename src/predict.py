"""
predict.py — Inferencia ultra-estable optimizada para GPU.
Proyecto 01: Reciclaje Inteligente · Grupo #4 · UMG 2026
"""

import torch
from PIL import Image
from typing import List, Tuple

from model import load_model, CLASSES, get_prediction_info, DISCLAIMER
from preprocess import get_inference_transforms


def predict_top3(
    image: Image.Image,
    model_path: str = 'models/modelo_reciclaje.pth',
    model: torch.nn.Module = None
) -> Tuple[List[dict], str]:
    """
    Predicción optimizada para imágenes estáticas.
    """
    if model is None:
        model = load_model(model_path)
    
    # Detectar el dispositivo en el que ya está el modelo
    device = next(model.parameters()).device
    
    transform = get_inference_transforms()
    img_rgb = image.convert('RGB')
    tensor = transform(img_rgb).unsqueeze(0).to(device)

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


def predict_frame(
    image: Image.Image,
    model: torch.nn.Module
) -> dict:
    """
    Predicción de alta velocidad para cuadros de video.
    Asume que el modelo ya está en el dispositivo correcto (GPU/CPU).
    """
    # Detectar dispositivo automáticamente
    device = next(model.parameters()).device
    
    transform = get_inference_transforms()
    img_rgb = image.convert('RGB')
    tensor = transform(img_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        prob, idx = torch.max(probs, dim=0)

    class_name = CLASSES[idx.item()]
    info = get_prediction_info(class_name)
    
    return {
        'clase':      class_name,
        'confianza':  round(prob.item() * 100, 1),
        'emoji':      info['emoji'],
        'color':      info['color'],
        'consejo':    info['consejo'],
    }