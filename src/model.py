"""
model.py — Definición del modelo MobileNetV2 y mapeo clase → recipiente.
Proyecto 01: Reciclaje Inteligente · Grupo #4 · UMG 2026
"""

import torch
from torchvision import models

# ── Clases del modelo ────────────────────────────────────────────────────────
CLASSES = [
    'bolsa_plastica',
    'botella_pet',
    'botella_vidrio',
    'carton',
    'electronicos_pequenos',
    'lata',
    'no_reciclable',
    'organico',
    'papel',
    'tetrapak',
]

# Colores alineados con AGG 164-2021 y Reforma AGG 184-2023 (MARN Guatemala)
# Última actualización: 11 de mayo de 2026
COLOR_MAP = {
    'lata':                 ('Metal',       'Gris',     '⚫', 'Enjuagar y aplastar antes de depositar.'),
    'botella_pet':          ('Plástico',    'Azul',     '🔵', 'Quitar el tapón y aplastar la botella.'),
    'botella_vidrio':       ('Vidrio',      'Celeste',  '🔷', 'No romper. Depositar entero en el recipiente.'),
    'papel':                ('Papel',       'Amarillo', '🟡', 'Mantener seco y limpio. No mezclar con papel sucio.'),
    'carton':               ('Papel',       'Amarillo', '🟡', 'Doblar y atar si es de gran tamaño.'),
    'bolsa_plastica':       ('Plástico',    'Azul',     '🔵', 'Vaciar completamente antes de depositar.'),
    'tetrapak':             ('Mixto',       'Naranja',  '🟠', 'Enjuagar, aplastar y retirar la pajilla.'),
    'organico':             ('Orgánico',    'Verde',    '🟢', 'Ideal para compostaje casero o biogás.'),
    'electronicos_pequenos':('Electrónico', 'Rojo',     '🔴', 'Llevar a un punto limpio autorizado. No tirar a la basura general.'),
    'no_reciclable':        ('Otros',       'Negro',    '⬛', 'Depositar en la basura general.'),
}

DISCLAIMER = (
    "⚠️ Este sistema es una herramienta de apoyo educativo. "
    "Las recomendaciones son orientativas y pueden variar según el municipio."
)


def load_model(path: str = 'models/modelo_reciclaje.pth', n_classes: int = None) -> torch.nn.Module:
    """
    Carga el modelo MobileNetV2 detectando automáticamente el número de clases
    desde el archivo de pesos para evitar errores de mismatch.
    """
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    
    # Detectar n_classes desde la capa de clasificación (classifier.1.weight)
    if n_classes is None:
        if 'classifier.1.weight' in checkpoint:
            n_classes = checkpoint['classifier.1.weight'].shape[0]
        else:
            n_classes = len(CLASSES)
            
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = torch.nn.Linear(1280, n_classes)
    model.load_state_dict(checkpoint)
    model.eval()
    return model


def build_model(n_classes: int = None) -> torch.nn.Module:
    """
    Construye el modelo MobileNetV2 con pesos preentrenados de ImageNet
    para Transfer Learning (usar durante el entrenamiento).

    Args:
        n_classes: Número de clases de salida. Si es None, usa len(CLASSES).

    Returns:
        Modelo con base congelada y clasificador reemplazado.
    """
    if n_classes is None:
        n_classes = len(CLASSES)
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

    # Congelar todas las capas base (Fase 1)
    for param in model.parameters():
        param.requires_grad = False

    # Reemplazar clasificador final con el número de clases del proyecto
    model.classifier[1] = torch.nn.Linear(1280, n_classes)

    return model


def unfreeze_last_layers(model: torch.nn.Module, n: int = 20) -> torch.nn.Module:
    """
    Descongela las últimas n capas del modelo para fine-tuning (Fase 2).

    Args:
        model: Modelo MobileNetV2 con base congelada.
        n: Número de capas a descongelar desde el final.

    Returns:
        Modelo con últimas n capas descongeladas.
    """
    params = list(model.parameters())
    for param in params[-n:]:
        param.requires_grad = True
    return model


def get_prediction_info(class_name: str) -> dict:
    """
    Retorna la información completa de una clase predicha.

    Args:
        class_name: Nombre de la clase predicha por el modelo.

    Returns:
        Diccionario con categoría, color, emoji y consejo.
    """
    if class_name not in COLOR_MAP:
        return {
            'categoria': 'Desconocido',
            'color': 'Negro',
            'emoji': '⬛',
            'consejo': 'No se pudo determinar. Depositar en basura general.',
        }
    categoria, color, emoji, consejo = COLOR_MAP[class_name]
    return {
        'categoria': categoria,
        'color': color,
        'emoji': emoji,
        'consejo': consejo,
    }
