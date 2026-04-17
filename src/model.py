"""
model.py — Definición del modelo MobileNetV2 y mapeo clase → recipiente.
Proyecto 01: Reciclaje Inteligente · Grupo #4 · UMG 2026
"""

import torch
from torchvision import models

# ── Clases del modelo ────────────────────────────────────────────────────────
CLASSES = [
    'lata',
    'botella_pet',
    'botella_vidrio',
    'papel',
    'carton',
    'bolsa_plastica',
    'tetrapak',
    'organico',
    'electronicos_pequenos',
    'no_reciclable',
]

# ── Mapeo clase → (categoría, color_recipiente, emoji, consejo) ──────────────
COLOR_MAP = {
    'lata':                 ('Metal',       'Amarillo', '🟡', 'Enjuagar y aplastar antes de depositar.'),
    'botella_pet':          ('Plástico',    'Azul',     '🔵', 'Quitar el tapón y aplastar la botella.'),
    'botella_vidrio':       ('Vidrio',      'Verde',    '🟢', 'No romper. Depositar entero en el recipiente.'),
    'papel':                ('Papel',       'Gris',     '⚫', 'Mantener seco y limpio. No mezclar con papel sucio.'),
    'carton':               ('Papel',       'Gris',     '⚫', 'Doblar y atar si es de gran tamaño.'),
    'bolsa_plastica':       ('Plástico',    'Azul',     '🔵', 'Vaciar completamente antes de depositar.'),
    'tetrapak':             ('Mixto',       'Azul',     '🔵', 'Enjuagar, aplastar y retirar la pajilla.'),
    'organico':             ('Orgánico',    'Café',     '🟤', 'Ideal para compostaje casero o biogás.'),
    'electronicos_pequenos':('Electrónico', 'Rojo',     '🔴', 'Llevar a un punto limpio autorizado. No tirar a la basura general.'),
    'no_reciclable':        ('Otros',       'Negro',    '⬛', 'Depositar en la basura general.'),
}

DISCLAIMER = (
    "⚠️ Este sistema es una herramienta de apoyo educativo. "
    "Las recomendaciones son orientativas y pueden variar según el municipio."
)


def load_model(path: str = 'models/modelo_reciclaje.pth') -> torch.nn.Module:
    """
    Carga el modelo MobileNetV2 con los pesos entrenados.

    Args:
        path: Ruta al archivo .pth con los pesos del modelo entrenado.

    Returns:
        Modelo en modo evaluación listo para inferencia.
    """
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = torch.nn.Linear(1280, len(CLASSES))
    model.load_state_dict(torch.load(path, map_location='cpu'))
    model.eval()
    return model


def build_model() -> torch.nn.Module:
    """
    Construye el modelo MobileNetV2 con pesos preentrenados de ImageNet
    para Transfer Learning (usar durante el entrenamiento).

    Returns:
        Modelo con base congelada y clasificador reemplazado.
    """
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

    # Congelar todas las capas base (Fase 1)
    for param in model.parameters():
        param.requires_grad = False

    # Reemplazar clasificador final con el número de clases del proyecto
    model.classifier[1] = torch.nn.Linear(1280, len(CLASSES))

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
