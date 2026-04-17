"""
preprocess.py — Preprocesamiento e data augmentation para el dataset.
Proyecto 01: Reciclaje Inteligente · Grupo #4 · UMG 2026
"""

from torchvision import transforms

# ── Parámetros de normalización ImageNet ─────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
IMAGE_SIZE    = 224


def get_train_transforms() -> transforms.Compose:
    """
    Transformaciones para el conjunto de entrenamiento.
    Incluye augmentation para reducir overfitting.

    Returns:
        Pipeline de transformaciones para imágenes de entrenamiento.
    """
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE + 32, IMAGE_SIZE + 32)),
        transforms.RandomCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transforms() -> transforms.Compose:
    """
    Transformaciones para validación y prueba (sin augmentation).

    Returns:
        Pipeline de transformaciones para imágenes de validación/prueba.
    """
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_inference_transforms() -> transforms.Compose:
    """
    Transformaciones para inferencia en la app Streamlit.
    Convierte una imagen PIL al tensor esperado por el modelo.

    Returns:
        Pipeline de transformaciones para inferencia en producción.
    """
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
