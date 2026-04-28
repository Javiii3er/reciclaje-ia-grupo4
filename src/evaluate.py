# -*- coding: utf-8 -*-
"""
evaluate.py — Evaluación detallada del modelo entrenado.
Proyecto 01: Reciclaje Inteligente · Grupo #4 · UMG 2026

Genera:
    1. Reporte de clasificación (Precisión, Recall, F1 por clase).
    2. Matriz de confusión visual (docs/matriz_confusion.png).
    3. Análisis de las clases con peor rendimiento.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from model import load_model, CLASSES
from preprocess import get_val_transforms

def evaluate(model_path, test_dir, device):
    print(f"🚀 Iniciando evaluación...")
    print(f"   Modelo: {model_path}")
    print(f"   Datos:   {test_dir}")

    # 1. Cargar datos de prueba
    transform = get_val_transforms()
    test_ds = ImageFolder(test_dir, transform=transform)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=2)
    
    actual_classes = test_ds.classes
    print(f"   Clases encontradas en test: {actual_classes}")

    # 2. Cargar modelo
    model = load_model(model_path, n_classes=len(actual_classes))
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    # 3. Inferencia
    print("   Realizando predicciones sobre el set de prueba...")
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # 4. Métricas
    acc = accuracy_score(all_labels, all_preds)
    report = classification_report(
        all_labels, 
        all_preds, 
        target_names=actual_classes,
        zero_division=0
    )
    
    print("\n" + "="*60)
    print(f"📊 RESULTADOS GLOBALES")
    print("="*60)
    print(f"Accuracy Total: {acc:.2%}")
    print("\nReporte por Clase:")
    print(report)

    # 5. Matriz de Confusión Visual
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=actual_classes,
        yticklabels=actual_classes
    )
    plt.title('Matriz de Confusión — Reciclaje Inteligente')
    plt.xlabel('Predicción (Modelo)')
    plt.ylabel('Clase Real (Humano)')
    
    # Guardar matriz
    os.makedirs('docs', exist_ok=True)
    save_path = 'docs/matriz_confusion.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Matriz de confusión guardada en: {save_path}")
    
    # 6. Identificar puntos débiles
    print("\n🔍 ANÁLISIS DE MEJORA")
    print("-" * 30)
    
    # Clases con más errores
    cm_diag = np.diag(cm)
    cm_sum_rows = np.sum(cm, axis=1)
    acc_per_class = cm_diag / cm_sum_rows
    
    worst_idx = np.argsort(acc_per_class)[:3]
    print("Clases que más necesitan fotos nuevas:")
    for idx in worst_idx:
        if acc_per_class[idx] < 1.0:
            print(f"  → {actual_classes[idx]}: {acc_per_class[idx]:.1%} de acierto")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Evaluación del modelo de reciclaje')
    parser.add_argument('--model', type=str, default='models/modelo_reciclaje.pth')
    parser.add_argument('--data',  type=str, default='data/processed/test')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if not os.path.exists(args.model):
        print(f"❌ Error: No se encuentra el modelo en {args.model}")
    elif not os.path.exists(args.data):
        print(f"❌ Error: No se encuentra la carpeta de test en {args.data}")
    else:
        evaluate(args.model, args.data, device)
