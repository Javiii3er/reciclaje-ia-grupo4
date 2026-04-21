"""
train.py — Entrenamiento de MobileNetV2 con Transfer Learning en 2 fases.
Proyecto 01: Reciclaje Inteligente · Grupo #4 · UMG 2026

Uso:
    python src/train.py --data_dir data/processed --epochs_fase1 5 --epochs_fase2 10
"""

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

from model import build_model, unfreeze_last_layers, CLASSES
from preprocess import get_train_transforms, get_val_transforms


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += imgs.size(0)
    return total_loss / total, correct / total


def evaluate(model, loader, device):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            preds.extend(outputs.argmax(1).cpu().tolist())
            targets.extend(labels.tolist())
    acc = sum(p == t for p, t in zip(preds, targets)) / len(targets)
    f1 = f1_score(targets, preds, average='macro', zero_division=0)
    return acc, f1


def plot_curves(history, save_path='docs/curva_aprendizaje.png'):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'],   label='Val')
    axes[0].set_title('Pérdida por época')
    axes[0].set_xlabel('Época')
    axes[0].set_ylabel('Loss')
    axes[0].legend()

    axes[1].plot(history['train_acc'], label='Train')
    axes[1].plot(history['val_acc'],   label='Val')
    axes[1].set_title('Accuracy por época')
    axes[1].set_xlabel('Época')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Curva de aprendizaje guardada en {save_path}")


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {device}")

    # ── Datasets ─────────────────────────────────────────────────────────────
    train_ds = ImageFolder(os.path.join(args.data_dir, 'train'), get_train_transforms())
    val_ds   = ImageFolder(os.path.join(args.data_dir, 'val'),   get_val_transforms())

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False, num_workers=2)

    print(f"Clases detectadas: {train_ds.classes}")
    print(f"Imágenes train: {len(train_ds)} | val: {len(val_ds)}")

    # Usar las clases realmente presentes en el dataset (ImageFolder detecta automáticamente)
    n_clases = len(train_ds.classes)
    print(f"Clases con datos: {train_ds.classes} ({n_clases} clases)")

    # ── Modelo ────────────────────────────────────────────────────────────────
    model = build_model(n_classes=n_clases).to(device)

    # Pesos por clase para manejar desbalance
    # Usar las clases realmente presentes en el dataset (ImageFolder detecta automáticamente)
    n_clases = len(train_ds.classes)
    class_counts = [train_ds.targets.count(i) for i in range(n_clases)]
    total = sum(class_counts)
    weights = torch.tensor([total / (n_clases * max(c, 1)) for c in class_counts]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_f1 = 0.0

    # ── FASE 1: Solo clasificador ─────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"FASE 1 — Entrenando solo el clasificador ({args.epochs_fase1} épocas, lr=1e-3)")
    print(f"{'='*50}")

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)

    for epoch in range(1, args.epochs_fase1 + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_acc, val_f1 = evaluate(model, val_loader, device)
        scheduler.step(1 - val_f1)

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(1 - val_acc)
        history['train_acc'].append(tr_acc)
        history['val_acc'].append(val_acc)

        print(f"Época {epoch:2d}/{args.epochs_fase1} | "
              f"Loss: {tr_loss:.4f} | Train acc: {tr_acc:.3f} | "
              f"Val acc: {val_acc:.3f} | Val F1-macro: {val_f1:.3f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), 'models/modelo_reciclaje.pth')
            print(f"  → Mejor modelo guardado (F1={best_f1:.3f})")

    # ── FASE 2: Fine-tuning ───────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"FASE 2 — Fine-tuning últimas 20 capas ({args.epochs_fase2} épocas, lr=1e-4)")
    print(f"{'='*50}")

    model = unfreeze_last_layers(model, n=20)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)

    for epoch in range(1, args.epochs_fase2 + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_acc, val_f1 = evaluate(model, val_loader, device)
        scheduler.step(1 - val_f1)

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(1 - val_acc)
        history['train_acc'].append(tr_acc)
        history['val_acc'].append(val_acc)

        print(f"Época {epoch:2d}/{args.epochs_fase2} | "
              f"Loss: {tr_loss:.4f} | Train acc: {tr_acc:.3f} | "
              f"Val acc: {val_acc:.3f} | Val F1-macro: {val_f1:.3f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), 'models/modelo_reciclaje.pth')
            print(f"  → Mejor modelo guardado (F1={best_f1:.3f})")

    print(f"\n✅ Entrenamiento completo. Mejor F1-macro: {best_f1:.3f}")
    plot_curves(history)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Entrenamiento Reciclaje Inteligente')
    parser.add_argument('--data_dir',      type=str, default='data/processed')
    parser.add_argument('--epochs_fase1',  type=int, default=5)
    parser.add_argument('--epochs_fase2',  type=int, default=10)
    args = parser.parse_args()
    main(args)
