"""
train.py — Entrenamiento de MobileNetV2 con Transfer Learning, Reentrenamiento y Early Stopping.
Proyecto 01: Reciclaje Inteligente · Grupo #4 · UMG 2026

Uso:
    python src/train.py --data_dir data/processed --epochs 25 --patience 5
"""

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

from model import build_model, load_model, unfreeze_last_layers, CLASSES
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


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    preds, targets = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item() * imgs.size(0)
            preds.extend(outputs.argmax(1).cpu().tolist())
            targets.extend(labels.tolist())
            total += imgs.size(0)
            
    val_loss = total_loss / total
    val_acc = sum(p == t for p, t in zip(preds, targets)) / len(targets)
    f1 = f1_score(targets, preds, average='macro', zero_division=0)
    return val_loss, val_acc, f1


def plot_curves(history, save_path='docs/curva_aprendizaje.png'):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'],   label='Val')
    axes[0].set_title('Pérdida (Cross Entropy)')
    axes[0].set_xlabel('Época')
    axes[0].set_ylabel('Loss')
    axes[0].legend()

    axes[1].plot(history['train_acc'], label='Train')
    axes[1].plot(history['val_acc'],   label='Val')
    axes[1].set_title('Accuracy (Precisión)')
    axes[1].set_xlabel('Época')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Curva de aprendizaje actualizada en {save_path}")


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {device}")

    # ── Datasets ─────────────────────────────────────────────────────────────
    train_ds = ImageFolder(os.path.join(args.data_dir, 'train'), get_train_transforms())
    val_ds   = ImageFolder(os.path.join(args.data_dir, 'val'),   get_val_transforms())

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False, num_workers=2)

    n_clases = len(train_ds.classes)
    print(f"Clases detectadas: {train_ds.classes} ({n_clases} clases)")

    # ── Modelo ────────────────────────────────────────────────────────────────
    model_path = 'models/modelo_reciclaje.pth'
    if os.path.exists(model_path):
        print(f"🔄 Cargando modelo existente para REENTRENAMIENTO: {model_path}")
        model = load_model(model_path, n_classes=n_clases).to(device)
    else:
        print(f"🆕 No se encontró modelo previo. Construyendo desde cero...")
        model = build_model(n_classes=n_clases).to(device)

    # Pesos por clase
    class_counts = [train_ds.targets.count(i) for i in range(n_clases)]
    total = sum(class_counts)
    weights = torch.tensor([total / (n_clases * max(c, 1)) for c in class_counts]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_f1 = 0.0
    epochs_no_improve = 0
    patience = args.patience

    # ── ENTRENAMIENTO (Fine-tuning directo si es reentrenamiento) ─────────────
    print(f"\n{'='*60}")
    print(f"INICIANDO REENTRENAMIENTO (Max {args.epochs} épocas, Paciencia {patience})")
    print(f"{'='*60}")

    # Desbloqueamos las últimas capas para un ajuste fino
    model = unfreeze_last_layers(model, n=30) 
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        v_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, device)
        scheduler.step(1 - val_f1)

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(v_loss)
        history['train_acc'].append(tr_acc)
        history['val_acc'].append(val_acc)

        print(f"Época {epoch:2d}/{args.epochs} | Loss: {tr_loss:.4f} | Val Loss: {v_loss:.4f} | Val F1: {val_f1:.3f}")

        # Guardar gráfico de progreso en cada época
        plot_curves(history)

        # Lógica de Early Stopping y Guardado del mejor
        if val_f1 > best_f1:
            best_f1 = val_f1
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_path)
            print(f"  ⭐ ¡Nuevo mejor modelo guardado! (F1: {best_f1:.3f})")
        else:
            epochs_no_improve += 1
            print(f"  [EarlyStopping] Sin mejora por {epochs_no_improve}/{patience} épocas.")

        if epochs_no_improve >= patience:
            print(f"\n🛑 Early Stopping activado. El modelo dejó de mejorar.")
            break

    print(f"\n✅ Proceso finalizado. Mejor F1-macro alcanzado: {best_f1:.3f}")
    plot_curves(history)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Entrenamiento con Early Stopping')
    parser.add_argument('--data_dir', type=str, default='data/processed')
    parser.add_argument('--epochs',   type=int, default=25)
    parser.add_argument('--patience', type=int, default=5)
    args = parser.parse_args()
    main(args)
