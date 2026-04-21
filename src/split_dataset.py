# -*- coding: utf-8 -*-
"""
split_dataset.py — Split estratificado del dataset en train / val / test.
Proyecto 01: Reciclaje Inteligente · Grupo #4 · UMG 2026

Lee imágenes de data/raw/<clase>/ y las distribuye en:
    data/processed/train/<clase>/   → 70%
    data/processed/val/<clase>/     → 15%
    data/processed/test/<clase>/    → 15%

La división es estratificada por clase (mismo porcentaje de cada clase en cada split).

Uso:
    python src/split_dataset.py
    python src/split_dataset.py --raw_dir data/raw --out_dir data/processed
    python src/split_dataset.py --seed 123           # semilla diferente para reproducibilidad
"""

import os
import sys
import shutil
import argparse
import random
from pathlib import Path
from collections import defaultdict

# Forzar UTF-8 en la consola de Windows para mostrar emojis correctamente
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

# ── Compatibilidad: scikit-learn opcional ────────────────────────────────────
try:
    from sklearn.model_selection import train_test_split
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False
    print("⚠️  scikit-learn no encontrado. Usando split manual (instala con: pip install scikit-learn)")

# ── Constantes ────────────────────────────────────────────────────────────────
EXTENSIONES_VALIDAS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp',
                       '.JPG', '.JPEG', '.PNG', '.BMP', '.WEBP'}

SPLITS = {
    'train': 0.70,
    'val':   0.15,
    'test':  0.15,
}

CLASES = [
    'lata', 'botella_pet', 'botella_vidrio', 'papel',
    'carton', 'bolsa_plastica', 'tetrapak', 'organico',
    'electronicos_pequenos', 'no_reciclable'
]

MIN_IMAGENES_CLASE = 10  # mínimo para poder hacer split


def split_manual(items: list, train_ratio: float, val_ratio: float, seed: int):
    """
    Split manual sin sklearn: mezcla la lista y divide por proporciones.
    """
    rng = random.Random(seed)
    items = items[:]
    rng.shuffle(items)
    n = len(items)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)
    return items[:n_train], items[n_train:n_train + n_val], items[n_train + n_val:]


def split_estratificado(items: list, train_ratio: float, val_ratio: float, seed: int):
    """
    Split estratificado usando sklearn o fallback manual.
    """
    if len(items) < MIN_IMAGENES_CLASE:
        return items, [], []

    test_ratio = 1.0 - train_ratio - val_ratio

    if SKLEARN_OK and len(items) >= 4:
        try:
            train, temp = train_test_split(items, test_size=(1 - train_ratio), random_state=seed)
            # Proporción relativa de val dentro del bloque que no es train
            val_ratio_rel = val_ratio / (val_ratio + test_ratio) if (val_ratio + test_ratio) > 0 else 0.5
            val, test = train_test_split(temp, test_size=(1 - val_ratio_rel), random_state=seed)
            return train, val, test
        except ValueError:
            pass  # Si falla (pocas imágenes), usar split manual

    return split_manual(items, train_ratio, val_ratio, seed)


def copiar_imagenes(rutas: list, destino: Path, clase: str, prefijo: str = ''):
    """
    Copia una lista de rutas de imagen a destino/<clase>/.
    Retorna la cantidad copiada.
    """
    destino_clase = destino / clase
    destino_clase.mkdir(parents=True, exist_ok=True)
    copiadas = 0

    for i, ruta in enumerate(rutas):
        ruta = Path(ruta)
        nuevo_nombre = f"{clase}_{prefijo}{i + 1:05d}{ruta.suffix.lower()}"
        try:
            shutil.copy2(ruta, destino_clase / nuevo_nombre)
            copiadas += 1
        except Exception as e:
            print(f"    ⚠️  Error copiando {ruta.name}: {e}")

    return copiadas


def main():
    parser = argparse.ArgumentParser(
        description='Split estratificado del dataset — Reciclaje Inteligente · Grupo #4'
    )
    parser.add_argument('--raw_dir', type=str, default='data/raw',
                        help='Directorio con imágenes por clase (default: data/raw)')
    parser.add_argument('--out_dir', type=str, default='data/processed',
                        help='Directorio de salida (default: data/processed)')
    parser.add_argument('--train',   type=float, default=0.70, help='Proporción train (default: 0.70)')
    parser.add_argument('--val',     type=float, default=0.15, help='Proporción val   (default: 0.15)')
    parser.add_argument('--seed',    type=int,   default=42,   help='Semilla aleatoria (default: 42)')
    parser.add_argument('--limpiar', action='store_true',
                        help='Borrar data/processed/ antes de ejecutar')
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    test_ratio = 1.0 - args.train - args.val

    print("=" * 70)
    print("  SPLIT DE DATASET — Reciclaje Inteligente · Grupo #4 · UMG 2026")
    print("=" * 70)
    print(f"\n  Origen:    {raw_dir.resolve()}")
    print(f"  Destino:   {out_dir.resolve()}")
    print(f"  Split:     train={args.train:.0%} | val={args.val:.0%} | test={test_ratio:.0%}")
    print(f"  Semilla:   {args.seed}")
    print(f"  sklearn:   {'✅ disponible' if SKLEARN_OK else '⚠️  no disponible (split manual)'}")

    # Validaciones
    if not raw_dir.exists():
        print(f"\n❌ No se encontró el directorio: {raw_dir}")
        print("   Ejecuta primero: python download_datasets.py")
        sys.exit(1)

    if abs(args.train + args.val + test_ratio - 1.0) > 1e-6:
        print(f"\n❌ Las proporciones no suman 1.0: train={args.train} + val={args.val} + test={test_ratio:.2f}")
        sys.exit(1)

    # Limpiar si se pide
    if args.limpiar and out_dir.exists():
        shutil.rmtree(out_dir)
        print(f"\n🧹 {out_dir} eliminado.")

    # Crear subdirectorios de salida
    for split in SPLITS:
        (out_dir / split).mkdir(parents=True, exist_ok=True)

    # ── Procesar cada clase ───────────────────────────────────────────────────
    print(f"\n{'Clase':<28} {'Total':>6} {'Train':>6} {'Val':>5} {'Test':>5} {'Estado':>14}")
    print("-" * 70)

    resumen = defaultdict(int)
    clases_ok = []
    clases_vacias = []
    clases_pocas = []

    clases_presentes = sorted([
        d.name for d in raw_dir.iterdir()
        if d.is_dir() and d.name in CLASES
    ])

    if not clases_presentes:
        print("\n❌ No se encontraron carpetas de clases en", raw_dir)
        print("   ¿Ya ejecutaste python download_datasets.py?")
        sys.exit(1)

    for clase in CLASES:
        clase_dir = raw_dir / clase

        if not clase_dir.exists():
            print(f"  {'⚠️':<3} {clase:<25} {'—':>6} {'—':>6} {'—':>5} {'—':>5}   ❌ No existe")
            clases_vacias.append(clase)
            continue

        imagenes = sorted([
            p for p in clase_dir.iterdir()
            if p.suffix in EXTENSIONES_VALIDAS and p.is_file()
        ])

        n_total = len(imagenes)

        if n_total == 0:
            print(f"  {'⚠️':<3} {clase:<25} {0:>6} {'—':>6} {'—':>5} {'—':>5}   ❌ Vacía")
            clases_vacias.append(clase)
            continue

        if n_total < MIN_IMAGENES_CLASE:
            print(f"  {'⚠️':<3} {clase:<25} {n_total:>6} {'—':>6} {'—':>5} {'—':>5}   ⚠️  < {MIN_IMAGENES_CLASE} imgs")
            clases_pocas.append((clase, n_total))
            continue

        # Split
        train_imgs, val_imgs, test_imgs = split_estratificado(
            imagenes, args.train, args.val, args.seed
        )

        # Copiar
        n_train = copiar_imagenes(train_imgs, out_dir / 'train', clase, prefijo='tr')
        n_val   = copiar_imagenes(val_imgs,   out_dir / 'val',   clase, prefijo='vl')
        n_test  = copiar_imagenes(test_imgs,  out_dir / 'test',  clase, prefijo='ts')

        resumen['train'] += n_train
        resumen['val']   += n_val
        resumen['test']  += n_test

        estado = "✅ OK" if n_total >= 80 else "⚠️  Pocas"
        print(f"  {'✅':<3} {clase:<25} {n_total:>6} {n_train:>6} {n_val:>5} {n_test:>5}   {estado}")
        clases_ok.append(clase)

    print("-" * 70)
    print(f"  {'TOTAL':<28} {sum(resumen.values()):>6} {resumen['train']:>6} {resumen['val']:>5} {resumen['test']:>5}")

    # ── Resumen final ─────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  RESUMEN")
    print(f"{'='*70}")
    print(f"  ✅ Clases procesadas con éxito: {len(clases_ok)}")
    if clases_pocas:
        print(f"  ⚠️  Clases con pocas imágenes (no incluidas):")
        for c, n in clases_pocas:
            print(f"       → {c}: {n} imgs (mínimo requerido: {MIN_IMAGENES_CLASE})")
    if clases_vacias:
        print(f"  ❌ Clases sin imágenes: {clases_vacias}")
        print(f"     → Agrega fotos propias del equipo en data/raw/<clase>/")

    print(f"\n  📁 Dataset procesado guardado en: {out_dir.resolve()}")
    print(f"      ├── train/  — {resumen['train']} imágenes")
    print(f"      ├── val/    — {resumen['val']} imágenes")
    print(f"      └── test/   — {resumen['test']} imágenes")

    if clases_ok:
        print(f"\n  ✅ Siguiente paso:")
        print(f"     python src/train.py --data_dir data/processed")
    else:
        print(f"\n  ❌ No hay clases con suficientes imágenes. Agrega imágenes a data/raw/")


if __name__ == '__main__':
    main()
