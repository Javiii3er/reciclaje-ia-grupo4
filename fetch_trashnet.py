# -*- coding: utf-8 -*-
"""
fetch_parquet.py — Descarga TrashNet desde HuggingFace usando URLs directas de Parquet.
No requiere cuenta ni token. Usa archivos parquet públicos convertidos automáticamente.
Proyecto 01: Reciclaje Inteligente · Grupo #4 · UMG 2026

Los 4 parquets de garythung/trashnet totalizan ~2,527 imágenes en 6 clases.
"""

import os
import sys
import io
import time
from pathlib import Path

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

import requests
import pyarrow.parquet as pq
import pyarrow as pa
from PIL import Image

# ── URLs directas de los parquets públicos ────────────────────────────────────
# Rama: refs/convert/parquet  (generada automáticamente por HuggingFace)
BASE_URL = "https://huggingface.co/datasets/garythung/trashnet/resolve/refs%2Fconvert%2Fparquet/default/train"
PARQUET_FILES = [
    f"{BASE_URL}/0000.parquet",
    f"{BASE_URL}/0001.parquet",
    f"{BASE_URL}/0002.parquet",
    f"{BASE_URL}/0003.parquet",
]

# Mapeo TrashNet → clases del proyecto
MAPEO = {
    'glass':     'botella_vidrio',
    'paper':     'papel',
    'cardboard': 'carton',
    'plastic':   'botella_pet',
    'metal':     'lata',
    'trash':     'no_reciclable',
}

CLASES_PROYECTO = [
    'lata', 'botella_pet', 'botella_vidrio', 'papel', 'carton',
    'bolsa_plastica', 'tetrapak', 'organico',
    'electronicos_pequenos', 'no_reciclable'
]

DATA_RAW = Path('data/raw')
TMP_DIR  = Path('data/tmp_parquet')


def descargar_archivo(url: str, destino: Path, reintentos: int = 3) -> bool:
    """Descarga un archivo con reintentos y barra de progreso simple."""
    for intento in range(1, reintentos + 1):
        try:
            r = requests.get(url, stream=True, timeout=120,
                             headers={'User-Agent': 'Mozilla/5.0'})
            r.raise_for_status()

            total = int(r.headers.get('content-length', 0))
            descargado = 0
            bloque = 1024 * 512  # 512 KB

            with open(destino, 'wb') as f:
                for chunk in r.iter_content(chunk_size=bloque):
                    if chunk:
                        f.write(chunk)
                        descargado += len(chunk)
                        if total > 0:
                            pct = descargado / total * 100
                            mb  = descargado / 1024 / 1024
                            print(f"\r  Progreso: {pct:5.1f}%  ({mb:.1f} MB)", end='', flush=True)
            print()
            return True

        except requests.RequestException as e:
            print(f"\n  ⚠️  Intento {intento}/{reintentos} falló: {e}")
            if intento < reintentos:
                print(f"  Reintentando en 5 segundos...")
                time.sleep(5)
    return False


def procesar_parquet(parquet_path: Path, label_names: list, contador: dict):
    """Lee un parquet de TrashNet y guarda imágenes en data/raw/<clase>/"""
    table = pq.read_table(str(parquet_path))

    # Detectar columnas de imagen y etiqueta
    col_names = table.schema.names
    img_col   = next((c for c in col_names if 'image' in c.lower()), col_names[0])
    lbl_col   = next((c for c in col_names if 'label' in c.lower()), col_names[-1])

    print(f"  Columnas: {col_names}")
    print(f"  Usando imagen='{img_col}', etiqueta='{lbl_col}', filas={table.num_rows}")

    n_guardadas = 0
    n_errores   = 0

    for i in range(table.num_rows):
        try:
            # Etiqueta
            raw_lbl = table[lbl_col][i].as_py()
            if isinstance(raw_lbl, int):
                nombre_lbl = label_names[raw_lbl] if raw_lbl < len(label_names) else str(raw_lbl)
            else:
                nombre_lbl = str(raw_lbl).lower().strip()

            clase_destino = MAPEO.get(nombre_lbl)
            if clase_destino is None:
                continue

            # Imagen — puede ser dict {'bytes': ..., 'path': ...} ó bytes directo
            img_cel = table[img_col][i].as_py()
            if isinstance(img_cel, dict):
                img_bytes = img_cel.get('bytes') or img_cel.get('data')
            elif isinstance(img_cel, bytes):
                img_bytes = img_cel
            else:
                continue

            if not img_bytes:
                continue

            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')

            # Nombre único
            ya_hay = len(list((DATA_RAW / clase_destino).glob('*.jpg')))
            contador[clase_destino] += 1
            fname = f"{clase_destino}_{ya_hay + contador[clase_destino]:05d}.jpg"
            img.save(str(DATA_RAW / clase_destino / fname), 'JPEG', quality=95)
            n_guardadas += 1

        except Exception:
            n_errores += 1
            continue

        if (i + 1) % 300 == 0:
            print(f"  [{i+1}/{table.num_rows}] guardadas: {n_guardadas}, errores: {n_errores}")

    print(f"  Parcial guardadas: {n_guardadas} | errores: {n_errores}")
    return n_guardadas


def main():
    print("=" * 65)
    print("  DESCARGA TrashNet (parquets HuggingFace) — Grupo #4 · UMG 2026")
    print("=" * 65)

    # Crear carpetas
    for clase in CLASES_PROYECTO:
        (DATA_RAW / clase).mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    # Etiquetas de TrashNet (orden en el dataset)
    # Verificado: 0=cardboard,1=glass,2=metal,3=paper,4=plastic,5=trash
    label_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
    print(f"\n  Clases TrashNet → Proyecto:")
    for tn, pr in MAPEO.items():
        print(f"    {tn:<12} → {pr}")

    contador = {c: 0 for c in CLASES_PROYECTO}
    total_guardadas = 0

    for idx, url in enumerate(PARQUET_FILES):
        fname = f"trashnet_{idx:04d}.parquet"
        dest  = TMP_DIR / fname

        print(f"\n{'='*65}")
        print(f"  Parquet {idx+1}/{len(PARQUET_FILES)}: {url.split('/')[-1]}")
        print(f"{'='*65}")

        if dest.exists() and dest.stat().st_size > 1000:
            print(f"  ✅ Ya descargado: {dest.stat().st_size / 1024 / 1024:.1f} MB")
        else:
            print(f"  📥 Descargando...")
            ok = descargar_archivo(url, dest)
            if not ok:
                print(f"  ❌ No se pudo descargar. Continuando con el siguiente...")
                continue

        print(f"  🖼️  Procesando imágenes...")
        n = procesar_parquet(dest, label_names, contador)
        total_guardadas += n

    # ── Resumen ───────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  RESULTADO FINAL")
    print("=" * 65)
    print(f"\n  {'Clase':<28} {'Imágenes en raw':>18}  {'Estado'}")
    print("  " + "-" * 55)

    total = 0
    for clase in CLASES_PROYECTO:
        n = len(list((DATA_RAW / clase).glob('*.jpg')))
        estado = "✅ OK" if n >= 80 else ("⚠️  Pocas" if n >= 20 else ("❌ Vacía" if n == 0 else f"⚠️  {n}"))
        print(f"  {clase:<28} {n:>18}   {estado}")
        total += n

    print("  " + "-" * 55)
    print(f"  {'TOTAL':<28} {total:>18}")

    clases_vacias = [c for c in CLASES_PROYECTO if len(list((DATA_RAW / c).glob('*.jpg'))) == 0]
    if clases_vacias:
        print(f"\n  ⚠️  Estas clases necesitan fotos propias del equipo:")
        for c in clases_vacias:
            print(f"     → data/raw/{c}/")

    # Limpiar tmp
    import shutil
    if TMP_DIR.exists():
        shutil.rmtree(TMP_DIR)
        print(f"\n  🧹 Carpeta temporal eliminada.")

    if total > 0:
        print(f"\n  ✅ Siguiente paso:")
        print(f"     python -X utf8 src/split_dataset.py")
    else:
        print(f"\n  ❌ No se pudieron obtener imágenes.")

    print("=" * 65)


if __name__ == '__main__':
    main()
