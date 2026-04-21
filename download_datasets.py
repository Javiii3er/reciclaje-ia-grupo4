# -*- coding: utf-8 -*-
"""
download_datasets.py — Descarga automática de datasets públicos para el proyecto.
Proyecto 01: Reciclaje Inteligente · Grupo #4 · UMG 2026

Requisitos previos (para descarga automática con Kaggle):
    1. Tener cuenta en Kaggle: https://www.kaggle.com
    2. Ir a: Account → Settings → API → Create New Token → descarga kaggle.json
    3. Colocar kaggle.json en:
       - Windows: C:/Users/TU_USUARIO/.kaggle/kaggle.json
       - Linux/Mac: ~/.kaggle/kaggle.json
    4. pip install kaggle

Uso:
    python download_datasets.py                  # descarga automática (requiere kaggle.json)
    python download_datasets.py --solo-estructura # solo crea carpetas, sin descargar
    python download_datasets.py --organizar       # organiza ZIPs ya descargados manualmente
"""

import os
import sys
import shutil
import argparse
import zipfile
import glob

# Forzar UTF-8 en la consola de Windows para mostrar emojis correctamente
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass  # Python < 3.7

# ── Configuración ─────────────────────────────────────────────────────────────
DATA_RAW = os.path.join('data', 'raw')
DATA_TMP = os.path.join('data', 'tmp')

CLASES = [
    'lata', 'botella_pet', 'botella_vidrio', 'papel',
    'carton', 'bolsa_plastica', 'tetrapak', 'organico',
    'electronicos_pequenos', 'no_reciclable'
]

# Mapeo extendido: nombre de carpeta en el dataset → clase del proyecto
MAPEO_CLASES = {
    # TrashNet / Garbage Classification
    'metal':          'lata',
    'plastic':        'botella_pet',
    'glass':          'botella_vidrio',
    'paper':          'papel',
    'cardboard':      'carton',
    'trash':          'no_reciclable',
    # Recycling Image Dataset (Kaggle - mostafaabla)
    'can':            'lata',
    'bottle':         'botella_pet',
    'white-glass':    'botella_vidrio',
    'brown-glass':    'botella_vidrio',
    'green-glass':    'botella_vidrio',
    'clothes':        'no_reciclable',
    'shoes':          'no_reciclable',
    'battery':        'electronicos_pequenos',
    'biological':     'organico',
    'organic':        'organico',
    # Mapeos específicos para Tetrapak y Bolsas
    'milk_carton':    'tetrapak',
    'juice_carton':   'tetrapak',
    'beverage_carton':'tetrapak',
    'tetra_pak':      'tetrapak',
    'plastic_bag':    'bolsa_plastica',
    'plastic-bag':    'bolsa_plastica',
    'bags':           'bolsa_plastica',
    'polythene':      'bolsa_plastica',
    # Nombres directos (ya coinciden)
    'lata':                 'lata',
    'botella_pet':          'botella_pet',
    'botella_vidrio':       'botella_vidrio',
    'papel':                'papel',
    'carton':               'carton',
    'bolsa_plastica':       'bolsa_plastica',
    'tetrapak':             'tetrapak',
    'organico':             'organico',
    'electronicos_pequenos':'electronicos_pequenos',
    'no_reciclable':        'no_reciclable',
}

# Datasets disponibles en Kaggle
KAGGLE_DATASETS = [
    {
        'nombre':     'Garbage Classification (asdasdasasdas)',
        'slug':       'asdasdasasdas/garbage-classification',
        'descripcion':'~2,500 imgs — 6 clases (metal, plastic, glass, paper, cardboard, trash)',
        'url_manual': 'https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification',
    },
    {
        'nombre':     'Recycling Image Dataset (mostafaabla)',
        'slug':       'mostafaabla/garbage-classification',
        'descripcion':'~15,000 imgs — incluye variantes de vidrio, orgánicos y algunos envases cartón',
        'url_manual': 'https://www.kaggle.com/datasets/mostafaabla/garbage-classification',
    },
    {
        'nombre':     'Litter Detection (vencerlanz09)',
        'slug':       'vencerlanz09/litter-detection-plastic-bottle-and-bag',
        'descripcion':'~2,000 imgs — Especial para bolsas plásticas (bolsa_plastica)',
        'url_manual': 'https://www.kaggle.com/datasets/vencerlanz09/litter-detection-plastic-bottle-and-bag',
    },
]


# ── Funciones de utilidad ─────────────────────────────────────────────────────

def crear_estructura():
    """Crea las carpetas data/raw/<clase>/ para todas las clases."""
    for clase in CLASES:
        os.makedirs(os.path.join(DATA_RAW, clase), exist_ok=True)
    os.makedirs(DATA_TMP, exist_ok=True)
    print("✅ Estructura de carpetas creada:")
    for clase in CLASES:
        print(f"   └── data/raw/{clase}/")


def verificar_kaggle():
    """Verifica que kaggle CLI esté disponible y configurado (compatible con Windows)."""
    # En Windows el redirect es diferente
    devnull = 'NUL' if sys.platform == 'win32' else '/dev/null'
    resultado = os.system(f'kaggle --version > {devnull} 2>&1')
    if resultado != 0:
        print("❌ kaggle CLI no encontrado.")
        print("   Instala con: pip install kaggle")
        return False

    # Verificar kaggle.json en ambas ubicaciones posibles
    rutas_json = [
        os.path.expanduser(r'~\.kaggle\kaggle.json'),
        os.path.expanduser('~/.kaggle/kaggle.json'),
    ]
    if not any(os.path.exists(r) for r in rutas_json):
        print("❌ kaggle.json no encontrado.")
        print("   Descárgalo desde: Kaggle → Settings → API → Create New Token")
        print(f"   Colócalo en: {rutas_json[0]}")
        return False

    print("✅ Kaggle CLI configurado correctamente.")
    return True


def descargar_dataset_kaggle(dataset_info: dict) -> bool:
    """Descarga un dataset de Kaggle al directorio temporal."""
    print(f"\n📥 Descargando: {dataset_info['nombre']}")
    print(f"   {dataset_info['descripcion']}")
    comando = f"kaggle datasets download -d {dataset_info['slug']} -p {DATA_TMP} --unzip"
    resultado = os.system(comando)
    if resultado == 0:
        print(f"✅ Descargado correctamente.")
        return True
    else:
        print(f"❌ Error al descargar. URL manual: {dataset_info['url_manual']}")
        return False


def descomprimir_zips_manuales():
    """
    Descomprime cualquier archivo .zip encontrado en data/tmp/.
    Permite flujo de trabajo manual: descargar ZIP desde navegador → colocar en data/tmp/
    """
    zips = glob.glob(os.path.join(DATA_TMP, '*.zip'))
    if not zips:
        print("   No se encontraron archivos ZIP en data/tmp/")
        return

    for zip_path in zips:
        nombre = os.path.basename(zip_path)
        print(f"   📦 Descomprimiendo {nombre}...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(DATA_TMP)
            os.remove(zip_path)
            print(f"   ✅ {nombre} descomprimido.")
        except Exception as e:
            print(f"   ❌ Error descomprimiendo {nombre}: {e}")


def organizar_imagenes():
    """
    Recorre data/tmp/ recursivamente y copia imágenes a data/raw/<clase>/
    según el mapeo de nombres de carpeta.
    """
    print("\n📂 Organizando imágenes por clase...")
    contador = {clase: 0 for clase in CLASES}
    ya_existentes = {
        clase: len([
            f for f in os.listdir(os.path.join(DATA_RAW, clase))
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        for clase in CLASES
        if os.path.exists(os.path.join(DATA_RAW, clase))
    }

    for root, dirs, files in os.walk(DATA_TMP):
        for fname in files:
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                continue

            carpeta_padre = os.path.basename(root).lower().strip()
            clase_destino = MAPEO_CLASES.get(carpeta_padre)

            if clase_destino is None:
                # Intentar buscar coincidencia parcial
                for key in MAPEO_CLASES:
                    if key in carpeta_padre or carpeta_padre in key:
                        clase_destino = MAPEO_CLASES[key]
                        break

            if clase_destino is None:
                continue

            # Nombre único para evitar sobreescribir
            base_idx = ya_existentes.get(clase_destino, 0) + contador[clase_destino] + 1
            nuevo_nombre = f"{clase_destino}_{base_idx:05d}.jpg"
            src = os.path.join(root, fname)
            dst = os.path.join(DATA_RAW, clase_destino, nuevo_nombre)

            try:
                shutil.copy2(src, dst)
                contador[clase_destino] += 1
            except Exception:
                pass

    # Resumen
    print("\n📊 Resultado de la organización:")
    print(f"{'Clase':<28} {'Existían':>8} {'Nuevas':>8} {'Total':>8} {'Estado':>12}")
    print("-" * 68)
    total_nuevas = 0
    for clase in CLASES:
        existian = ya_existentes.get(clase, 0)
        nuevas   = contador[clase]
        total    = existian + nuevas
        total_nuevas += nuevas
        estado = "✅ OK" if total >= 50 else "⚠️  Insuficiente"
        if total == 0:
            estado = "❌ Vacía"
        print(f"  {clase:<26} {existian:>8} {nuevas:>8} {total:>8}   {estado}")
    print("-" * 68)
    print(f"  {'TOTAL':<26} {sum(ya_existentes.values()):>8} {total_nuevas:>8} {sum(ya_existentes.values()) + total_nuevas:>8}")

    # Limpiar tmp
    if os.path.exists(DATA_TMP):
        shutil.rmtree(DATA_TMP)
        os.makedirs(DATA_TMP, exist_ok=True)
        print("\n🧹 Carpeta temporal limpiada.")

    # Clases con datos insuficientes
    vacias = [c for c in CLASES if (ya_existentes.get(c, 0) + contador[c]) < 50]
    if vacias:
        print("\n⚠️  Clases con menos de 50 imágenes (requieren fotos propias):")
        for c in vacias:
            faltan = 50 - (ya_existentes.get(c, 0) + contador[c])
            print(f"   → {c}: necesitas al menos {faltan} imágenes más")


def instrucciones_descarga_manual():
    """Muestra instrucciones detalladas para descarga manual."""
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║         DESCARGA MANUAL — Alternativa sin Kaggle CLI                ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  OPCIÓN A — Descarga directa desde Kaggle (navegador):               ║
║                                                                      ║
║  1. Garbage Classification (~2,500 imágenes):                        ║
║     https://www.kaggle.com/datasets/asdasdasasdas/                   ║
║     garbage-classification                                           ║
║                                                                      ║
║  2. Recycling Image Dataset (~15,000 imágenes):                      ║
║     https://www.kaggle.com/datasets/mostafaabla/                     ║
║     garbage-classification                                           ║
║                                                                      ║
║  3. Descarga los ZIP y colócalos en:  data/tmp/                      ║
║                                                                      ║
║  4. Ejecuta:  python download_datasets.py --organizar                ║
║     (descomprime y organiza automáticamente)                         ║
║                                                                      ║
║  OPCIÓN B — TrashNet (GitHub, sin cuenta):                           ║
║     https://github.com/garythung/trashnet                            ║
║     → Descarga "dataset-resized-original.zip" desde Releases        ║
║     → Coloca el ZIP en data/tmp/                                     ║
║     → Ejecuta: python download_datasets.py --organizar               ║
║                                                                      ║
║  OPCIÓN C — Imágenes propias del equipo:                             ║
║     Fotografíen residuos reales y coloquen las imágenes              ║
║     directamente en las carpetas:                                    ║
║       data/raw/tetrapak/                                             ║
║       data/raw/electronicos_pequenos/                                ║
║       data/raw/bolsa_plastica/                                       ║
║       data/raw/no_reciclable/                                        ║
║     Meta: mínimo 80 imágenes por clase para la entrega final.        ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
""")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Descarga y organización de datasets — Reciclaje Inteligente · Grupo #4'
    )
    parser.add_argument(
        '--solo-estructura',
        action='store_true',
        help='Solo crear las carpetas de datos, sin descargar nada'
    )
    parser.add_argument(
        '--organizar',
        action='store_true',
        help='Organizar ZIPs ya descargados manualmente en data/tmp/'
    )
    args = parser.parse_args()

    print("=" * 70)
    print("  DESCARGA DE DATASETS — Reciclaje Inteligente · Grupo #4 · UMG 2026")
    print("=" * 70)

    # Siempre crear la estructura de carpetas
    crear_estructura()

    if args.solo_estructura:
        print("\n✅ Modo --solo-estructura: carpetas creadas, sin descargas.")
        instrucciones_descarga_manual()
        return

    if args.organizar:
        print("\n📦 Modo --organizar: procesando ZIPs en data/tmp/...")
        descomprimir_zips_manuales()
        organizar_imagenes()
    else:
        # Modo automático con Kaggle CLI
        if verificar_kaggle():
            print("\n🚀 Iniciando descarga de datasets desde Kaggle...")
            alguno_ok = False
            for ds in KAGGLE_DATASETS:
                ok = descargar_dataset_kaggle(ds)
                alguno_ok = alguno_ok or ok

            if alguno_ok:
                organizar_imagenes()
            else:
                print("\n❌ No se pudo descargar ningún dataset automáticamente.")
                instrucciones_descarga_manual()
        else:
            instrucciones_descarga_manual()

    print("\n" + "=" * 70)
    print("  ✅ Proceso finalizado.")
    print("  Siguiente paso: python src/split_dataset.py")
    print("  Luego:          python src/train.py --data_dir data/processed")
    print("=" * 70)


if __name__ == '__main__':
    main()
