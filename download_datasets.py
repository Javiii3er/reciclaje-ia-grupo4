"""
download_datasets.py — Descarga automática de datasets públicos para el proyecto.
Proyecto 01: Reciclaje Inteligente · Grupo #4 · UMG 2026

Requisitos previos:
    1. Tener cuenta en Kaggle: https://www.kaggle.com
    2. Ir a: Account → API → Create New Token → descarga kaggle.json
    3. Colocar kaggle.json en:
       - Windows: C:\\Users\\TU_USUARIO\\.kaggle\\kaggle.json
       - Linux/Mac: ~/.kaggle/kaggle.json
    4. pip install kaggle

Uso:
    python download_datasets.py
"""

import os
import zipfile
import shutil

DATA_RAW = os.path.join('data', 'raw')

CLASES = [
    'lata', 'botella_pet', 'botella_vidrio', 'papel',
    'carton', 'bolsa_plastica', 'tetrapak', 'organico',
    'electronicos_pequenos', 'no_reciclable'
]

# Mapeo de clases TrashNet → nuestras clases
TRASHNET_MAP = {
    'metal':  'lata',
    'plastic':'botella_pet',
    'glass':  'botella_vidrio',
    'paper':  'papel',
    'cardboard': 'carton',
    'trash':  'no_reciclable',
}

def crear_estructura():
    """Crea las carpetas de destino por clase."""
    for clase in CLASES:
        os.makedirs(os.path.join(DATA_RAW, clase), exist_ok=True)
    print("✅ Estructura de carpetas creada en data/raw/")

def descargar_trashnet():
    """Descarga TrashNet desde Kaggle."""
    print("\n📥 Descargando TrashNet...")
    try:
        os.system('kaggle datasets download -d asdasdasasdas/garbage-classification -p data/tmp --unzip')
        print("✅ TrashNet descargado")
        return True
    except Exception as e:
        print(f"❌ Error descargando TrashNet: {e}")
        print("   → Descarga manual: https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification")
        return False

def descargar_garbage_classification():
    """Descarga Garbage Classification desde Kaggle."""
    print("\n📥 Descargando Garbage Classification...")
    try:
        os.system('kaggle datasets download -d mostafaabla/garbage-classification -p data/tmp --unzip')
        print("✅ Garbage Classification descargado")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        print("   → Descarga manual: https://www.kaggle.com/datasets/mostafaabla/garbage-classification")
        return False

def organizar_imagenes():
    """
    Mueve imágenes de data/tmp/ a data/raw/ según el mapeo de clases.
    Renombra archivos para evitar conflictos.
    """
    print("\n📂 Organizando imágenes en carpetas por clase...")
    tmp_dir = os.path.join('data', 'tmp')
    contador = {clase: 0 for clase in CLASES}

    for root, dirs, files in os.walk(tmp_dir):
        for fname in files:
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            # Determinar clase según el nombre de la carpeta padre
            carpeta_padre = os.path.basename(root).lower()
            clase_destino = None

            # Buscar coincidencia directa
            if carpeta_padre in CLASES:
                clase_destino = carpeta_padre
            # Buscar por mapeo TrashNet
            elif carpeta_padre in TRASHNET_MAP:
                clase_destino = TRASHNET_MAP[carpeta_padre]

            if clase_destino is None:
                continue

            # Copiar con nombre único
            contador[clase_destino] += 1
            nuevo_nombre = f"{clase_destino}_{contador[clase_destino]:04d}.jpg"
            src = os.path.join(root, fname)
            dst = os.path.join(DATA_RAW, clase_destino, nuevo_nombre)

            try:
                shutil.copy2(src, dst)
            except Exception:
                pass

    print("\n📊 Imágenes organizadas por clase:")
    total = 0
    for clase in CLASES:
        n = len(os.listdir(os.path.join(DATA_RAW, clase)))
        total += n
        estado = "✅" if n >= 50 else "⚠️  (necesitas más imágenes propias)"
        print(f"   {estado} {clase:<25} → {n:>4} imágenes")
    print(f"\n   Total: {total} imágenes")

    # Limpiar tmp
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
        print("\n🧹 Carpeta temporal limpiada.")

def verificar_kaggle():
    """Verifica que kaggle CLI esté disponible y configurado."""
    resultado = os.system('kaggle --version > /dev/null 2>&1')
    if resultado != 0:
        print("❌ kaggle CLI no encontrado.")
        print("   Instala con: pip install kaggle")
        print("   Luego configura tu kaggle.json desde: https://www.kaggle.com/settings")
        return False

    kaggle_json = os.path.expanduser('~/.kaggle/kaggle.json')
    if not os.path.exists(kaggle_json):
        print("❌ kaggle.json no encontrado en ~/.kaggle/")
        print("   Descárgalo desde: Kaggle → Account → API → Create New Token")
        return False

    print("✅ Kaggle CLI configurado correctamente.")
    return True

def instrucciones_manuales():
    """Muestra instrucciones para descarga manual si Kaggle no está disponible."""
    print("""
╔══════════════════════════════════════════════════════════════════╗
║           DESCARGA MANUAL — Alternativa sin Kaggle CLI           ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  1. TrashNet:                                                    ║
║     → https://github.com/garythung/trashnet                     ║
║     Descarga dataset-resized-original.zip                        ║
║                                                                  ║
║  2. Garbage Classification (Kaggle):                             ║
║     → https://www.kaggle.com/datasets/asdasdasasdas/            ║
║       garbage-classification                                     ║
║                                                                  ║
║  3. Descomprime y coloca las imágenes en:                        ║
║     data/raw/lata/         (imágenes de latas)                   ║
║     data/raw/botella_pet/  (botellas plásticas)                  ║
║     data/raw/papel/        (papel)                               ║
║     ... (una carpeta por clase)                                  ║
║                                                                  ║
║  4. Imágenes propias del equipo:                                 ║
║     Fotografíen residuos reales y agréguenlos a cada carpeta.    ║
║     Meta: mínimo 50 imágenes por clase.                          ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
""")

if __name__ == '__main__':
    print("=" * 60)
    print("  DESCARGA DE DATASETS — Reciclaje Inteligente · Grupo #4")
    print("=" * 60)

    crear_estructura()

    if verificar_kaggle():
        ok1 = descargar_trashnet()
        ok2 = descargar_garbage_classification()
        if ok1 or ok2:
            organizar_imagenes()
        else:
            instrucciones_manuales()
    else:
        instrucciones_manuales()

    print("\n✅ Proceso finalizado.")
    print("   Siguiente paso: ejecuta el notebook notebooks/01_EDA.ipynb")
