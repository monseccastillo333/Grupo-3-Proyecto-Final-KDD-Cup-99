import os
import pandas as pd
from sklearn.datasets import fetch_kddcup99

DIR_DESTINO = "data/raw"
ARCHIVO_CSV = os.path.join(DIR_DESTINO, "kddcup_10_percent.csv")

def ejecutar_ingesta():
    # Crear carpeta data/raw si no existe
    os.makedirs(DIR_DESTINO, exist_ok=True)
    
    print("Descargando dataset KDD Cup 1999 mediante Scikit-Learn...")
    
    # Se elimina 'decode_toks' para evitar incompatibilidad de versión
    dataset = fetch_kddcup99(subset=None, percent10=True, as_frame=True)
    
    # Obtener el DataFrame con características y la etiqueta target
    df = dataset.frame
    
    # Convertir columnas tipo bytes a cadenas (strings) si es necesario
    for col in df.columns:
        if df[col].dtype == object or str(df[col].dtype).startswith('bytes'):
            df[col] = df[col].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
    
    # Renombrar la columna objetivo a 'label'
    if 'target' in df.columns:
        df = df.rename(columns={'target': 'label'})
    elif 'labels' in df.columns:
        df = df.rename(columns={'labels': 'label'})
    
    # Guardar CSV localmente
    df.to_csv(ARCHIVO_CSV, index=False)
    
    print(f"\n¡Ingesta completada con éxito!")
    print(f"Archivo guardado en: {ARCHIVO_CSV}")
    print(f"Dimensiones: {df.shape[0]} filas | {df.shape[1]} columnas")

if __name__ == "__main__":
    ejecutar_ingesta()