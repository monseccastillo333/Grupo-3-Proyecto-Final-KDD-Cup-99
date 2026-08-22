import os
import pandas as pd

ARCHIVO_RAW = os.path.join("data", "raw", "kddcup_10_percent.csv")

def validar_datos():
    print("Iniciando Data Quality Gates...")
    
    # 1. Verificar existencia del archivo
    if not os.path.exists(ARCHIVO_RAW):
        raise FileNotFoundError(f"ALERT: El archivo {ARCHIVO_RAW} no existe.")
    
    df = pd.read_csv(ARCHIVO_RAW)
    
    # Quality Gate 1: Dataset no vacío
    assert df.shape[0] > 0, "ALERT: El dataset en raw está completamente vacío."
    
    # Quality Gate 2: Mínimo de filas esperadas (~490k filas para el 10% de KDD Cup)
    assert df.shape[0] >= 100000, f"ALERT: Se esperaban al menos 100,000 filas y se encontraron {df.shape[0]}."
    
    # Quality Gate 3: Cantidad exacta de columnas esperadas (42 en KDD Cup 1999)
    assert df.shape[1] == 42, f"ALERT: Esquema no válido. Se esperaban 42 columnas y hay {df.shape[1]}."
    
    # Quality Gate 4: Presencia de la columna objetivo 'label'
    assert 'label' in df.columns, "ALERT: La columna objetivo 'label' no existe en el dataset."
    
    # Quality Gate 5: Umbral máximo de duplicados masivos (< 90%)
    tasa_duplicados = df.duplicated().mean()
    assert tasa_duplicados < 0.90, f"ALERT: Tasa de duplicados excesiva ({tasa_duplicados:.2%})."
    
    print("\n¡Data Quality Gates superados con éxito! [PASS]")
    print(f" Filas validadas: {df.shape[0]} | Columnas: {df.shape[1]}")
    print(f" Tasa de duplicados en raw: {tasa_duplicados:.2%}")

if __name__ == "__main__":
    validar_datos()