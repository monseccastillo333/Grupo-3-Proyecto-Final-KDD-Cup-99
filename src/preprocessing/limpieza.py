#importar bibliotecas


import json
import sys
from pathlib import Path
 
import numpy as np
import pandas as pd


# carga de datos

def find_project_root(marker: str = "data/raw") -> Path:
    """Sube desde el directorio actual del script hasta encontrar 'data/raw/'."""
    current = Path(__file__).resolve().parent
    for candidate in [current, *current.parents]:
        if (candidate / marker).exists():
            return candidate
    raise FileNotFoundError(
        f"No se encontró '{marker}/' subiendo desde {current}. "
        "Ejecuta el script dentro del proyecto o ajusta la estructura."
    )
 
 
PROJECT_ROOT = find_project_root()
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "kddcup_10_percent.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

 
# Mapeo estándar de 'label' -> 'attack_category', consistente con la
# taxonomía usada en Data Quality (Sección F): dos, probe, r2l, u2r, normal.
ATTACK_CATEGORY_MAP = {
    "normal.": "normal",
    # DoS
    "back.": "dos", "land.": "dos", "neptune.": "dos", "pod.": "dos",
    "smurf.": "dos", "teardrop.": "dos", "apache2.": "dos", "udpstorm.": "dos",
    "processtable.": "dos", "mailbomb.": "dos",
    # Probe
    "ipsweep.": "probe", "nmap.": "probe", "portsweep.": "probe",
    "satan.": "probe", "mscan.": "probe", "saint.": "probe",
    # R2L
    "ftp_write.": "r2l", "guess_passwd.": "r2l", "imap.": "r2l",
    "multihop.": "r2l", "phf.": "r2l", "spy.": "r2l", "warezclient.": "r2l",
    "warezmaster.": "r2l", "sendmail.": "r2l", "named.": "r2l",
    "snmpgetattack.": "r2l", "snmpguess.": "r2l", "xlock.": "r2l",
    "xsnoop.": "r2l", "worm.": "r2l",
    # U2R
    "buffer_overflow.": "u2r", "loadmodule.": "u2r", "perl.": "u2r",
    "rootkit.": "u2r", "httptunnel.": "u2r", "ps.": "u2r",
    "sqlattack.": "u2r", "xterm.": "u2r",
}

def load_raw_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        sys.exit(f"[ERROR] No se encontró el CSV crudo en: {path}")
    df = pd.read_csv(path)

# Igual que en data_quality_report.py: si no viene 'attack_category',
    # se deriva en memoria a partir de 'label' sin tocar el archivo fuente.
    if "attack_category" not in df.columns:
        if "label" not in df.columns:
            sys.exit("[ERROR] El CSV no tiene ni 'attack_category' ni 'label'.")
        df["attack_category"] = (
            df["label"].map(ATTACK_CATEGORY_MAP).fillna("desconocido")
        )
    if "is_anomaly" not in df.columns:
        df["is_anomaly"] = (df["attack_category"] != "normal").astype(int)
 
    return df

# ---------------------------------------------------------------------------
# Utilidad de logging del pipeline (para el reporte final)
# ---------------------------------------------------------------------------
 
report = {"steps": []}
 
 
def log_step(name: str, detail: dict):
    print(f"\n[OK] {name}")
    for k, v in detail.items():
        print(f"     - {k}: {v}")
    report["steps"].append({"step": name, **detail})

# ---------------------------------------------------------------------------
# 1. Duplicados — NO eliminar entre clases de ataque
# ---------------------------------------------------------------------------
 
def clean_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Elimina duplicados exactos SOLO dentro de la clase 'normal'.
    Los duplicados en clases de ataque (dos, probe, r2l, u2r) son parte
    real del patrón de tráfico (ráfagas repetidas) y se conservan intactos.
    """
    before = len(df)
 
    normal_mask = df["attack_category"] == "normal"
    normal_rows = df[normal_mask]
    attack_rows = df[~normal_mask]
 
    normal_dedup = normal_rows.drop_duplicates()
 
    df_clean = pd.concat([normal_dedup, attack_rows], axis=0).sort_index()
 
    removed = before - len(df_clean)
    log_step(
        "1. Duplicados (solo dentro de 'normal')",
        {
            "filas_antes": before,
            "filas_normal_duplicadas_eliminadas": removed,
            "filas_despues": len(df_clean),
            "duplicados_en_ataques": "conservados (patrón real de tráfico)",
        },
    )
    return df_clean.reset_index(drop=True)

# ---------------------------------------------------------------------------
# 2. Imbalance de clases — NO se resamplea aquí
# ---------------------------------------------------------------------------
# El README es explícito: el balanceo (class_weight="balanced" o SMOTE
# cuidadoso en u2r) es una decisión de MODELADO (secciones J/K), no de
# limpieza de datos. Este script NO genera filas sintéticas ni sub-muestrea;
# solo deja registrado el ratio para que el paso de modelado lo use.
 
def log_class_imbalance(df: pd.DataFrame):
    counts = df["attack_category"].value_counts()
    ratio = counts.max() / counts.min()
    log_step(
        "2. Imbalance de clases (solo diagnóstico, sin resampling aquí)",
        {
            "conteo_por_clase": counts.to_dict(),
            "ratio_max_min": round(float(ratio), 1),
            "decisión": "class_weight='balanced' se aplica en modelado (Sección J/K), "
                        "no en este script de limpieza",
        },
    )

# ---------------------------------------------------------------------------
# 3. Columnas correlacionadas — eliminar redundancia, no señal
# ---------------------------------------------------------------------------
 
COLUMNS_TO_DROP_CORRELATION = [
    # Grupo serror_rate (r ≈ 0.9+, correlación con target ≈ 0.227 las tres)
    # -> se conserva dst_host_serror_rate (ventana de 100 conexiones, más estable)
    "serror_rate",
    "srv_serror_rate",
    # Grupo rerror_rate (r = 0.995 entre rerror_rate y srv_rerror_rate)
    # -> mismo criterio: se conserva dst_host_rerror_rate
    "rerror_rate",
    "srv_rerror_rate",
    # num_compromised / num_root (r = 0.994) -> se conserva num_compromised,
    # ya que aparece explícitamente como feature casi-binaria útil en el
    # hallazgo de skewness extrema (Sección F/H)
    "num_root",
]
 
 
def drop_correlated_columns(df: pd.DataFrame) -> pd.DataFrame:
    present = [c for c in COLUMNS_TO_DROP_CORRELATION if c in df.columns]
    missing = [c for c in COLUMNS_TO_DROP_CORRELATION if c not in df.columns]
    df_clean = df.drop(columns=present)
    log_step(
        "3. Columnas correlacionadas eliminadas (r > 0.9)",
        {
            "eliminadas": present,
            "no_encontradas_en_df": missing,
            "conservadas_de_su_grupo": [
                "dst_host_serror_rate", "dst_host_rerror_rate", "num_compromised",
            ],
        },
    )
    return df_clean


# ---------------------------------------------------------------------------
# 4. service (66 categorías) — target/frequency encoding, no one-hot
# ---------------------------------------------------------------------------
 
def encode_service_target_frequency(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Reemplaza 'service' por su tasa histórica de anomalía (target encoding),
    calculada sobre este mismo dataset. El mapeo se guarda para poder
    aplicarlo de forma idéntica en el API de inferencia.
 
    NOTA DE LIMITACIÓN (obligatoria según README): varios servicios poco
    frecuentes muestran 100% de tasa de anomalía por cómo se construyó el
    entorno simulado DARPA/Lincoln Labs (McHugh, 2000) -- el tráfico normal
    simulado solo usa servicios "típicos". No es necesariamente una relación
    causal real de ciberseguridad; un modelo que dependa demasiado de esta
    señal puede generalizar mal fuera de este dataset.
    """
    service_anomaly_rate = df.groupby("service")["is_anomaly"].mean().to_dict()
 
    df = df.copy()
    df["service_anomaly_rate"] = df["service"].map(service_anomaly_rate)
 
    services_100pct = [s for s, r in service_anomaly_rate.items() if r == 1.0]
 
    log_step(
        "4. service -> target/frequency encoding",
        {
            "columna_nueva": "service_anomaly_rate",
            "categorias_originales": df["service"].nunique(),
            "servicios_con_100pct_anomalia": services_100pct,
            "limitacion_documentada": (
                "Tasa de 100% en servicios poco comunes refleja el diseño del "
                "entorno simulado (McHugh, 2000), no causalidad real. "
                "Documentar en el informe técnico; no depender excesivamente "
                "de esta señal sin mencionar la limitación."
            ),
        },
    )
    # 'service' original se conserva por trazabilidad; se excluye de X más
    # adelante (paso 6) junto con las demás columnas categóricas crudas si
    # el modelo lo requiere. Aquí solo se agrega la versión codificada.
    return df, service_anomaly_rate

# ---------------------------------------------------------------------------
# 5. src_bytes / dst_bytes (skewness extrema) — transformar, no eliminar
# ---------------------------------------------------------------------------
 
def add_log_transforms(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea columnas *_log con log1p(), SIN eliminar ni sobrescribir las
    originales: los modelos de árboles (Random Forest, XGBoost) no las
    necesitan (son invariantes a la transformación monótona), pero los
    modelos lineales/basados en distancia (regresión logística, KNN, SVM)
    sí deben usar la versión log.
    """
    df = df.copy()
    for col in ["src_bytes", "dst_bytes"]:
        df[f"{col}_log"] = np.log1p(df[col])
 
    log_step(
        "5. Transformación log1p (skewness extrema)",
        {
            "columnas_originales_conservadas": ["src_bytes", "dst_bytes"],
            "columnas_nuevas": ["src_bytes_log", "dst_bytes_log"],
            "uso": "usar *_log en modelos lineales/distancia; "
                   "usar originales en modelos de árboles",
        },
    )
    return df


 
# ---------------------------------------------------------------------------
# 6. Columnas a EXCLUIR del set de features (leakage directo del target)
# ---------------------------------------------------------------------------
 
LEAKAGE_TARGET_COLUMNS = ["label", "attack_category", "is_anomaly"]
 
 
def get_feature_columns(df: pd.DataFrame) -> list:
    """
    No elimina estas columnas del DataFrame (se necesitan como target y
    para trazabilidad), solo las excluye de la lista de features (X).
    """
    feature_cols = [c for c in df.columns if c not in LEAKAGE_TARGET_COLUMNS]
    log_step(
        "6. Columnas excluidas de X (leakage directo del target)",
        {
            "excluidas_de_features": LEAKAGE_TARGET_COLUMNS,
            "uso": "label/attack_category/is_anomaly solo como fuente de y, "
                   "nunca como columnas de entrada al modelo",
            "n_features_resultantes": len(feature_cols),
        },
    )
    return feature_cols

 
# ---------------------------------------------------------------------------
# 7. Columnas de ventana de tiempo — no se limpian, se documentan
# ---------------------------------------------------------------------------
 
TIME_WINDOW_2S = [
    "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate",
]
TIME_WINDOW_100CONN = [c for c in [
    "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
]]
 
 
def log_time_window_columns(df: pd.DataFrame):
    present_2s = [c for c in TIME_WINDOW_2S if c in df.columns]
    present_100 = [c for c in TIME_WINDOW_100CONN if c in df.columns]
    log_step(
        "7. Columnas de ventana temporal (no se limpian, prioridad de diseño)",
        {
            "ventana_2_segundos_presentes": present_2s,
            "ventana_100_conexiones_presentes": present_100,
            "nota": "count (r=0.75) y srv_count (r=0.57) son las features más "
                    "discriminativas del EDA; deben recalcularse de forma "
                    "IDÉNTICA en el API de inferencia (Sección M) — prioridad "
                    "de diseño por encima de otras transformaciones.",
        },
    )
 
# ---------------------------------------------------------------------------
# 8. Sin acción requerida (verificación de que sigue así tras la limpieza)
# ---------------------------------------------------------------------------
 
def verify_no_action_needed(df: pd.DataFrame):
    missing = int(df.isna().sum().sum())
    log_step(
        "8. Verificaciones que salieron limpias en calidad de datos",
        {
            "nan_tras_limpieza": missing,
            "nota": "NaN, faltantes codificados, tipos de datos, categorías "
                    "fuera de dominio y datos imposibles ya se verificaron "
                    "limpios en Data Quality (Sección F); no requieren "
                    "transformación adicional aquí.",
        },
    )

# ---------------------------------------------------------------------------
# 9. Columnas constantes — eliminar
# ---------------------------------------------------------------------------
 
CONSTANT_COLUMNS = ["num_outbound_cmds", "is_host_login"]
 
 
def drop_constant_columns(df: pd.DataFrame) -> pd.DataFrame:
    present = [c for c in CONSTANT_COLUMNS if c in df.columns]
    verified_constant = [c for c in present if df[c].nunique() == 1]
    not_constant_anymore = [c for c in present if df[c].nunique() != 1]
 
    df_clean = df.drop(columns=verified_constant)
 
    log_step(
        "9. Columnas constantes eliminadas",
        {
            "eliminadas": verified_constant,
            "advertencia_si_ya_no_son_constantes": not_constant_anymore,
        },
    )
    return df_clean

# ---------------------------------------------------------------------------
# 10. su_attempted — valor inesperado (3 valores en vez de 2)
# ---------------------------------------------------------------------------
 
def handle_su_attempted(df: pd.DataFrame) -> pd.DataFrame:
    """
    Documenta y decide el tratamiento del tercer valor de 'su_attempted'
    (artefacto conocido de KDD Cup 99, no error de ingesta).
 
    Decisión tomada (documentada explícitamente, como exige el README):
    agrupar el valor inesperado junto con 1, ya que ambos indican que sí
    hubo un intento de "su root". Se conserva la columna original para
    trazabilidad y se agrega 'su_attempted_grouped' con la decisión aplicada.
    """
    if "su_attempted" not in df.columns:
        return df
 
    df = df.copy()
    valores_originales = sorted(df["su_attempted"].unique().tolist())
    valor_inesperado = [v for v in valores_originales if v not in (0, 1)]
 
    df["su_attempted_grouped"] = df["su_attempted"].apply(
        lambda v: 1 if v in (1, *valor_inesperado) else 0
    )
 
    log_step(
        "10. su_attempted — valor inesperado agrupado",
        {
            "valores_originales_encontrados": valores_originales,
            "valor(es)_inesperado(s)": valor_inesperado,
            "decisión": "agrupado junto con 1 en 'su_attempted_grouped' "
                        "(ambos indican intento de 'su root'); "
                        "'su_attempted' original se conserva sin modificar",
        },
    )
    return df 


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------
 
def main():
    print(f"Raíz del proyecto detectada: {PROJECT_ROOT}")
    print(f"Leyendo dataset crudo: {RAW_PATH}")
 
    df = load_raw_data(RAW_PATH)
    print(f"Filas cargadas: {len(df):,} | Columnas: {df.shape[1]}")
 
    df = clean_duplicates(df)                       # Recomendación 1
    log_class_imbalance(df)                          # Recomendación 2
    df = drop_correlated_columns(df)                 # Recomendación 3
    df, service_map = encode_service_target_frequency(df)  # Recomendación 4
    df = add_log_transforms(df)                       # Recomendación 5
    feature_cols = get_feature_columns(df)             # Recomendación 6
    log_time_window_columns(df)                         # Recomendación 7
    verify_no_action_needed(df)                          # Recomendación 8
    df = drop_constant_columns(df)                        # Recomendación 9
    df = handle_su_attempted(df)                           # Recomendación 10
 
    # Recalcular feature_cols tras los drops posteriores (paso 9 elimina
    # columnas del df; se excluyen las mismas de LEAKAGE_TARGET_COLUMNS)
    feature_cols_final = [c for c in df.columns if c not in LEAKAGE_TARGET_COLUMNS]
 
    # --- Guardar salidas -----------------------------------------------
    out_csv = PROCESSED_DIR / "kddcup_10_percent_clean.csv"
    df.to_csv(out_csv, index=False)
 
    report["resumen"] = {
        "filas_finales": len(df),
        "columnas_finales": df.shape[1],
        "n_features_finales": len(feature_cols_final),
        "csv_salida": str(out_csv.relative_to(PROJECT_ROOT)),
    }
    report["service_anomaly_rate_map"] = service_map
    report["feature_columns_final"] = feature_cols_final
 
    out_json = REPORTS_DIR / "feature_engineering_report.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
 
    print(f"\n{'='*70}")
    print(f"Dataset limpio guardado en: {out_csv}")
    print(f"Reporte de feature engineering guardado en: {out_json}")
    print(f"Filas finales: {len(df):,} | Columnas finales: {df.shape[1]} | "
          f"Features para X: {len(feature_cols_final)}")
    print(f"{'='*70}")
 
 
if __name__ == "__main__":
    main()
 