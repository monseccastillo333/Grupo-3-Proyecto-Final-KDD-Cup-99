"""

Integra la limpieza, transformación y extracción de características en un único
Pipeline de Scikit-Learn, conservando las utilidades de reporte JSON y mapeo
de categorías del proyecto KDD Cup 99.

Garantiza la compatibilidad entre Entrenamiento e Inferencia:
  - En Entrenamiento: Se llama pipeline.fit_transform(df)
  - En Inferencia/API: Se carga el pipeline serializado y se llama pipeline.transform(X_new)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

# ---------------------------------------------------------------------------
# Definición de Rutas Dinámicas y Carpetas del Proyecto
# ---------------------------------------------------------------------------

def find_project_root(marker: str = "data") -> Path:
    """Sube desde el directorio actual del script hasta encontrar la carpeta 'data'."""
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
MODELS_DIR = PROJECT_ROOT / "models"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Mapeo de Taxonomía y Carga de Datos Crudos
# ---------------------------------------------------------------------------

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

    if "attack_category" not in df.columns:
        if "label" not in df.columns:
            sys.exit("[ERROR] El CSV no tiene ni 'attack_category' ni 'label'.")
        df["attack_category"] = (
            df["label"].map(ATTACK_CATEGORY_MAP).fillna("desconocido")
        )
    if "is_anomaly" not in df.columns:
        df["is_anomaly"] = (df["attack_category"] != "normal").astype(int)

    return df

# Sistema de reporte global en JSON
report = {"steps": []}

def log_step(name: str, detail: dict):
    print(f"\n[OK] {name}")
    for k, v in detail.items():
        print(f"     - {k}: {v}")
    report["steps"].append({"step": name, **detail})

# ---------------------------------------------------------------------------
# Clases Transformer para Scikit-Learn Pipeline
# ---------------------------------------------------------------------------

CONSTANT_COLUMNS = ["num_outbound_cmds", "is_host_login"]

CORRELATED_COLUMNS_TO_DROP = [
    "serror_rate", "srv_serror_rate",      # Se conserva dst_host_serror_rate
    "rerror_rate", "srv_rerror_rate",      # Se conserva dst_host_rerror_rate
    "num_root",                             # Se conserva num_compromised
]

LOG_TRANSFORM_COLUMNS = ["src_bytes", "dst_bytes"]
LEAKAGE_TARGET_COLUMNS = ["label", "attack_category", "is_anomaly"]


class DeduplicateNormalTraffic(BaseEstimator, TransformerMixin):
    """Elimina duplicados exactos ÚNICAMENTE dentro de la clase 'normal'."""

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        before = len(X)
        
        target_col = None
        for col in ["attack_category", "target", "is_anomaly", "label"]:
            if col in X.columns:
                target_col = col
                break
        
        if target_col is None:
            return X.drop_duplicates()

        if target_col in ["target", "is_anomaly"]:
            is_normal = X[target_col] == 0
        else:
            is_normal = X[target_col].astype(str).str.contains("normal", case=False)

        dups_normal = X[is_normal].duplicated(keep="first")
        indices_to_drop = dups_normal[dups_normal].index
        df_clean = X.drop(index=indices_to_drop).reset_index(drop=True)

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
        return df_clean


class DropCorrelatedColumns(BaseEstimator, TransformerMixin):
    """Elimina columnas altamente correlacionadas (r > 0.9)."""

    def __init__(self, columns: list[str] | None = None):
        self.columns = columns or CORRELATED_COLUMNS_TO_DROP

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        present = [c for c in self.columns if c in X.columns]
        missing = [c for c in self.columns if c not in X.columns]
        df_clean = X.drop(columns=present, errors="ignore")
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


class LogTransformer(BaseEstimator, TransformerMixin):
    """Aplica log1p con .clip(lower=0) para corregir posibles valores negativos."""

    def __init__(self, columns: list[str] | None = None):
        self.columns = columns or LOG_TRANSFORM_COLUMNS

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in self.columns:
            if col in X.columns:
                X[f"{col}_log"] = np.log1p(X[col].clip(lower=0))

        log_step(
            "5. Transformación log1p (skewness extrema)",
            {
                "columnas_originales_conservadas": [c for c in self.columns if c in X.columns],
                "columnas_nuevas": [f"{c}_log" for c in self.columns if c in X.columns],
                "uso": "usar *_log en modelos lineales/distancia; usar originales en árboles",
            },
        )
        return X


class DropConstantColumns(BaseEstimator, TransformerMixin):
    """Elimina columnas sin varianza (constantes)."""

    def __init__(self, columns: list[str] | None = None):
        self.columns = columns or CONSTANT_COLUMNS

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        present = [c for c in self.columns if c in X.columns]
        verified_constant = [c for c in present if X[c].nunique() <= 1]
        not_constant_anymore = [c for c in present if X[c].nunique() > 1]

        df_clean = X.drop(columns=verified_constant, errors="ignore")
        log_step(
            "9. Columnas constantes eliminadas",
            {
                "eliminadas": verified_constant,
                "advertencia_si_ya_no_son_constantes": not_constant_anymore,
            },
        )
        return df_clean


class SuAttemptedGrouper(BaseEstimator, TransformerMixin):
    """Agrupa valores anómalos de su_attempted usando np.where vectorizado."""

    def __init__(self, column: str = "su_attempted"):
        self.column = column

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.column not in X.columns:
            return X
        X = X.copy()
        valores_originales = sorted(X[self.column].unique().tolist())
        valor_inesperado = [v for v in valores_originales if v not in (0, 1)]

        X["su_attempted_grouped"] = np.where(X[self.column] != 0, 1, 0)

        log_step(
            "10. su_attempted — valor inesperado agrupado",
            {
                "valores_originales_encontrados": valores_originales,
                "valor(es)_inesperado(s)": valor_inesperado,
                "decisión": "agrupado junto con 1 en 'su_attempted_grouped' (ambos indican intento de 'su root')",
            },
        )
        return X

# ---------------------------------------------------------------------------
# Funciones Complementarias de Diagnóstico (Sección F/G)
# ---------------------------------------------------------------------------

def log_class_imbalance(df: pd.DataFrame):
    if "attack_category" in df.columns:
        counts = df["attack_category"].value_counts()
        ratio = counts.max() / counts.min()
        log_step(
            "2. Imbalance de clases (solo diagnóstico, sin resampling aquí)",
            {
                "conteo_por_clase": counts.to_dict(),
                "ratio_max_min": round(float(ratio), 1),
                "decisión": "class_weight='balanced' se aplica en modelado (Sección J/K), no en limpieza",
            },
        )

def log_time_window_columns(df: pd.DataFrame):
    time_2s = ["count", "srv_count", "serror_rate", "srv_serror_rate", "same_srv_rate"]
    present_2s = [c for c in time_2s if c in df.columns]
    log_step(
        "7. Columnas de ventana temporal (prioridad de diseño)",
        {
            "ventana_2_segundos_presentes": present_2s,
            "nota": "Features discriminativas del EDA; deben mantenerse consistentes en inferencia",
        },
    )

def verify_no_action_needed(df: pd.DataFrame):
    missing = int(df.isna().sum().sum())
    log_step(
        "8. Verificaciones de calidad de datos",
        {
            "nan_tras_limpieza": missing,
            "nota": "Datos sin valores faltantes ni inconsistencias estructurales",
        },
    )

# ---------------------------------------------------------------------------
# Construcción del Pipeline Principal
# ---------------------------------------------------------------------------

def build_pipeline() -> Pipeline:
    return Pipeline(steps=[
        ("deduplicate_normal", DeduplicateNormalTraffic()),
        ("drop_correlated", DropCorrelatedColumns()),
        ("log_transform", LogTransformer()),
        ("drop_constant", DropConstantColumns()),
        ("su_attempted", SuAttemptedGrouper()),
    ])

# ---------------------------------------------------------------------------
# Ejecución del Script Main
# ---------------------------------------------------------------------------

def main():
    print(f"Raíz del proyecto detectada: {PROJECT_ROOT}")
    print(f"Leyendo dataset crudo desde: {RAW_PATH}")

    df_raw = load_raw_data(RAW_PATH)
    print(f"Filas cargadas: {len(df_raw):,} | Columnas: {df_raw.shape[1]}")

    # 1. Construir e invocar Pipeline
    pipeline = build_pipeline()
    df_processed = pipeline.fit_transform(df_raw)

    # 2. Registros diagnósticos adicionales en el reporte
    log_class_imbalance(df_processed)
    log_time_window_columns(df_processed)
    verify_no_action_needed(df_processed)

    # 3. Determinar lista de características finales para X
    feature_cols_final = [c for c in df_processed.columns if c not in LEAKAGE_TARGET_COLUMNS]

    log_step(
        "6. Columnas excluidas de X (leakage directo del target)",
        {
            "excluidas_de_features": LEAKAGE_TARGET_COLUMNS,
            "n_features_resultantes": len(feature_cols_final),
        },
    )

    # 4. Guardar archivo CSV procesado
    out_csv = PROCESSED_DIR / "kddcup_10_percent_clean.csv"
    df_processed.to_csv(out_csv, index=False)

    # 5. Guardar Pipeline Serializado (.joblib)
    pipeline_path = MODELS_DIR / "feature_pipeline.joblib"
    joblib.dump(pipeline, pipeline_path)

    # 6. Exportar Reporte JSON de Feature Engineering
    report["resumen"] = {
        "filas_finales": len(df_processed),
        "columnas_finales": df_processed.shape[1],
        "n_features_finales": len(feature_cols_final),
        "csv_salida": str(out_csv.relative_to(PROJECT_ROOT)),
        "pipeline_joblib": str(pipeline_path.relative_to(PROJECT_ROOT)),
    }
    report["feature_columns_final"] = feature_cols_final

    out_json = REPORTS_DIR / "feature_engineering_report.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*70}")
    print(f"Dataset limpio guardado en: {out_csv}")
    print(f"Pipeline serializado guardado en: {pipeline_path}")
    print(f"Reporte de feature engineering guardado en: {out_json}")
    print(f"Filas finales: {len(df_processed):,} | Columnas finales: {df_processed.shape[1]} | "
          f"Features para X: {len(feature_cols_final)}")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()