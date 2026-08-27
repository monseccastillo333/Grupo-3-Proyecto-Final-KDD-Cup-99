# Importar librerías

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin



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



class DeduplicateGlobal(BaseEstimator, TransformerMixin):
    """Elimina duplicados exactos en TODO el dataset (no distingue por clase).
    ...
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        before = len(X)
        print("\n[DEBUG] Columnas al momento de deduplicar:", X.columns.tolist())
        print("[DEBUG] Tipos de datos:\n", X.dtypes)
        print("[DEBUG] Duplicados detectados por pandas:", X.duplicated().sum())
        X_clean = X.drop_duplicates().reset_index(drop=True)
        print(f"Duplicados eliminados: {before - len(X_clean)} de {before} filas")
        return X_clean


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


# Sistema de reporte global en JSON
report = {"steps": []}

def log_step(name: str, detail: dict):
    print(f"\n[OK] {name}")
    for k, v in detail.items():
        print(f"     - {k}: {v}")
    report["steps"].append({"step": name, **detail})