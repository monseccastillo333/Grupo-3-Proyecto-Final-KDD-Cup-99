"""
Grupo-3-Proyecto-Final-KDD-Cup-99
Feature Engineering / Transformación reutilizable (Sección I)

Pipeline de datos  -> este módulo (fit_transform en entrenamiento,
                           transform en inferencia)
Pipeline ML        -> consume la salida de este módulo (features,
                           entrenamiento, comparación, evaluación)

Todos los transformadores siguen la interfaz estándar de scikit-learn
(BaseEstimator, TransformerMixin), por lo que son compatibles con
GridSearchCV, joblib.dump/load, y con cualquier Pipeline de sklearn.

Uso típico:

    Entrenamiento / notebook
    -------------------------
    from feature_engineering import build_pipeline, save_pipeline

    df = pd.read_csv("data/raw/kddcup_10_percent.csv")
    y = derive_is_anomaly(df)              # ver helper más abajo

    pipe = build_pipeline()
    X_transformed = pipe.fit_transform(df, y)
    save_pipeline(pipe, "models/feature_pipeline.joblib")

    Producción / API de inferencia
    -------------------------------
    from feature_engineering import load_pipeline

    pipe = load_pipeline("models/feature_pipeline.joblib")
    X_new = pipe.transform(nueva_conexion_df)   # MISMA lógica, sin y
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

# ---------------------------------------------------------------------------
# Constantes compartidas (una sola definición, usada por todos los steps)
# ---------------------------------------------------------------------------

CONSTANT_COLUMNS = ["num_outbound_cmds", "is_host_login"]

CORRELATED_COLUMNS_TO_DROP = [
    "serror_rate", "srv_serror_rate",      # se conserva dst_host_serror_rate
    "rerror_rate", "srv_rerror_rate",      # se conserva dst_host_rerror_rate
    "num_root",                             # se conserva num_compromised
]

LOG_TRANSFORM_COLUMNS = ["src_bytes", "dst_bytes"]

LEAKAGE_TARGET_COLUMNS = ["label", "attack_category", "is_anomaly"]


# ---------------------------------------------------------------------------
# Transformer 1: columnas constantes
# ---------------------------------------------------------------------------

class DropConstantColumns(BaseEstimator, TransformerMixin):
    """
    Elimina columnas sin variación (num_outbound_cmds, is_host_login).
    fit() VERIFICA sobre el set de entrenamiento cuáles son realmente
    constantes -- no asume, por si una versión futura del dataset cambia.
    """

    def __init__(self, columns: list[str] | None = None):
        self.columns = columns or CONSTANT_COLUMNS

    def fit(self, X: pd.DataFrame, y=None):
        self.columns_to_drop_ = [
            c for c in self.columns if c in X.columns and X[c].nunique() == 1
        ]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.drop(columns=self.columns_to_drop_, errors="ignore")


# ---------------------------------------------------------------------------
# Transformer 2: columnas correlacionadas (r > 0.9)
# ---------------------------------------------------------------------------

class DropCorrelatedColumns(BaseEstimator, TransformerMixin):
    """
    Elimina las variantes redundantes de serror_rate/rerror_rate/num_root
    identificadas en Calidad de Datos (Sección F) y confirmadas con el
    EDA (Sección H). Es stateless (lista fija), pero se implementa como
    transformer para poder vivir dentro del mismo Pipeline.
    """

    def __init__(self, columns: list[str] | None = None):
        self.columns = columns or CORRELATED_COLUMNS_TO_DROP

    def fit(self, X: pd.DataFrame, y=None):
        self.fitted_ = True  # marca requerida por sklearn (check_is_fitted)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.drop(columns=self.columns, errors="ignore")


# ---------------------------------------------------------------------------
# Transformer 3: su_attempted -- agrupar valor inesperado
# ---------------------------------------------------------------------------

class SuAttemptedGrouper(BaseEstimator, TransformerMixin):
    """
    su_attempted trae 3 valores en vez de los 2 esperados (artefacto
    conocido de KDD Cup 99). Se agrega 'su_attempted_grouped': cualquier
    valor distinto de 0 se trata como 1 (hubo intento de 'su root').
    La columna original se conserva sin modificar, por trazabilidad.
    Vectorizado con np.where -- sin apply() por fila, importante para
    escalar a los ~4M registros del dataset completo.
    """

    def __init__(self, column: str = "su_attempted"):
        self.column = column

    def fit(self, X: pd.DataFrame, y=None):
        self.fitted_ = True  # marca requerida por sklearn (check_is_fitted)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if self.column in X.columns:
            X[f"{self.column}_grouped"] = np.where(X[self.column] != 0, 1, 0)
        return X


# ---------------------------------------------------------------------------
# Transformer 4: log1p sobre src_bytes / dst_bytes
# ---------------------------------------------------------------------------

class LogTransformer(BaseEstimator, TransformerMixin):
    """
    Agrega columnas *_log con log1p(), sin eliminar ni sobrescribir las
    originales -- los modelos de árboles usan las originales (invariantes
    a esta transformación monótona); los modelos lineales/basados en
    distancia deben usar las *_log. Stateless y vectorizado.
    """

    def __init__(self, columns: list[str] | None = None):
        self.columns = columns or LOG_TRANSFORM_COLUMNS

    def fit(self, X: pd.DataFrame, y=None):
        self.fitted_ = True  # marca requerida por sklearn (check_is_fitted)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in self.columns:
            if col in X.columns:
                X[f"{col}_log"] = np.log1p(X[col].clip(lower=0))
        return X


# ---------------------------------------------------------------------------
# Transformer 5: target/frequency encoding de 'service'
# ---------------------------------------------------------------------------

class ServiceTargetEncoder(BaseEstimator, TransformerMixin):
    """
    Reemplaza 'service' (66 categorías) por su tasa histórica de anomalía,
    en vez de one-hot. El mapeo se APRENDE en fit() (solo con datos de
    entrenamiento, para no filtrar información del set de validación/prod)
    y se aplica igual en transform(). Categorías no vistas en entrenamiento
    (frecuentes en producción con tráfico nuevo) reciben la tasa global
    promedio como fallback, en vez de fallar o devolver NaN.

    Limitación documentada (Sección H del EDA): varios servicios poco
    comunes muestran 100% de tasa de anomalía por cómo se construyó el
    entorno simulado DARPA/Lincoln Labs (McHugh, 2000) -- no implica
    causalidad real de ciberseguridad. Un modelo que dependa demasiado de
    esta señal puede generalizar mal fuera de este dataset.
    """

    def __init__(self, column: str = "service", new_column: str = "service_anomaly_rate"):
        self.column = column
        self.new_column = new_column

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray):
        if y is None:
            raise ValueError(
                "ServiceTargetEncoder requiere y (is_anomaly) en fit(). "
                "En producción NO se llama fit(), solo transform()."
            )
        y = pd.Series(np.asarray(y), index=X.index)
        self.mapping_ = X.groupby(self.column).apply(
            lambda idx: y.loc[idx.index].mean()
        ).to_dict()
        self.global_mean_ = float(y.mean())
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X[self.new_column] = (
            X[self.column].map(self.mapping_).fillna(self.global_mean_)
        )
        return X


# ---------------------------------------------------------------------------
# Transformer 6: optimización de dtypes (escalabilidad, ~4M filas)
# ---------------------------------------------------------------------------

class DtypeOptimizer(BaseEstimator, TransformerMixin):
    """
    Reduce el uso de memoria para que el pipeline escale a los ~4 millones
    de filas del dataset completo (no solo el 10%):
      - columnas float64 -> float32
      - columnas int64   -> el entero más pequeño que las representa
      - columnas object de baja cardinalidad -> category

    No cambia ningún valor, solo su representación en memoria. Stateless.
    """

    def __init__(self, category_max_unique: int = 100):
        self.category_max_unique = category_max_unique

    def fit(self, X: pd.DataFrame, y=None):
        self.fitted_ = True  # marca requerida por sklearn (check_is_fitted)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in X.columns:
            dtype = X[col].dtype
            if dtype == "float64":
                X[col] = pd.to_numeric(X[col], downcast="float")
            elif dtype == "int64":
                X[col] = pd.to_numeric(X[col], downcast="integer")
            elif dtype == "object" and X[col].nunique() <= self.category_max_unique:
                X[col] = X[col].astype("category")
        return X


# ---------------------------------------------------------------------------
# Factory del pipeline -- ÚNICA función que construye la transformación
# ---------------------------------------------------------------------------

def build_pipeline() -> Pipeline:
    """
    Punto único de construcción del pipeline de Feature Engineering.
    Tanto el notebook exploratorio como el código de producción DEBEN
    llamar a esta función en vez de reimplementar los pasos -- así se
    garantiza que nunca exista una lógica distinta en cada lado.
    """
    return Pipeline(steps=[
        ("drop_constant", DropConstantColumns()),
        ("drop_correlated", DropCorrelatedColumns()),
        ("su_attempted", SuAttemptedGrouper()),
        ("log_transform", LogTransformer()),
        ("service_encoder", ServiceTargetEncoder()),
        ("dtype_optimizer", DtypeOptimizer()),
    ])


# ---------------------------------------------------------------------------
# Utilidades de persistencia (compartidas entre notebook y producción)
# ---------------------------------------------------------------------------

def save_pipeline(pipeline: Pipeline, path: str) -> None:
    """Serializa el pipeline YA ENTRENADO (con mapeo de service aprendido)."""
    joblib.dump(pipeline, path)


def load_pipeline(path: str) -> Pipeline:
    """
    Carga el pipeline entrenado para usarlo en producción con
    pipeline.transform(nueva_conexion_df) -- nunca se vuelve a llamar
    fit() en inferencia, así se garantiza que el mapeo de 'service' y
    demás estadísticos aprendidos sean idénticos a los de entrenamiento.
    """
    return joblib.load(path)


# ---------------------------------------------------------------------------
# Helpers de datos (también compartidos, para no duplicar la derivación
# de la variable objetivo entre notebook y producción)
# ---------------------------------------------------------------------------

def derive_is_anomaly(df: pd.DataFrame) -> pd.Series:
    """
    Deriva is_anomaly (0/1) a partir de attack_category, sin usarla como
    feature de entrada (ver LEAKAGE_TARGET_COLUMNS).
    """
    if "is_anomaly" in df.columns:
        return df["is_anomaly"]
    if "attack_category" not in df.columns:
        raise ValueError("El DataFrame no tiene 'is_anomaly' ni 'attack_category'.")
    return (df["attack_category"] != "normal").astype(int)


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Lista de columnas de entrada (X) para el Pipeline ML, excluyendo
    label/attack_category/is_anomaly (leakage directo del target).
    Se usa igual en entrenamiento y en producción para construir X.
    """
    return [c for c in df.columns if c not in LEAKAGE_TARGET_COLUMNS]


def read_raw_data_chunked(path: str, chunksize: int = 500_000):
    """
    Generador para leer el dataset COMPLETO (~4M filas) por chunks cuando
    no entra cómodamente en memoria de una sola vez. Cada chunk puede
    pasarse por DtypeOptimizer antes de concatenar, para reducir el pico
    de memoria durante la carga:

        chunks = []
        for chunk in read_raw_data_chunked("data/raw/kddcup_full.csv"):
            chunks.append(DtypeOptimizer().fit_transform(chunk))
        df = pd.concat(chunks, ignore_index=True)

    Para el 10% del dataset (494,021 filas) esto no es necesario -- se usa
    solo cuando se trabaje con el dataset completo de ~4M instancias.
    """
    return pd.read_csv(path, chunksize=chunksize)


# ---------------------------------------------------------------------------
# Ejecución directa: fit sobre el dataset limpio y guarda el pipeline
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from pathlib import Path

    def find_project_root(marker: str = "data/processed") -> Path:
        current = Path(__file__).resolve().parent
        for candidate in [current, *current.parents]:
            if (candidate / marker).exists():
                return candidate
        raise FileNotFoundError(f"No se encontró '{marker}/' subiendo desde {current}")

    root = find_project_root()
    input_csv = root / "data" / "processed" / "kddcup_10_percent_clean.csv"
    models_dir = root / "models"
    models_dir.mkdir(exist_ok=True)

    print(f"Leyendo dataset limpio: {input_csv}")
    df = pd.read_csv(input_csv)
    y = derive_is_anomaly(df)

    pipeline = build_pipeline()
    X_transformed = pipeline.fit_transform(df, y)

    out_path = models_dir / "feature_pipeline.joblib"
    save_pipeline(pipeline, str(out_path))

    print(f"Pipeline entrenado y guardado en: {out_path}")
    print(f"Shape resultante: {X_transformed.shape}")
    print(f"Features para X (excluyendo leakage): {len(get_feature_columns(X_transformed))}")
