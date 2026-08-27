"""
train.py — Entrenamiento y evaluación de modelos de detección de anomalías
Grupo 3 — KDD Cup 1999 — Sección J/K del proyecto MLOps End-to-End

Este script asume que ya se ejecutó `Limpieza_transformacion.py` (que genera
data/processed/kddcup_10_percent_clean.csv y models/feature_pipeline.joblib).

Ejecución:
    py -m src.training.train

Genera:
    - models/baseline_logistic_regression.joblib
    - models/random_forest.joblib
    - models/xgboost.joblib   (modelo seleccionado, ver informe técnico)
    - reports/model_baseline_report.json
    - reports/model_random_forest_report.json
    - reports/model_xgboost_report.json
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


# ---------------------------------------------------------------------------
# Rutas del proyecto
# ---------------------------------------------------------------------------

def find_project_root(marker: str = "README.md") -> Path:
    current = Path(__file__).resolve().parent
    for candidate in [current, *current.parents]:
        if (candidate / marker).exists():
            return candidate
    raise FileNotFoundError(f"No se encontró '{marker}' subiendo desde {current}.")


PROJECT_ROOT = find_project_root()
CLEAN_CSV_PATH = PROJECT_ROOT / "data" / "processed" / "kddcup_10_percent_clean.csv"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

LEAKAGE_COLUMNS = ["label", "attack_category", "is_anomaly"]
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Codificador de tasa de anomalía por servicio (aprende solo de train)
# ---------------------------------------------------------------------------

class ServiceTargetEncoder:
    """Aprende la tasa de anomalía por servicio SOLO con datos de entrenamiento,
    y aplica ese mismo mapeo a cualquier dato nuevo (train o test)."""

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "ServiceTargetEncoder":
        temp = X_train[["service"]].copy()
        temp["is_anomaly"] = y_train.values
        self.mapping_ = temp.groupby("service")["is_anomaly"].mean().to_dict()
        self.global_mean_ = y_train.mean()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X["service_anomaly_rate"] = X["service"].map(self.mapping_).fillna(self.global_mean_)
        return X


# ---------------------------------------------------------------------------
# Preparación de datos
# ---------------------------------------------------------------------------

def load_clean_dataset() -> pd.DataFrame:
    if not CLEAN_CSV_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró {CLEAN_CSV_PATH}. "
            "Ejecuta primero: py -m src.preprocessing.Limpieza_transformacion"
        )
    df = pd.read_csv(CLEAN_CSV_PATH)

    # Verificación de seguridad: el dataset debe llegar ya sin duplicados desde
    # el pipeline compartido. Si esto elimina filas, algo cambió en el pipeline
    # de origen y hay que investigar antes de continuar (ver informe técnico,
    # Sección 5, sobre la discrepancia conocida de deduplicación).
    filas_antes = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    if len(df) != filas_antes:
        print(
            f"⚠️  ADVERTENCIA: se eliminaron {filas_antes - len(df)} duplicados "
            "adicionales que no venían resueltos desde el pipeline de origen."
        )
    else:
        print(f"✅ Dataset verificado sin duplicados: {len(df):,} filas.")

    return df


def prepare_train_test(df: pd.DataFrame):
    X = df.drop(columns=LEAKAGE_COLUMNS)
    y = df["is_anomaly"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Train: {X_train.shape[0]:,} filas | Test: {X_test.shape[0]:,} filas")
    print(f"Proporción de anomalías en train: {y_train.mean():.4f}")
    print(f"Proporción de anomalías en test:  {y_test.mean():.4f}")

    # Codificador de servicio, ajustado solo con train
    encoder = ServiceTargetEncoder()
    encoder.fit(X_train, y_train)
    X_train = encoder.transform(X_train).drop(columns=["service"])
    X_test = encoder.transform(X_test).drop(columns=["service"])

    # One-hot de variables categóricas restantes
    X_train = pd.get_dummies(X_train, columns=["protocol_type", "flag"])
    X_test = pd.get_dummies(X_test, columns=["protocol_type", "flag"])

    # Alinear columnas por si alguna categoría rara solo aparece en un lado
    X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)

    categorias_test = df.loc[X_test.index, "attack_category"]

    return X_train, X_test, y_train, y_test, categorias_test, encoder


# ---------------------------------------------------------------------------
# Evaluación y reporte por categoría (mismo patrón para los 3 modelos)
# ---------------------------------------------------------------------------

def evaluate_model(model, X_test, y_test, categorias_test, nombre: str):
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    print(f"\n--- Evaluación: {nombre} ---")
    print(classification_report(y_test, y_pred, target_names=["normal", "anomalía"]))

    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"AUC: {auc:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    print("Matriz de confusión:")
    print(cm)

    resultado = pd.DataFrame({
        "categoria_real": categorias_test,
        "prediccion": y_pred,
        "real": y_test.values,
    })
    resumen = resultado.groupby("categoria_real").apply(
        lambda g: pd.Series({
            "total": len(g),
            "detectados": int((g["prediccion"] == g["real"]).sum()),
            "no_detectados": int((g["prediccion"] != g["real"]).sum()),
            "recall": float((g["prediccion"] == g["real"]).mean()),
        })
    )
    print(resumen)

    recall_por_categoria = {
        f"recall_{cat}": float(row["recall"]) for cat, row in resumen.iterrows()
    }
    return auc, recall_por_categoria


def save_model_and_report(model, model_path: Path, report_path: Path, run_name: str,
                           algorithm: str, parameters: dict, auc: float, recall_por_categoria: dict):
    joblib.dump(model, model_path)

    reporte = {
        "run_name": run_name,
        "algorithm": algorithm,
        "parameters": parameters,
        "metrics": {"auc": float(auc), **recall_por_categoria},
        "model_path": str(model_path.relative_to(PROJECT_ROOT)),
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(reporte, f, indent=2, ensure_ascii=False)

    print(f"{run_name} guardado en {model_path}")


# ---------------------------------------------------------------------------
# Los 3 modelos de producción (baseline, Random Forest, XGBoost base)
# ---------------------------------------------------------------------------

def train_baseline(X_train, X_test, y_train, y_test, categorias_test):
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE)
    model.fit(X_train_scaled, y_train)

    auc, recall_por_categoria = evaluate_model(
        model, X_test_scaled, y_test, categorias_test, "Baseline (Regresión Logística)"
    )

    save_model_and_report(
        model,
        MODELS_DIR / "baseline_logistic_regression.joblib",
        REPORTS_DIR / "model_baseline_report.json",
        run_name="baseline_logistic_regression",
        algorithm="LogisticRegression",
        parameters={"class_weight": "balanced", "max_iter": 1000, "random_state": RANDOM_STATE},
        auc=auc,
        recall_por_categoria=recall_por_categoria,
    )
    joblib.dump(scaler, MODELS_DIR / "scaler.joblib")
    return model


def train_random_forest(X_train, X_test, y_train, y_test, categorias_test):
    model = RandomForestClassifier(
        n_estimators=100, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
    )
    model.fit(X_train, y_train)

    auc, recall_por_categoria = evaluate_model(
        model, X_test, y_test, categorias_test, "Random Forest"
    )

    save_model_and_report(
        model,
        MODELS_DIR / "random_forest.joblib",
        REPORTS_DIR / "model_random_forest_report.json",
        run_name="random_forest",
        algorithm="RandomForestClassifier",
        parameters={"n_estimators": 100, "class_weight": "balanced", "random_state": RANDOM_STATE},
        auc=auc,
        recall_por_categoria=recall_por_categoria,
    )
    return model


def train_xgboost_base(X_train, X_test, y_train, y_test, categorias_test):
    """Modelo seleccionado para producción — ver criterio explícito en el
    informe técnico (Sección 4): mejor recall en la clase minoritaria u2r,
    superando a Random Forest y a los intentos de ajuste de hiperparámetros."""
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"scale_pos_weight calculado: {scale_pos_weight:.4f}")

    model = XGBClassifier(
        n_estimators=100,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    auc, recall_por_categoria = evaluate_model(
        model, X_test, y_test, categorias_test, "XGBoost base (MODELO SELECCIONADO)"
    )

    save_model_and_report(
        model,
        MODELS_DIR / "xgboost.joblib",
        REPORTS_DIR / "model_xgboost_report.json",
        run_name="xgboost",
        algorithm="XGBClassifier",
        parameters={
            "n_estimators": 100,
            "scale_pos_weight": float(scale_pos_weight),
            "random_state": RANDOM_STATE,
        },
        auc=auc,
        recall_por_categoria=recall_por_categoria,
    )
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Raíz del proyecto detectada: {PROJECT_ROOT}")

    df = load_clean_dataset()
    X_train, X_test, y_train, y_test, categorias_test, encoder = prepare_train_test(df)

    print(f"\nColumnas finales en X_train: {X_train.shape[1]}")

    joblib.dump(encoder, MODELS_DIR / "service_encoder.joblib")

    train_baseline(X_train, X_test, y_train, y_test, categorias_test)
    train_random_forest(X_train, X_test, y_train, y_test, categorias_test)
    train_xgboost_base(X_train, X_test, y_train, y_test, categorias_test)

    print("\n" + "=" * 70)
    print("Entrenamiento completo. Modelo seleccionado para producción: XGBoost base")
    print("Ver informe técnico (Sección 4) para el criterio explícito de selección.")
    print("=" * 70)


if __name__ == "__main__":
    main()