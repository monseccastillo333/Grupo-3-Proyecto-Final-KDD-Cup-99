"""Registra en MLflow los modelos ya entrenados, sin volver a entrenarlos.

Ejecución desde la raíz del proyecto:
    python -m src.tracking.log_existing_models
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from src.training.train import (
    CLEAN_CSV_PATH,
    LEAKAGE_COLUMNS,
    MODELS_DIR,
    PROJECT_ROOT,
    RANDOM_STATE,
    REPORTS_DIR,
    prepare_train_test,
)


MLFLOW_DB_PATH = PROJECT_ROOT / "mlflow.db"
MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_DB_PATH.as_posix()}"
EXPERIMENT_NAME = "KDD Cup 99 - deteccion de anomalias"
FEATURE_SET = "cleaned_numeric+service_target_encoding+protocol_flag_one_hot"

MODEL_DEFINITIONS = {
    "baseline_logistic_regression": {
        "path": MODELS_DIR / "baseline_logistic_regression.joblib",
        "report": REPORTS_DIR / "model_baseline_report.json",
        "algorithm": "LogisticRegression",
        "registered_name": "kdd99-baseline-logistic-regression",
        "extra_artifacts": [MODELS_DIR / "scaler.joblib"],
    },
    "random_forest": {
        "path": MODELS_DIR / "random_forest.joblib",
        "report": REPORTS_DIR / "model_random_forest_report.json",
        "algorithm": "RandomForestClassifier",
        "registered_name": "kdd99-random-forest",
        "extra_artifacts": [],
    },
    "xgboost": {
        "path": MODELS_DIR / "xgboost.joblib",
        "report": REPORTS_DIR / "model_xgboost_report.json",
        "algorithm": "XGBClassifier",
        "registered_name": "kdd99-xgboost",
        "extra_artifacts": [],
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def code_version() -> str:
    files = [
        PROJECT_ROOT / "src" / "training" / "train.py",
        PROJECT_ROOT / "src" / "preprocessing" / "Limpieza_transformacion.py",
        PROJECT_ROOT / "src" / "preprocessing" / "transformers.py",
    ]
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def flatten_parameters(parameters: dict) -> dict[str, str | int | float | bool]:
    return {
        key: value if isinstance(value, (str, int, float, bool)) else json.dumps(value)
        for key, value in parameters.items()
    }


def log_metrics(y_test, y_pred, y_pred_proba, report: dict) -> None:
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, y_pred, zero_division=0),
        "f1_macro": f1_score(y_test, y_pred, zero_division=0),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "auc": roc_auc_score(y_test, y_pred_proba),
    }
    metrics.update(report["metrics"])
    mlflow.log_metrics({key: float(value) for key, value in metrics.items()})


def already_imported(model_hash: str) -> bool:
    runs = mlflow.search_runs(
        experiment_names=[EXPERIMENT_NAME],
        filter_string=f"tags.model_sha256 = '{model_hash}'",
        max_results=1,
    )
    return not runs[runs["status"] == "FINISHED"].empty


def log_one_model(name: str, definition: dict, X_test, y_test, data_version: str, source_version: str) -> None:
    model_path = definition["path"]
    report_path = definition["report"]
    if not model_path.exists() or not report_path.exists():
        raise FileNotFoundError(f"Falta el modelo o reporte de {name}: {model_path}")

    model_hash = sha256_file(model_path)
    if already_imported(model_hash):
        print(f"MLflow ya contiene {name}; se omite para evitar duplicar el run.")
        return

    model = joblib.load(model_path)
    model_for_logging = model
    if name == "baseline_logistic_regression":
        scaler = joblib.load(MODELS_DIR / "scaler.joblib")
        model_for_logging = Pipeline([("scaler", scaler), ("classifier", model)])
    report = json.loads(report_path.read_text(encoding="utf-8"))
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    with tempfile.TemporaryDirectory() as temporary_dir:
        temporary_path = Path(temporary_dir)
        confusion_path = temporary_path / "confusion_matrix.png"
        ConfusionMatrixDisplay.from_predictions(
            y_test, y_pred, display_labels=["normal", "anomaly"], cmap="Blues"
        )
        plt.tight_layout()
        plt.savefig(confusion_path, dpi=150)
        plt.close()

        classification_path = temporary_path / "classification_report.json"
        classification_path.write_text(
            json.dumps(classification_report(y_test, y_pred, output_dict=True), indent=2),
            encoding="utf-8",
        )
        config_path = temporary_path / "run_config.json"
        config_path.write_text(
            json.dumps(
                {
                    "data_version_sha256": data_version,
                    "code_version_sha256": source_version,
                    "model_sha256": model_hash,
                    "feature_set": FEATURE_SET,
                    "n_features": X_test.shape[1],
                    "feature_columns": list(X_test.columns),
                    "leakage_columns_excluded": LEAKAGE_COLUMNS,
                    "test_rows": len(X_test),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        with mlflow.start_run(run_name=name) as run:
            parameters = {
                **report["parameters"],
                "algorithm": definition["algorithm"],
                "feature_set": FEATURE_SET,
                "random_seed": RANDOM_STATE,
                "data_version": data_version[:16],
            }
            mlflow.log_params(flatten_parameters(parameters))
            mlflow.set_tags(
                {
                    "model_sha256": model_hash,
                    "code_version_sha256": source_version,
                    "data_file": str(CLEAN_CSV_PATH.relative_to(PROJECT_ROOT)),
                    "tracking_mode": "import_existing_model_no_retraining",
                }
            )
            log_metrics(y_test, y_pred, y_pred_proba, report)
            mlflow.log_artifact(str(confusion_path), artifact_path="evaluation")
            mlflow.log_artifact(str(classification_path), artifact_path="evaluation")
            mlflow.log_artifact(str(config_path), artifact_path="configuration")
            mlflow.log_artifact(str(report_path), artifact_path="reports")
            figures_path = REPORTS_DIR / "figures"
            if figures_path.exists():
                mlflow.log_artifacts(str(figures_path), artifact_path="figures")
            mlflow.log_artifact(str(PROJECT_ROOT / "src" / "training" / "train.py"), artifact_path="source")
            mlflow.log_artifact(str(PROJECT_ROOT / "models" / "feature_pipeline.joblib"), artifact_path="preprocessing")
            mlflow.log_artifact(str(PROJECT_ROOT / "models" / "service_encoder.joblib"), artifact_path="preprocessing")
            for artifact in definition["extra_artifacts"]:
                if artifact.exists():
                    mlflow.log_artifact(str(artifact), artifact_path="preprocessing")

            signature = infer_signature(X_test, model_for_logging.predict(X_test))
            log_model = (
                mlflow.xgboost.log_model
                if name == "xgboost"
                else mlflow.sklearn.log_model
            )
            log_model(
                model_for_logging,
                artifact_path="model",
                signature=signature,
                registered_model_name=definition["registered_name"],
            )
            print(f"Registrado {name}: run_id={run.info.run_id}")


def main() -> None:
    if not CLEAN_CSV_PATH.exists():
        raise FileNotFoundError(f"No se encontró el dataset limpio: {CLEAN_CSV_PATH}")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    data_version = sha256_file(CLEAN_CSV_PATH)
    source_version = code_version()

    df = pd.read_csv(CLEAN_CSV_PATH)
    X_train, X_test, y_train, y_test, categorias_test, _ = prepare_train_test(df)
    del X_train, y_train

    for name, definition in MODEL_DEFINITIONS.items():
        log_one_model(name, definition, X_test, y_test, data_version, source_version)

    client = MlflowClient()
    client.set_registered_model_alias(
        name="kdd99-xgboost",
        alias="Production",
        version="1",
    )
    print("Alias 'Production' asignado a kdd99-xgboost versión 1.")

    print(f"MLflow listo. URI: {MLFLOW_TRACKING_URI}")
    print("Consulta la interfaz con: mlflow ui --backend-store-uri mlflow.db")


if __name__ == "__main__":
    main()