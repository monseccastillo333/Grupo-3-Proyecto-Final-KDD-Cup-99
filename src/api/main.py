import os
import glob
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict

import joblib
import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MODEL_URI = "models:/kdd99-xgboost@Production"
MODEL_CANDIDATES = [
    Path("models/xgboost.joblib"),
    Path("models/random_forest.joblib"),
    Path("models/baseline_logistic_regression.joblib"),
    Path("models/model.xgb"),
    Path("models/xgboost_model.xgb"),
]

app_state = {}


def _score_model_path(path: str):
    lower = path.lower()
    if "xgboost" in lower or "model.xgb" in lower or "model.json" in lower:
        priority = 0
    elif "random_forest" in lower:
        priority = 1
    elif "baseline" in lower or "logistic" in lower:
        priority = 2
    else:
        priority = 3
    return priority


def load_local_model():
    model_files = []
    for search_path in ["mlruns/**/artifacts/model/*", "models/*.*"]:
        model_files.extend(glob.glob(search_path, recursive=True))

    for candidate in MODEL_CANDIDATES:
        if candidate.exists():
            model_files.append(str(candidate))

    valid_files = [
        path for path in model_files
        if path.lower().endswith((".xgb", ".json", ".pkl", ".joblib", ".cb"))
    ]
    valid_files = sorted(set(valid_files), key=_score_model_path)

    seen = set()
    for path in valid_files:
        normalized = str(Path(path).resolve())
        if normalized in seen:
            continue
        seen.add(normalized)

        try:
            print(f"Cargando binario local desde: {path}")
            if path.lower().endswith((".xgb", ".json", ".cb")):
                booster = xgb.Booster()
                booster.load_model(path)
                return booster
            return joblib.load(path)
        except Exception as exc:
            print(f"Fallo al cargar {path}: {exc}")

    raise FileNotFoundError("No se encontraron artefactos de modelo en el proyecto.")


def _prepare_inference_dataframe(payload: Dict[str, Any], expected_features: list[str] | None):
    default_row = {
        "duration": 0,
        "protocol_type": "tcp",
        "service": "http",
        "flag": "SF",
        "src_bytes": 0,
        "dst_bytes": 0,
        "land": 0,
        "wrong_fragment": 0,
        "urgent": 0,
        "hot": 0,
        "num_failed_logins": 0,
        "logged_in": 0,
        "num_compromised": 0,
        "root_shell": 0,
        "su_attempted": 0,
        "num_file_creations": 0,
        "num_shells": 0,
        "num_access_files": 0,
        "is_guest_login": 0,
        "count": 0,
        "srv_count": 0,
        "same_srv_rate": 0,
        "diff_srv_rate": 0,
        "srv_diff_host_rate": 0,
        "dst_host_count": 0,
        "dst_host_srv_count": 0,
        "dst_host_same_srv_rate": 0,
        "dst_host_diff_srv_rate": 0,
        "dst_host_same_src_port_rate": 0,
        "dst_host_srv_diff_host_rate": 0,
        "dst_host_serror_rate": 0,
        "dst_host_srv_serror_rate": 0,
        "dst_host_rerror_rate": 0,
        "dst_host_srv_rerror_rate": 0,
    }
    row = {**default_row, **payload}
    data = pd.DataFrame([row])

    for col in [
        "duration", "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent", "hot",
        "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted",
        "num_file_creations", "num_shells", "num_access_files", "is_guest_login", "count",
        "srv_count", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
        "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
        "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
        "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate"
    ]:
        data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0)

    data["src_bytes_log"] = np.log1p(data["src_bytes"].clip(lower=0))
    data["dst_bytes_log"] = np.log1p(data["dst_bytes"].clip(lower=0))
    data["su_attempted_grouped"] = np.where(data.get("su_attempted", 0) != 0, 1, 0)

    encoder_path = Path("models/service_encoder.joblib")
    if encoder_path.exists():
        try:
            encoder = joblib.load(encoder_path)
            mapping = getattr(encoder, "mapping_", {})
            global_mean = getattr(encoder, "global_mean_", 0.0)
            data["service_anomaly_rate"] = data["service"].map(mapping).fillna(global_mean)
        except Exception:
            data["service_anomaly_rate"] = 0.5
    else:
        data["service_anomaly_rate"] = 0.5

    data["service"] = data["service"].astype(str)
    data = pd.get_dummies(data, columns=["protocol_type", "flag"], dtype=int)
    data = data.drop(columns=["service"], errors="ignore")

    for col in [
        "protocol_type_icmp", "protocol_type_tcp", "protocol_type_udp",
        "flag_OTH", "flag_REJ", "flag_RSTO", "flag_RSTOS0", "flag_RSTR",
        "flag_S0", "flag_S1", "flag_S2", "flag_S3", "flag_SF", "flag_SH"
    ]:
        data[col] = data.get(col, pd.Series(np.zeros(len(data), dtype=int))).astype(int)

    if expected_features:
        for col in expected_features:
            if col not in data.columns:
                data[col] = 0
        return data.reindex(columns=expected_features, fill_value=0)
    return data

@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"Conectando a MLflow Tracking URI: {MLFLOW_TRACKING_URI}")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    try:
        print(f"Cargando modelo desde MLflow: {MODEL_URI} ...")
        app_state["model"] = mlflow.xgboost.load_model(MODEL_URI)
        print("¡Modelo cargado exitosamente desde MLflow Registry!")
    except Exception as exc:
        print(f"Advertencia MLflow: {exc}. Intentando fallback local...")
        try:
            app_state["model"] = load_local_model()
            print("¡Modelo cargado exitosamente mediante fallback local!")
        except Exception as fallback_exc:
            print(f"Error crítico al cargar modelo: {fallback_exc}")
            app_state["model"] = None

    yield
    app_state.clear()

app = FastAPI(
    title="KDD Cup 99 Anomaly Detection API",
    version="1.0.0",
    lifespan=lifespan,
)

class PredictionResponse(BaseModel):
    anomaly: bool
    anomaly_score: float
    model_version: str

def ensure_model_loaded():
    if app_state.get("model") is None:
        try:
            app_state["model"] = load_local_model()
        except Exception:
            app_state["model"] = None
    return app_state.get("model")


@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    model = ensure_model_loaded()
    loaded = model is not None
    return {"status": "healthy" if loaded else "unhealthy", "model_loaded": loaded}

@app.post("/predict", response_model=PredictionResponse)
def predict(payload: Dict[str, Any]):
    model = ensure_model_loaded()
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="El modelo no se encuentra cargado.",
        )

    try:
        if hasattr(model, "get_booster"):
            expected_features = list(model.get_booster().feature_names)
        elif hasattr(model, "feature_names_in_"):
            expected_features = list(model.feature_names_in_)
        elif hasattr(model, "feature_names"):
            expected_features = list(model.feature_names)
        else:
            expected_features = None

        input_df = _prepare_inference_dataframe(payload, expected_features)

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_df)
            score = float(probabilities[0][1])
        else:
            dmatrix = xgb.DMatrix(input_df)
            preds = model.predict(dmatrix)
            score = float(preds[0])

        return PredictionResponse(
            anomaly=bool(score >= 0.5),
            anomaly_score=round(score, 4),
            model_version=MODEL_URI,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))