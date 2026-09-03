import time
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score



def get_system_metrics(start_time: float, request_count: int, error_count: int):
    """Calcula métricas operativas de la infraestructura/API."""
    elapsed_time = time.time() - start_time
    latency_ms = (elapsed_time / request_count * 1000) if request_count > 0 else 0.0
    throughput = request_count / elapsed_time if elapsed_time > 0 else 0.0
    error_rate = (error_count / request_count) if request_count > 0 else 0.0
    availability = 1.0 - error_rate

    return {
        "latency_ms_per_req": round(latency_ms, 2),
        "throughput_req_per_sec": round(throughput, 2),
        "error_rate": round(error_rate, 4),
        "availability": round(availability, 4),
    }



def calculate_psi(reference: np.ndarray, production: np.ndarray, num_buckets: int = 10) -> float:
    """Calcula el Population Stability Index (PSI) entre la referencia y producción."""
    ref_clean = reference[~np.isnan(reference)]
    prod_clean = production[~np.isnan(production)]

    if len(ref_clean) == 0 or len(prod_clean) == 0:
        return 0.0

    percentiles = np.linspace(0, 100, num_buckets + 1)
    buckets = np.percentile(ref_clean, percentiles)
    buckets[0] -= 1e-5
    buckets[-1] += 1e-5

    ref_counts, _ = np.histogram(ref_clean, bins=buckets)
    prod_counts, _ = np.histogram(prod_clean, bins=buckets)

    ref_pct = ref_counts / len(ref_clean)
    prod_pct = prod_counts / len(prod_clean)

    # Reemplazar ceros para evitar división por cero / log(0)
    ref_pct = np.where(ref_pct == 0, 1e-4, ref_pct)
    prod_pct = np.where(prod_pct == 0, 1e-4, prod_pct)

    psi_val = np.sum((prod_pct - ref_pct) * np.log(prod_pct / ref_pct))
    return float(psi_val)


def evaluate_data_drift(
    df_ref: pd.DataFrame, df_prod: pd.DataFrame, numerical_features: list
):
    """Efectúa pruebas KS y PSI sobre los features numéricos."""
    drift_results = {}
    for col in numerical_features:
        if col in df_ref.columns and col in df_prod.columns:
            ref_data = df_ref[col].dropna().values
            prod_data = df_prod[col].dropna().values

            psi_value = calculate_psi(ref_data, prod_data)
            ks_stat, p_value = ks_2samp(ref_data, prod_data)

            # Clasificación de alerta basada en PSI
            if psi_value < 0.1:
                status = "OK"
            elif psi_value < 0.25:
                status = "WARNING"
            else:
                status = "ALERT (DRIFT DETECTED)"

            drift_results[col] = {
                "PSI": round(psi_value, 4),
                "KS_p_value": round(float(p_value), 4),
                "status": status,
            }
    return drift_results



def evaluate_model_performance(y_true=None, y_pred=None, y_prob=None):
    """Mide métricas del modelo (Detección de Anomalías / Clasificación)."""
    metrics = {}

    if y_pred is not None:
        # Muestra la tasa instantánea de anomalías detectadas P_prod(Y=1)
        anomaly_rate = float(np.mean(y_pred))
        metrics["anomaly_rate"] = round(anomaly_rate, 4)

    if y_true is not None and y_pred is not None:
        # Si se dispone de Ground Truth posterior
        metrics["precision"] = round(float(precision_score(y_true, y_pred, zero_division=0)), 4)
        metrics["recall"] = round(float(recall_score(y_true, y_pred, zero_division=0)), 4)
        metrics["f1_score"] = round(float(f1_score(y_true, y_pred, zero_division=0)), 4)

        if y_prob is not None and len(np.unique(y_true)) > 1:
            metrics["auc_roc"] = round(float(roc_auc_score(y_true, y_prob)), 4)

    return metrics



if __name__ == "__main__":
    print("--- INICIANDO DIAGNÓSTICO DE MONITOREO MLOps ---")

    # 1. Simulación System Monitoring
    start_sim = time.time() - 10
    system_report = get_system_metrics(
        start_time=start_sim, request_count=1000, error_count=2
    )
    print("\n[O1. SYSTEM MONITORING]")
    for k, v in system_report.items():
        print(f"  - {k}: {v}")

    # 2. Carga/Simulación Data Monitoring
    # Creación de muestras sintéticas (Referencia vs Producción)
    np.random.seed(42)
    ref_df = pd.DataFrame(
        {
            "duration": np.random.exponential(scale=2, size=1000),
            "src_bytes": np.random.normal(loc=200, scale=50, size=1000),
        }
    )

    # Simular Drift en Producción (Incremento brusco de tráfico)
    prod_df = pd.DataFrame(
        {
            "duration": np.random.exponential(scale=10, size=1000),  # Drift simulado
            "src_bytes": np.random.normal(loc=200, scale=50, size=1000),  # Sin drift
        }
    )

    drift_report = evaluate_data_drift(ref_df, prod_df, ["duration", "src_bytes"])
    print("\n[O2. DATA MONITORING (DRIFT)]")
    for feat, res in drift_report.items():
        print(f"  - Feature: {feat} | PSI: {res['PSI']} | Status: {res['status']}")

    # 3. Model Monitoring
    y_true_sim = np.random.choice([0, 1], size=500, p=[0.8, 0.2])
    y_pred_sim = np.random.choice([0, 1], size=500, p=[0.75, 0.25])
    model_report = evaluate_model_performance(y_true=y_true_sim, y_pred=y_pred_sim)

    print("\n[O3. MODEL MONITORING]")
    for k, v in model_report.items():
        print(f"  - {k}: {v}")