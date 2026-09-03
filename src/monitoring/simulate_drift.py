import json
import os
import numpy as np
import pandas as pd
from src.monitoring.monitoring import calculate_psi


def run_drift_simulation():
    np.random.seed(42)
    n_samples = 1000

    # 1. Dataset de Referencia (Distribución Base)
    reference_data = {
        "duration": np.random.exponential(scale=2.0, size=n_samples),
        "src_bytes": np.random.normal(loc=200, scale=30, size=n_samples),
        "dst_bytes": np.random.normal(loc=500, scale=100, size=n_samples),
    }
    df_ref = pd.DataFrame(reference_data)

    # 2. Batches de Producción Simulados
    # Batch 1: Comportamiento normal (Sin Drift)
    batch_1 = pd.DataFrame(
        {
            "duration": np.random.exponential(scale=2.1, size=n_samples),
            "src_bytes": np.random.normal(loc=202, scale=31, size=n_samples),
            "dst_bytes": np.random.normal(loc=505, scale=102, size=n_samples),
        }
    )

    # Batch 2: Ligera variación en el tráfico (Drift Moderado)
    batch_2 = pd.DataFrame(
        {
            "duration": np.random.exponential(scale=3.8, size=n_samples),
            "src_bytes": np.random.normal(loc=235, scale=40, size=n_samples),
            "dst_bytes": np.random.normal(loc=580, scale=120, size=n_samples),
        }
    )

    # Batch 3: Anomaly / DoS Simulation (Drift Crítico)
    batch_3 = pd.DataFrame(
        {
            "duration": np.random.exponential(scale=15.0, size=n_samples),
            "src_bytes": np.random.normal(loc=800, scale=200, size=n_samples),
            "dst_bytes": np.random.normal(loc=1800, scale=400, size=n_samples),
        }
    )

    batches = [
        ("PRODUCTION BATCH 1", batch_1),
        ("PRODUCTION BATCH 2", batch_2),
        ("PRODUCTION BATCH 3", batch_3),
    ]

    # 3. Evaluación de PSI por Lote
    results = {}
    print("SIMULACION DE DRIFT EN PRODUCCION")

    for batch_name, df_batch in batches:
        print(f"\nEvaluando: {batch_name}")
        batch_metrics = {}

        for col in df_ref.columns:
            psi = calculate_psi(df_ref[col].values, df_batch[col].values)

            # Clasificación según umbrales de PSI
            if psi < 0.10:
                status = "OK"
            elif psi < 0.25:
                status = "WARNING"
            else:
                status = "ALERT"

            batch_metrics[col] = {"PSI": round(psi, 4), "status": status}
            print(f"Feature: {col} | PSI: {psi:.4f} | Status: {status}")

        results[batch_name] = batch_metrics

    # Guardar reporte
    os.makedirs("reports", exist_ok=True)
    with open("reports/drift_simulation_report.json", "w") as f:
        json.dump(results, f, indent=4)

    print("\nReporte guardado en reports/drift_simulation_report.json")


if __name__ == "__main__":
    run_drift_simulation()