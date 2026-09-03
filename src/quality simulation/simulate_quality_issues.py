import json
import os
import numpy as np
import pandas as pd


def generate_clean_batch():
    """Genera un lote limpio de producción."""
    return pd.DataFrame(
        {
            "duration": [0, 2, 0, 5, 1],
            "protocol_type": ["tcp", "udp", "tcp", "icmp", "tcp"],
            "src_bytes": [181, 239, 235, 0, 215],
            "dst_bytes": [5450, 486, 1337, 0, 2050],
            "count": [9, 19, 2, 1, 5],
        }
    )


def inject_quality_issues(df: pd.DataFrame) -> pd.DataFrame:
    """Inyecta de forma controlada los 6 defectos de calidad exigidos."""
    corrupted_df = df.copy()

    # 1. Missing Values (Valores nulos)
    corrupted_df.loc[0, "duration"] = np.nan

    # 2. Duplicated Rows (Filas duplicadas)
    corrupted_df = pd.concat([corrupted_df, corrupted_df.iloc[[1]]], ignore_index=True)

    # 3. Extreme Outlier (Outlier extremo/negativo imprevisto)
    corrupted_df.loc[2, "src_bytes"] = -999999

    # 4. Incorrect Datatype (Tipo de dato incorrecto)
    corrupted_df["count"] = corrupted_df["count"].astype(str)
    corrupted_df.loc[3, "count"] = "tres"

    # 5. Unknown Category (Categoría desconocida)
    corrupted_df.loc[4, "protocol_type"] = "UNKNOWN_PROTOCOL"

    # 6. Schema Modification (Columna no esperada / Columna faltante)
    corrupted_df["unexpected_column"] = "anomaly_data"

    return corrupted_df


def validate_and_log_batch(df: pd.DataFrame, expected_schema: dict):
    """Pipeline de validación: Detecta -> Bloquea/Advierte -> Registra."""
    incidents = []

    # Validar Schema Modificado
    missing_cols = set(expected_schema.keys()) - set(df.columns)
    extra_cols = set(df.columns) - set(expected_schema.keys())

    if missing_cols:
        incidents.append({"type": "SCHEMA_ERROR", "detail": f"Columnas faltantes: {list(missing_cols)}", "severity": "HIGH"})
    if extra_cols:
        incidents.append({"type": "SCHEMA_MODIFICATION", "detail": f"Columnas inesperadas: {list(extra_cols)}", "severity": "MEDIUM"})

    # Validar Filas Duplicadas
    duplicated_count = int(df.duplicated().sum())
    if duplicated_count > 0:
        incidents.append({"type": "DUPLICATED_ROWS", "detail": f"Se encontraron {duplicated_count} filas duplicadas", "severity": "LOW"})

    # Validaciones por Columna
    for col, expected_type in expected_schema.items():
        if col in df.columns:
            # Missing Values
            nulls = int(df[col].isnull().sum())
            if nulls > 0:
                incidents.append({"type": "MISSING_VALUES", "detail": f"Columna '{col}' tiene {nulls} valores nulos", "severity": "MEDIUM"})

            # Datatypes & Contenido
            for idx, val in df[col].dropna().items():
                # Incorrect Datatype
                if expected_type == "numeric":
                    try:
                        num_val = float(val)
                        # Extreme Outlier
                        if num_val < 0 or num_val > 100000:
                            incidents.append({"type": "EXTREME_OUTLIER", "detail": f"Fila {idx}, '{col}' con valor fuera de rango: {val}", "severity": "HIGH"})
                    except ValueError:
                        incidents.append({"type": "INCORRECT_DATATYPE", "detail": f"Fila {idx}, '{col}' esperaba tipo numérico pero recibió: {val}", "severity": "HIGH"})

                # Unknown Category
                elif expected_type == "categorical":
                    allowed = ["tcp", "udp", "icmp"]
                    if val not in allowed:
                        incidents.append({"type": "UNKNOWN_CATEGORY", "detail": f"Fila {idx}, '{col}' contiene categoría no registrada: {val}", "severity": "HIGH"})

    # Determinar Acción (Bloquea o Advierte)
    has_high_severity = any(item["severity"] == "HIGH" for item in incidents)
    action_taken = "BLOCKED_BATCH" if has_high_severity else "WARNED_BATCH"

    summary = {
        "status": "FAILED" if has_high_severity else "PASSED_WITH_WARNINGS",
        "action": action_taken,
        "total_incidents": len(incidents),
        "incidents": incidents
    }

    return summary


def run_quality_simulation():
    print("SIMULACION DE INYECCION Y VALIDACION DE CALIDAD")

    # Definición de Esquema Esperado
    expected_schema = {
        "duration": "numeric",
        "protocol_type": "categorical",
        "src_bytes": "numeric",
        "dst_bytes": "numeric",
        "count": "numeric",
    }

    # 1. Generar e inyectar errores
    clean_batch = generate_clean_batch()
    corrupted_batch = inject_quality_issues(clean_batch)

    # 2. Ejecutar Pipeline de Validación
    validation_report = validate_and_log_batch(corrupted_batch, expected_schema)

    print(f"\nResultado del Pipeline: {validation_report['status']}")
    print(f"Acción Tomada: {validation_report['action']}")
    print(f"Total de Incidentes Registrados: {validation_report['total_incidents']}\n")

    for i, inc in enumerate(validation_report["incidents"], 1):
        print(f"{i}. [{inc['severity']}] {inc['type']}: {inc['detail']}")

    # 3. Registrar el Incidente (Logging)
    os.makedirs("reports", exist_ok=True)
    report_path = "reports/quality_incidents_report.json"
    with open(report_path, "w") as f:
        json.dump(validation_report, f, indent=4)

    print(f"\nReporte de incidentes registrado en '{report_path}'")


if __name__ == "__main__":
    run_quality_simulation()