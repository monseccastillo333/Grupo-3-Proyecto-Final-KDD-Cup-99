"""
retraining_policy.py — Sección R del proyecto: Estrategia de reentrenamiento

Implementa la lógica de decisión (no un sistema autónomo de Continuous
Training) que determina cuándo un modelo en producción DEBERÍA
reentrenarse, combinando dos señales independientes: drift de datos (PSI)
y degradación real de desempeño.

Regla de decisión:
    IF PSI > PSI_THRESHOLD  AND  performance < PERFORMANCE_THRESHOLD
    THEN Trigger Retraining Pipeline

Justificación (ver README / informe técnico, Sección R):
Data Drift != Model Degradation. Un cambio de distribución sin caída de
desempeño no amerita reentrenar (desperdicia cómputo y puede introducir
inestabilidad); una caída de desempeño sin drift detectado indica un
problema distinto (p. ej. un cambio de comportamiento del atacante que
evade el modelo sin alterar las estadísticas agregadas) y debe
investigarse antes de reentrenar automáticamente.
"""

from dataclasses import dataclass


PSI_THRESHOLD = 0.25          # por encima de esto, se considera drift significativo (sección O2)
PERFORMANCE_THRESHOLD = 0.80  # recall mínimo aceptable en la clase de referencia (ver nota abajo)


@dataclass
class RetrainingDecision:
    should_retrain: bool
    reason: str
    psi_value: float
    performance_value: float


def evaluate_retraining_trigger(psi_value: float, current_performance: float) -> RetrainingDecision:
    """Decide si se debe disparar el pipeline de reentrenamiento.

    current_performance: recall (u otra métrica de referencia) medido sobre
    tráfico reciente con ground truth disponible (ej. incidentes confirmados
    por el equipo de seguridad, o una muestra etiquetada manualmente).
    """
    drift_detected = psi_value > PSI_THRESHOLD
    performance_degraded = current_performance < PERFORMANCE_THRESHOLD

    if drift_detected and performance_degraded:
        return RetrainingDecision(
            should_retrain=True,
            reason=(
                f"Drift confirmado (PSI={psi_value:.4f} > {PSI_THRESHOLD}) "
                f"Y desempeño degradado (performance={current_performance:.4f} "
                f"< {PERFORMANCE_THRESHOLD}). Se dispara el pipeline de reentrenamiento."
            ),
            psi_value=psi_value,
            performance_value=current_performance,
        )

    if drift_detected and not performance_degraded:
        return RetrainingDecision(
            should_retrain=False,
            reason=(
                f"Drift detectado (PSI={psi_value:.4f}) pero el desempeño "
                f"sigue siendo aceptable ({current_performance:.4f}). No se "
                "reentrena: el modelo sigue siendo válido pese al cambio de "
                "distribución. Se recomienda monitoreo reforzado, no acción inmediata."
            ),
            psi_value=psi_value,
            performance_value=current_performance,
        )

    if not drift_detected and performance_degraded:
        return RetrainingDecision(
            should_retrain=False,
            reason=(
                f"El desempeño cayó ({current_performance:.4f}) sin drift "
                f"significativo en los datos (PSI={psi_value:.4f}). Esto sugiere "
                "una causa distinta a un cambio de distribución (ej. un cambio "
                "de comportamiento del atacante que evade el modelo sin alterar "
                "las estadísticas agregadas). Requiere investigación manual "
                "antes de reentrenar — reentrenar a ciegas no resolvería la "
                "causa raíz."
            ),
            psi_value=psi_value,
            performance_value=current_performance,
        )

    return RetrainingDecision(
        should_retrain=False,
        reason="Sin drift y sin degradación de desempeño. El modelo sigue vigente.",
        psi_value=psi_value,
        performance_value=current_performance,
    )


if __name__ == "__main__":
    print("--- SIMULACIÓN DE POLÍTICA DE REENTRENAMIENTO ---\n")

    escenarios = [
        ("Batch 1 (sin drift, sin degradación)", 0.02, 0.94),
        ("Batch 2 (drift moderado, modelo sigue bien)", 0.27, 0.88),
        ("Batch 3 (drift severo Y modelo degradado)", 1.46, 0.55),
        ("Caso hipotético: degradación sin drift", 0.05, 0.60),
    ]

    for nombre, psi, performance in escenarios:
        decision = evaluate_retraining_trigger(psi, performance)
        print(f"{nombre}")
        print(f"  ¿Reentrenar?: {'SÍ' if decision.should_retrain else 'NO'}")
        print(f"  Razón: {decision.reason}\n")