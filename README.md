# Grupo-3-Proyecto-Final-KDD-Cup-99


## Data Quality (Sección F)

**Script:** `src/quality/data_quality_report.py`
**Versión exploratoria por celdas (VS Code):** `src/quality/data_quality_report_interactive.py`
**Reporte generado:** `reports/data_quality_report.json`

### Cómo ejecutarlo

```powershell
python src/quality/data_quality_report.py
```

No requiere argumentos: detecta automáticamente la raíz del proyecto y busca
`data/raw/kddcup_10_percent.csv`. Si el CSV no incluye una columna
`attack_category` (por ejemplo, si se generó con
`sklearn.datasets.fetch_kddcup99` en vez del script de ingesta propio), el
script la deriva en memoria a partir de `label`, sin modificar el archivo
fuente.

### Qué cubre

El script implementa los 17 puntos de diagnóstico exigidos en la Sección F
del proyecto: valores faltantes, faltantes codificados, duplicados,
inconsistencias lógicas, tipos de datos, categorías fuera de dominio,
fechas inválidas, datos imposibles, outliers, cardinalidad, skewness,
errores de unidad, leakage, imbalance, gaps temporales, correlación
excesiva y anomalías estadísticas. Cada hallazgo se imprime junto con su
justificación — el script diagnostica, no limpia ni transforma los datos.

### Hallazgos principales (sobre el 10% del dataset, 494,021 filas)

| Hallazgo | Resultado | Implicación |
|---|---|---|
| Duplicados exactos | 348,435 filas (70.53%) | No eliminar entre clases de ataque (es el patrón de tráfico real); sí revisar duplicados dentro de `normal`. Cifra consistente con el `validate.py` de Data Quality Gates. |
| Imbalance de clases | Ratio 7,528:1 (`dos` vs `u2r`) | Accuracy no es una métrica válida. Usar F1 por clase / AUC y `class_weight` en el modelado (secciones J/K). |
| Correlación excesiva | 17 pares de columnas con r > 0.9 | Candidatas a eliminar en Feature Engineering (mayormente variantes de `*serror_rate`/`*rerror_rate`). |
| Leakage potencial | 19 columnas con ventanas de cálculo (2 segundos y 100 conexiones) | Deben recalcularse de forma idéntica en el API de inferencia; `attack_category`/`is_anomaly` deben excluirse como features. |
| Skewness extrema | `src_bytes`, `urgent`, `num_compromised` (>400) | Features casi-binarias que solo se activan en ataques raros (U2R/R2L); no requieren transformación log estándar. |
| Anomalías estadísticas | 4.43% de filas con z-score extremo, solo 27.3% son ataques reales | Confirma que un umbral estadístico simple no basta como detector; se requiere un modelo entrenado. |
| Sin hallazgos | NaN, faltantes codificados, tipos, categorías fuera de dominio, datos imposibles | El dataset es internamente coherente en estos aspectos, verificado contra `kddcup.names` y `task.html` (Stolfo et al.). |

### Notas de documentación relevantes

- **Inconsistencia de nombre entre fuentes oficiales**: `kddcup.names` llama a una columna `is_host_login`, mientras que el paper original (`task.html`) la llama `is_hot_login` (relacionada con la lista de indicadores "hot"). Se documenta como ejemplo de inconsistencia de categorías/nombres entre fuentes del mismo dataset.
- **Dos ventanas de cálculo distintas conviven en el dataset**: las columnas de la Tabla 3 (`count`, `*_rate`) usan una ventana de 2 segundos; las columnas `dst_host_*` usan una ventana de las últimas 100 conexiones al mismo host. Ambas deben poder recalcularse igual en producción.


## EDA — Análisis Exploratorio de Datos (Sección H)

**Script:** `src/eda/eda_kdd.py`
**Versión exploratoria por celdas (VS Code):** `src/eda/eda_kdd_interactive.py`
**Figuras generadas:** `reports/figures/`

### Cómo ejecutarlo

```powershell
python src/eda/eda_kdd.py
```

Se ejecuta sobre el dataset crudo (`data/raw/kddcup_10_percent.csv`), antes
de la limpieza de Feature Engineering — el propósito del EDA es informar
esas decisiones, no depender de un dataset ya transformado.

### Hallazgos y decisiones que cambian

| # | Gráfico | Hallazgo | Decisión que cambia |
|---|---|---|---|
| 1 | Distribución de `attack_category` | Confirma visualmente el ratio 7,528:1 (dos vs u2r) ya cuantificado en calidad de datos | **Modelado**: métrica = F1/AUC por clase, nunca accuracy global |
| 2 | `src_bytes`/`dst_bytes` por categoría | `dos` tiene `dst_bytes` cercano a 0 (tráfico de una sola dirección); `normal` tiene mediana ~6 (log) | **Feature engineering**: aplicar `log1p`, no eliminar la columna; no requerido para modelos de árboles |
| 3 | `protocol_type` por categoría | `dos` concentrado en `icmp` (71.8%); `r2l`/`u2r` casi 100% en `tcp` (0% `icmp`) | **Negocio/Arquitectura**: posible pre-filtro rápido por protocolo en el API antes del modelo completo (reduce latencia) |
| 4 | Top 15 servicios por tasa de anomalía | Varios servicios con 100% de tasa de anomalía (`bgp`, `courier`, `echo`, etc.) | **Feature engineering**: usar target encoding de `service` en vez de one-hot (66 categorías); **ver limitación abajo** |
| 5 | `serror_rate` por categoría | Las 3 variantes de `serror_rate` aportan señal casi idéntica (correlación con target ≈ 0.227 las tres) | **Feature engineering**: conservar solo `dst_host_serror_rate`, descartar las otras 2 variantes redundantes |
| 6 | Heatmap + correlación con `is_anomaly` | `count` (r=0.75) y `srv_count` (r=0.57) son las features más discriminativas del dataset, por encima de las tasas de error | **Modelado**: priorizar `count`/`srv_count` en el set de features del baseline (Run 001 de MLflow) |

### Limitación importante del dataset (a documentar en el informe técnico)

El hallazgo #4 requiere una nota de cautela: varios servicios de red poco
comunes (`bgp`, `courier`, `echo`, `gopher`, `kshell`, entre otros) muestran
una tasa de anomalía del 100% exacto. Esto refleja cómo se construyó el
entorno simulado de DARPA/Lincoln Labs (el tráfico normal simulado solo usa
servicios "típicos", mientras que los escaneos de *probing* sí tocan
servicios inusuales) — es una limitación **documentada en la literatura**
(McHugh, 2000) sobre este dataset, no necesariamente una relación causal
real de ciberseguridad. Un modelo que dependa demasiado de esta señal podría
generalizar mal fuera de este dataset específico.

### Relación con el diagnóstico de calidad de datos (Sección F)

- Los 17 pares de columnas correlacionadas (r>0.9) identificados en calidad
  de datos se resuelven aquí con evidencia: se recomienda conservar
  `dst_host_serror_rate` del grupo de `serror_rate`.
- `count` y `srv_count` — las features más valiosas según este EDA —
  pertenecen a las 9 columnas de "ventana de 2 segundos" marcadas como
  riesgo de leakage en la Sección F. Esto eleva la prioridad de que el API
  de inferencia (Sección M) las recalcule de forma idéntica al histórico.


## Recomendaciones de limpieza y transformación (para Feature Engineering)

**De:** Diagnóstico de calidad de datos (Sección F) + EDA (Sección H)
**Para:** Feature Engineering / limpieza (Sección I)
**Fuente de datos:** `reports/data_quality_report.json` + `reports/figures/`


---

### 1. Duplicados — NO eliminar entre clases de ataque

**Hallazgo:** 348,435 filas duplicadas exactas (70.53% del dataset).

**Recomendación:** No usar `df.drop_duplicates()` de forma global. Los
duplicados entre conexiones de ataque (especialmente `dos`) son parte real
del patrón de tráfico (ráfagas repetidas de smurf/neptune), no basura.
Eliminarlos subestimaría artificialmente la frecuencia real de esos ataques.

**Si se necesita reducir volumen**, limitar la eliminación de duplicados
únicamente a filas dentro de la clase `normal`, donde no aportan señal
adicional.

---

### 2. Imbalance de clases — no usar oversampling ciego

**Hallazgo:** Ratio 7,528:1 entre `dos` (391,458) y `u2r` (52 filas).

**Recomendación:**
- Preferir `class_weight="balanced"` en el modelo sobre técnicas de
  resampling agresivas.
- Si se usa SMOTE, aplicarlo con mucho cuidado en `u2r` — con solo 52 casos
  reales, generar miles de ejemplos sintéticos duplicaría prácticamente el
  mismo patrón y puede producir sobreajuste.
- No usar accuracy para validar el efecto de cualquier técnica de balanceo;
  usar F1 por clase.

---

### 3. Columnas correlacionadas — eliminar redundancia, no señal

**Hallazgo:** 17 pares de columnas con correlación > 0.9, mayormente
variantes de `serror_rate`/`rerror_rate` calculadas sobre distintas
ventanas (por host, por servicio, por host-destino).

**Recomendación concreta (confirmada con el EDA):**
- Conservar **`dst_host_serror_rate`** (ventana de 100 conexiones, más
  estable) y descartar `serror_rate` y `srv_serror_rate` — las tres aportan
  correlación casi idéntica con el target (≈0.227), así que no se pierde
  señal al quedarse con una sola.
- Aplicar el mismo criterio para el grupo de `rerror_rate` (r=0.995 entre
  `rerror_rate` y `srv_rerror_rate`).
- Revisar el par `num_compromised`/`num_root` (r=0.994) — probablemente
  basta con una de las dos.

---

### 4. `service` (66 categorías) — target encoding, no one-hot

**Hallazgo:** Alta cardinalidad (66 valores únicos). Varios servicios poco
frecuentes muestran 100% de tasa de anomalía.

**Recomendación:** Usar target/frequency encoding (tasa histórica de
anomalía por servicio) en vez de one-hot encoding, para evitar 66 columnas
nuevas. **Importante:** documentar en el informe que esta señal puede
reflejar una limitación del entorno simulado del dataset (ver sección de
EDA) y no necesariamente una relación causal real — evitar que el modelo
dependa excesivamente de esto sin mencionar la limitación.

---

### 5. `src_bytes` / `dst_bytes` (skewness extrema) — transformar, no eliminar

**Hallazgo:** Skewness > 400 en ambas columnas. El EDA confirmó que sí
distinguen bien entre clases (ej. `dos` con `dst_bytes` cercano a 0 por ser
tráfico de una sola dirección).

**Recomendación:** Aplicar `log1p()` antes de modelos lineales o basados en
distancia (regresión logística, KNN, SVM). No es necesario para modelos de
árboles (Random Forest, XGBoost), que son invariantes a esta
transformación.

---

### 6. Columnas a EXCLUIR del set de features de entrenamiento

**Hallazgo (Leakage, sección F):** `attack_category` e `is_anomaly` se
derivan directamente de `label`.

**Recomendación:** Excluir `label`, `attack_category` e `is_anomaly` de
`X` (features) — deben usarse únicamente como fuente del target `y`.

---

### 7. Columnas de "ventana de tiempo" — prioridad alta, no solo limpieza

**Hallazgo:** 9 columnas se calculan sobre una ventana de 2 segundos
(`count`, `srv_count`, `*_rate`) y 10 columnas (`dst_host_*`) sobre una
ventana de 100 conexiones. El EDA confirmó que **`count` y `srv_count` son
las dos features más discriminativas de todo el dataset** (correlación
0.75 y 0.57 con el target, respectivamente).

**Recomendación:** No es una decisión de limpieza tradicional, sino de
diseño: el pipeline de Feature Engineering debe implementarse de forma que
estas 19 columnas puedan recalcularse de forma **idéntica** tanto en
entrenamiento como en el API de inferencia (sección M). Priorizar esto por
encima de otras transformaciones, dado que son las features de mayor valor
predictivo identificadas.

---

### 8. Sin acción requerida

Estas verificaciones de calidad salieron limpias — no requieren
transformación ni justificación adicional:

- Valores faltantes (NaN): 0 encontrados
- Valores faltantes codificados (`?`, `-`, etc.): 0 encontrados
- Tipos de datos: todos coinciden con `kddcup.names`
- Categorías fuera de dominio (`protocol_type`, `flag`): 0 encontradas
- Datos imposibles (negativos, tasas fuera de [0,1]): 0 encontrados

---

### 9. Columnas constantes — eliminar, no transformar

**Hallazgo:** Al revisar cuántos valores distintos tiene cada columna, se
encontraron 2 columnas con un solo valor en las 494,021 filas:

- `num_outbound_cmds` → siempre el mismo valor, sin ninguna variación
- `is_host_login` → siempre el mismo valor, sin ninguna variación

**Recomendación:** Eliminar ambas columnas del set de features. Una
columna sin variación no distingue entre conexiones normales y ataques,
así que no aporta ninguna señal útil al modelo — solo ocupa espacio y
tiempo de cómputo sin beneficio.

---

### 10. Valor inesperado en columna binaria (`su_attempted`)

**Hallazgo:** Según `kddcup.names`, `su_attempted` debería tener solo 2
valores posibles (0 o 1), pero en los datos reales aparecen 3 valores
distintos.

**Recomendación:** Esto es un artefacto conocido de este dataset
específico, documentado en la literatura académica sobre KDD Cup 99 — no
es un error de la ingesta. Feature Engineering debe decidir explícitamente
cómo tratar el tercer valor (por ejemplo, agruparlo junto con 1, ya que
ambos indican que sí hubo un intento de "su root", o mantenerlo como
categoría aparte) y documentar esa decisión.


M
## MLflow — Tracking de modelos existentes

El entrenamiento ya realizado se puede importar en MLflow sin volver a
entrenar los modelos. El importador registra un run por modelo, sus
hiperparámetros, `feature_set`, semilla, versión SHA-256 del dataset y del
código, métricas, matriz de confusión, gráficos, configuración y el modelo
registrado.

si no tiene instalado mlflow se ejecuta `python -m pip install mlflow`

Desde la raíz del proyecto, ejecutar una sola vez:

```powershell
python -m src.tracking.log_existing_models
```

Esto crea `mlflow.db` localmente y es idempotente: si un modelo ya fue
importado, no crea otro run. Para abrir la interfaz:

```powershell
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Los archivos `.joblib` y los reportes JSON originales se conservan como
fuente de auditoría.
