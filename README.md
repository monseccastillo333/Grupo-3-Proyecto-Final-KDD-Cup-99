# Grupo-3-Proyecto-Final-KDD-Cup-99

## Instalación

Requiere Python 3.10+ instalado.

```powershell
# Clonar el repositorio y entrar a la carpeta del proyecto
git clone <URL-del-repositorio>
cd Grupo-3-Proyecto-Final-KDD-Cup-99

# (Opcional pero recomendado) crear entorno virtual
py -m venv venv
venv\Scripts\Activate.ps1

# Instalar dependencias
pip install -r requirements.txt
```

Si `Activate.ps1` está bloqueado por la política de ejecución de PowerShell,
se puede omitir el entorno virtual y usar directamente `py -m pip install`,
o ejecutar el proyecto llamando siempre al Python del venv por ruta completa
(`venv\Scripts\python.exe`).

## Orden de ejecución completo (pipeline end-to-end)

Todos los comandos se ejecutan desde la raíz del proyecto
(`Grupo-3-Proyecto-Final-KDD-Cup-99`).

```powershell
# 1. Ingesta de datos
python src/ingestion/ingest.py

# 2. Data Quality Gates (validación automática, falla si algo no cumple el mínimo)
python src/validation/validate.py

# 3. Limpieza y Feature Engineering (genera el dataset limpio y el pipeline compartido)
python -m src.preprocessing.Limpieza_transformacion

# 4. Entrenamiento y evaluación de modelos
python -m src.training.train

# 5. Registro en MLflow
python -m src.tracking.cargamlflow

# 6. Pruebas automatizadas
python -m pytest -q

Archivos de prueba:
- `test_data.py` — esquema, tipos, nulos y rangos del dataset procesado.
- `test_model.py` — validez de predicciones (rango [0,1]) y determinismo
  del modelo (misma entrada → misma salida).
- `test_api.py` — endpoint `/health`, contrato de respuesta de `/predict`
  con input válido (HTTP 200) y rechazo de input inválido (HTTP 422).

# 7. Levantar la API
python -m uvicorn src.api.main:app --reload --port 8000

- http://127.0.0.1:8000/docs#/

# 8. Monitoreo y simulación de drift (con la API corriendo o de forma independiente)
python -m src.monitoring.monitoring
python -m src.monitoring.simulate_drift

# 9. Simulación de problemas de calidad
python "src/quality simulation/simulate_quality_issues.py"


## Data Quality (Sección F) y EDA (Sección H)

**Notebook:** `src/quality & EDA/data quality & EDA.ipynb`
**Reporte generado:** `reports/data_quality_report.json`
**Figuras generadas:** `reports/figures/`

### Cómo ejecutarlo

Abrir el notebook en VS Code o Jupyter y ejecutar todas las celdas en
orden (no requiere argumentos: detecta automáticamente la raíz del
proyecto y busca `data/raw/kddcup_10_percent.csv`). Si el CSV no incluye
una columna `attack_category`, el notebook la deriva en memoria a partir
de `label`, sin modificar el archivo fuente.

Este notebook combina en un solo flujo el diagnóstico de calidad de datos
(Sección F) y el análisis exploratorio (Sección H), ya que ambos procesos
comparten la misma carga de datos y se complementan directamente: cada
hallazgo de calidad se profundiza con una visualización correspondiente.

### Qué cubre — Calidad de datos

Los 17 puntos de diagnóstico exigidos en la Sección F: valores faltantes,
faltantes codificados, duplicados, inconsistencias lógicas, tipos de
datos, categorías fuera de dominio, fechas inválidas, datos imposibles,
outliers, cardinalidad, skewness, errores de unidad, leakage, imbalance,
gaps temporales, correlación excesiva y anomalías estadísticas. Cada
hallazgo se documenta junto con su justificación — este bloque diagnostica,
no limpia ni transforma los datos.

### Hallazgos principales de calidad (sobre el 10% del dataset, 494,021 filas)

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

### Qué cubre — EDA

Se ejecuta sobre el dataset crudo, antes de la limpieza de Feature
Engineering — el propósito del EDA es informar esas decisiones, no
depender de un dataset ya transformado.

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

### Relación entre ambos análisis

- Los 17 pares de columnas correlacionadas (r>0.9) identificados en calidad
  de datos se resuelven aquí con evidencia: se recomienda conservar
  `dst_host_serror_rate` del grupo de `serror_rate`.
- `count` y `srv_count` — las features más valiosas según el EDA —
  pertenecen a las 9 columnas de "ventana de 2 segundos" marcadas como
  riesgo de leakage en el diagnóstico de calidad. Esto eleva la prioridad
  de que el API de inferencia (Sección M) las recalcule de forma idéntica
  al histórico.


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

## Data Quality Gates (Sección G)

**Script:** `src/validation/validate.py`

Antes de entrenar, el pipeline valida automáticamente que el dataset crudo
cumpla un mínimo de condiciones. Si alguna falla, el script se detiene con
un `AssertionError` — el entrenamiento no debe continuar con datos que no
pasen estas reglas.

### Cómo ejecutarlo

```powershell
python src/validation/validate.py
```

### Reglas implementadas (5, mínimo exigido por la Sección G)

| # | Regla | Umbral |
|---|---|---|
| 1 | El archivo crudo existe y no está vacío | `shape[0] > 0` |
| 2 | Volumen mínimo de filas | `>= 100,000` filas |
| 3 | Esquema válido (número de columnas) | exactamente 42 columnas |
| 4 | Presencia de la columna objetivo | `label` debe existir |
| 5 | Tasa de duplicados no catastrófica | `< 90%` |

Nota: el umbral de duplicados (90%) es deliberadamente holgado respecto al
70.06% real detectado en el diagnóstico de calidad — esta regla protege
contra una corrupción grave del archivo fuente, no reemplaza la
deduplicación que se aplica más adelante en Feature Engineering.

## Feature Engineering (Sección I)

**Script principal:** `src/preprocessing/Limpieza_transformacion.py`
**Clases de transformación (importables):** `src/preprocessing/transformers.py`

Este script integra en un único `sklearn.Pipeline` todas las
transformaciones de limpieza aplicadas al dataset, y garantiza que la misma
lógica se pueda usar tanto en entrenamiento como en inferencia (Sección M).

### Cómo ejecutarlo

```powershell
py -m src.preprocessing.Limpieza_transformacion
```

Se ejecuta con `-m` (no llamando al archivo por ruta directa) porque el
script importa `src.preprocessing.transformers` como parte del paquete del
proyecto.

### Qué hace

1. Carga el CSV crudo (`data/raw/kddcup_10_percent.csv`) y deriva
   `attack_category`/`is_anomaly` si no existen.
2. Aplica, en este orden, el pipeline de transformación:
   - `DeduplicateGlobal` — elimina duplicados exactos sobre todo el dataset.
   - `DropCorrelatedColumns` — elimina columnas con r > 0.9 (conserva
     `dst_host_serror_rate`, `dst_host_rerror_rate`, `num_compromised`).
   - `LogTransformer` — aplica `log1p` a `src_bytes`/`dst_bytes`.
   - `DropConstantColumns` — elimina `num_outbound_cmds` e `is_host_login`.
   - `SuAttemptedGrouper` — agrupa el valor inesperado de `su_attempted`.
3. Guarda el dataset limpio en `data/processed/kddcup_10_percent_clean.csv`.
4. Serializa el pipeline ya ajustado en `models/feature_pipeline.joblib`
   (usado también por la API de inferencia, ver Sección M).
5. Exporta el detalle de cada paso en `reports/feature_engineering_report.json`.

**Nota de diseño importante:** las clases de transformación viven en
`transformers.py`, no dentro del script ejecutable — esto es necesario para
que el archivo `.joblib` se pueda cargar correctamente desde cualquier otro
punto del proyecto (el notebook de modelado, `train.py`, o la API). Si las
clases se definieran dentro de `Limpieza_transformacion.py`, joblib no
podría deserializarlas fuera de ese mismo script.

**Nota de reproducibilidad conocida:** existe una discrepancia menor
(~0.33% de las filas) entre la deduplicación aplicada en memoria por este
pipeline y la que resulta de recargar el CSV ya persistido, atribuible a
pérdida de precisión decimal al guardar valores `float64` como texto plano.
Ver el informe técnico de modelado, Sección 5, para el detalle completo.

## Entrenamiento y Modelado (Secciones J y K)

**Script:** `src/training/train.py`

Entrena y evalúa los tres modelos de producción sobre el dataset ya
limpio, usando el mismo criterio de features en los tres.

### Requisito previo

Debe haberse ejecutado antes el paso de Feature Engineering (sección
anterior), ya que este script consume
`data/processed/kddcup_10_percent_clean.csv`.

### Cómo ejecutarlo

```powershell
py -m src.training.train
```

### Qué hace

1. Carga el dataset limpio y verifica que no queden duplicados (si
   encuentra alguno, lo reporta como advertencia — ver nota de
   reproducibilidad de la sección anterior).
2. Divide en entrenamiento/prueba (80/20, estratificado por `is_anomaly`).
3. Ajusta un codificador de tasa de anomalía por `service`
   (`ServiceTargetEncoder`), entrenado solo con el conjunto de
   entrenamiento para evitar leakage.
4. Aplica one-hot encoding a `protocol_type` y `flag`, y alinea las
   columnas entre train y test.
5. Entrena y evalúa, en orden: Regresión Logística (baseline), Random
   Forest, y XGBoost base.
6. Guarda cada modelo (`models/*.joblib`) y su reporte de parámetros y
   métricas (`reports/model_*_report.json`), junto con el `scaler` y el
   codificador de servicio usados.

### Modelo seleccionado

**XGBoost base**, sin ajuste de hiperparámetros. El criterio explícito de
selección es el recall en la categoría `u2r` (escalación de privilegios,
la clase más rara y de mayor severidad), donde XGBoost base obtiene 93.75%
frente a 81.25% de Random Forest y de las versiones con tuning automático.
El detalle completo de los 5 experimentos, incluyendo por qué el ajuste
automático de hiperparámetros no mejoró el resultado, está en el informe
técnico de modelado (`Informe_Tecnico_Modelado.docx`).



M
## MLflow — Tracking de modelos existentes

El entrenamiento ya realizado se puede importar en MLflow sin volver a
entrenar los modelos. El importador registra un run por modelo, sus
hiperparámetros, `feature_set`, semilla, versión SHA-256 del dataset y del
código, métricas, matriz de confusión, gráficos, configuración y el modelo
registrado.

si no tiene instalado mlflow se ejecuta `python -m pip install mlflow` o  `python -m pip install xgboost` o `pip install anyio`

Desde la raíz del proyecto, ejecutar una sola vez:

```powershell
python -m src.tracking.cargamlflow
```

Esto crea `mlflow.db` localmente y es idempotente: si un modelo ya fue
importado, no crea otro run. Para abrir la interfaz:

```powershell
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Los archivos `.joblib` y los reportes JSON originales se conservan como
fuente de auditoría.


## API

archivo creado en src/api/main.py

- pip install uvicorn

- python -m uvicorn src.api.main:app --reload --port 8000

- http://127.0.0.1:8000/docs#/


## Docker

archivos creados: Dockerfile, requirements.txt 

- docker build -t grupo-mlops-kdd99 .

- docker run -p 8000:8000 --name kdd99-service grupo-mlops-kdd99


## Pruebas

- pip install pytest httpx

- python -m pytest -q

## Monitoring 

- python -m src.monitoring.monitoring

## Simulación de producción y DRIFT

- python -m src.monitoring.simulate_drift

## Simulación obligatoria de problemas de calidad

- python "src/quality simulation/simulate_quality_issues.py"