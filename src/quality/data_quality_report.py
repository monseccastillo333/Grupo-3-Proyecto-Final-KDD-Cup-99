"""
src/quality/data_quality_report.py

Diagnóstico de calidad de datos (Sección F) para el dataset KDD Cup 1999.

En esta sección determinaremos si hay: valores faltantes, faltantes codificados por símbolos, duplicados,
registros inconsistentes, tipos incorrectos, categorías inconsistentes,
fechas inválidas, datos imposibles, valores extremos, cardinalidad,
skewness, errores de unidad, leakage, imbalance, gaps temporales,
correlación excesiva y anomalías estadísticas.

Usamos la base de datos de entrenamiento (10% del dataset completo) para este diagnóstico siguiente:
- python src/quality/data_quality_report.py --input data/raw/kddcup_10_percent.csv

Se genera:
    - Reporte impreso en consola con justificación de cada hallazgo.
    - reports/data_quality_report.json con todos los resultados estructurados.
"""

# Importe de librerías

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
pd.set_option("display.width", 120)

# Columnas simbólicas (categóricas) conocidas según kddcup.names
SYMBOLIC_COLS = [
    "protocol_type", "service", "flag", "land", "logged_in",
    "is_host_login", "is_guest_login",
]

# Valores válidos conocidos para columnas categóricas dominados por el protocolo de red (RFC). Se usan para detectar valores fuera de dominio.
#
# 'protocol_type' y 'flag' solo pueden tener unos pocos valores posibles, y ya
# sabemos cuáles son (por ejemplo, 'protocol_type' solo puede ser: ('tcp', 'udp' o
# 'icmp'). Por eso podemos comparar los datos
# reales contra esta lista fija definida y detectar si aparece un dato diferente.
#
# 'service', en cambio, tiene 66 valores distintos posibles (nombres de
# servicios de red), así que no tiene sentido escribir una lista fija con
# los 66 — por eso esta columna se revisa de otra forma más adelante
# (sección de cardinalidad), no aquí.

EXPECTED_CATEGORIES = {
    "protocol_type": {"tcp", "udp", "icmp"},
    "flag": {
        "SF", "S0", "S1", "S2", "S3", "OTH", "REJ", "RSTO",
        "RSTOS0", "RSTR", "SH",
    },
    "land": {0, 1},
    "logged_in": {0, 1},
    "is_host_login": {0, 1},
    "is_guest_login": {0, 1},
}

# Columnas que, por definición del protocolo de red, NUNCA deberían ser negativas
NON_NEGATIVE_COLS = [
    "duration", "src_bytes", "dst_bytes", "wrong_fragment", "urgent", "hot",
    "num_failed_logins", "num_compromised", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds", "count",
    "srv_count", "dst_host_count", "dst_host_srv_count",
]

# Columnas de tasa (rate) que por definición matemática deben estar en [0, 1]
RATE_COLS = [
    "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
]


# Mapeo ataque -> categoría, según training_attack_types (DOS/U2R/R2L/PROBE)
ATTACK_TYPE_MAP = {
    "normal": "normal",
    "back": "dos", "land": "dos", "neptune": "dos", "pod": "dos",
    "smurf": "dos", "teardrop": "dos",
    "buffer_overflow": "u2r", "loadmodule": "u2r", "perl": "u2r", "rootkit": "u2r",
    "ftp_write": "r2l", "guess_passwd": "r2l", "imap": "r2l", "multihop": "r2l",
    "phf": "r2l", "spy": "r2l", "warezclient": "r2l", "warezmaster": "r2l",
    "ipsweep": "probe", "nmap": "probe", "portsweep": "probe", "satan": "probe",
}


def encontrar_raiz_proyecto(marcador="README.md"):
    """Busca hacia arriba desde el directorio actual hasta encontrar la raíz
    del repo (identificada por README.md). Evita este error(FileNotFoundError) que se presentó cuando el
    script se ejecuta desde una subcarpeta (ej. src/quality)."""
    actual = Path.cwd()
    for carpeta in [actual, *actual.parents]:
        if (carpeta / marcador).exists():
            return carpeta
    return actual


def asegurar_attack_category(df: pd.DataFrame) -> pd.DataFrame:
    """A veces el archivo CSV no trae la columna 'attack_category' (por
    ejemplo, cuando se generó con la herramienta de scikit-learn en vez del
    script de ingesta propio del equipo). Cuando eso pasa, esta función crea
    esa columna nueva usando la columna 'label' que sí existe, pero SOLO
    en la memoria de la computadora mientras corre el programa — nunca
    modifica ni guarda cambios en el archivo CSV original."""
    if "attack_category" in df.columns:
        return df
    if "label" not in df.columns:
        return df
    label_limpio = df["label"].astype(str).str.replace("b'", "", regex=False)
    label_limpio = label_limpio.str.replace("'", "", regex=False).str.rstrip(".")
    df["attack_category"] = label_limpio.map(ATTACK_TYPE_MAP).fillna("unknown")
    df["is_anomaly"] = (label_limpio != "normal").astype(int)
    return df


def section(title: str):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def check_missing_values(df: pd.DataFrame) -> dict:
    section("1. Valores faltantes (NaN reales)")
    nulls = df.isna().sum()
    nulls = nulls[nulls > 0]
    result = {"columns_with_nulls": nulls.to_dict(), "total_null_cells": int(nulls.sum())}
    if nulls.empty:
        print("No se encontraron NaN explícitos. Esto es consistente con la ficha "
              "de UCI en la página, que declara 'Has Missing Values: No'. Justificación: no se "
              "requiere imputación por NaN, pero eso NO descarta faltantes codificados "
              "(ver sección 2).")
    else:
        print(nulls)
    return result


def check_encoded_missing(df: pd.DataFrame) -> dict:
    section("2. Valores faltantes representados mediante símbolos")
    result = {}
# En este tipo de datasets es común que, en vez de dejar una celda vacía,
# alguien haya puesto el símbolo así '?' o cualquiera otro para representar un dato que falta 
# IMPORTANTE: en este dataset específico, las palabras 'other' y 'private'
# SÍ son nombres reales de servicios de red (no significan "dato faltante"),
# así que las excluimos a propósito de esta búsqueda para la columna
# 'service', para no marcarlas por error como si fueran un problema.
    suspicious_simbols = ["?", "-", "unknown", "none"]
    for col in df.select_dtypes(include="object").columns:
        simbols = suspicious_simbols if col != "service" else [t for t in suspicious_simbols]
        found = df[col].isin(simbols)
        if found.any():
            result[col] = int(found.sum())
    if not result:
        print("No se encontraron símbolos sospechosos ('?', '-', 'unknown', 'none') "
              "en columnas categóricas. Nota: 'other' y 'private' SÍ aparecen en "
              "'service' de forma legítima como nombres de servicio reales, no como "
              "código de faltante — se excluyeron intencionalmente del escaneo.")
    else:
        print(f"Simbolos sospechosos encontrados: {result}")
    return result


def check_duplicates(df: pd.DataFrame) -> dict:
    section("3. Registros duplicados")
    dup_count = int(df.duplicated().sum())
    dup_pct = round(dup_count / len(df) * 100, 2)
    print(f"Duplicados exactos: {dup_count:,} de {len(df):,} filas ({dup_pct}%).")
    print("Justificación de la decisión: KDD Cup 1999 es conocido en la literatura por "
          "tener un porcentaje MUY alto de duplicados (hasta ~78% en el set completo), "
          "producto de cómo se generó el tráfico simulado (ráfagas repetidas de la misma "
          "conexión, especialmente en ataques DOS tipo smurf/neptune). Si borráramos esos duplicados sin pensarlo," 
          "el modelo 'vería' menos ejemplos de esos ataques de los que realmente ocurren, y aprendería a"
          "descartarlos. Decisión recomendada: "
          "NO eliminar duplicados entre clases de ataque (son parte del patrón real), pero SÍ "
          "eliminar duplicados exactos dentro de la clase 'normal', donde no aportan "
          "información nueva y sí aumentan el desbalance.")
    return {"duplicate_rows": dup_count, "duplicate_pct": dup_pct}


def check_inconsistent_records(df: pd.DataFrame) -> dict:
    section("4. Registros lógicamente inconsistentes")
    result = {}
    if {"su_attempted", "root_shell"}.issubset(df.columns):
        inconsistent = df[(df["su_attempted"] == 1) & (df["root_shell"] == 0) & (df["num_root"] == 0)]
        result["su_attempted_sin_evidencia_root"] = int(len(inconsistent))
        print(f"Filas con su_attempted=1(alguien intentó usar el comando 'su' para volverse superusuario/administrador),"
                "pero sin root_shell ni num_root>0(o sea, no logró obtener acceso de root ni hacer cambios como root). "
              "No se eliminan: 'su_attempted' solo registra el intento, no el éxito, así que "
              "un intento fallido es información válida y es justamente como si identificará un ataque U2R"
              "(cuando alguien intenta tomar el control de nivel administrador de un equipo).")
    if {"land", "src_bytes", "dst_bytes"}.issubset(df.columns):
        land_no_bytes = df[(df["land"] == 1) & (df["src_bytes"] == 0) & (df["dst_bytes"] == 0)]
        result["land_attack_sin_bytes"] = int(len(land_no_bytes))
    return result


def check_incorrect_types(df: pd.DataFrame) -> dict:
    section("5. Tipos de datos incorrectos")
    result = {}
    for col in SYMBOLIC_COLS:
        if col in df.columns and col not in ("land", "logged_in", "is_host_login", "is_guest_login"):
            is_text = pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col])
            if not is_text:
                result[col] = str(df[col].dtype)
    numeric_expected = [c for c in df.columns if c not in SYMBOLIC_COLS + ["label", "attack_category", "is_anomaly"]]
    for col in numeric_expected:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            result[col] = str(df[col].dtype)
    if result:
        print(f"Columnas con tipo inesperado: {result}")
    else:
        print("Todas las columnas tienen el tipo esperado según kddcup.names "
              "(symbolic: object/int binario, continuous: numérico).")
    return result


def check_inconsistent_categories(df: pd.DataFrame) -> dict:
    section("6. Categorías inconsistentes / valores no permitidos en columnas de categorías")
    result = {}
    for col, expected in EXPECTED_CATEGORIES.items():
        if col not in df.columns:
            continue
        actual = set(df[col].unique())
        unexpected = actual - expected
        if unexpected:
            result[col] = list(unexpected)
    if result:
        print(f"Valores no permitidos en columnas de categorías: {result}")
        print("Justificación: cualquier valor no listado en kddcup.names/RFC de protocolos "
              "debe tratarse como categoría desconocida y mapearse explícitamente a "
              "'unknown' en Feature Engineering, no descartarse, ya que en producción "
              "simularemos justamente la aparición de categorías nuevas.")
    else:
        print("protocol_type, flag y las columnas binarias están dentro del dominio esperado.")
    if "service" in df.columns:
        n_services = df["service"].nunique()
        print(f"'service' tiene {n_services} categorías distintas (alta cardinalidad esperada, "
              "no se restringe a una lista cerrada, ver sección de cardinalidad).")
    return result

#Hallazgo adicional para la sección 6 (categorías inconsistentes):

#su_attempted tiene 3 valores distintos en los datos reales, aunque
#kddcup.names la documenta como binaria (0/1). Esto es un artefacto
#conocido de este dataset específico, no un error de la ingesta. Se
#recomienda que Feature Engineering decida explícitamente cómo tratar
#el tercer valor (ej. agruparlo con 1, o mantenerlo como categoría aparte).

def check_invalid_dates() -> dict:
    section("7. Fechas inválidas")
    print("No aplica: el dataset no contiene columnas de fecha/timestamp exactas. "
          "Justificación: KDD Cup 1999 solo guarda cuánto duró cada conexión (en segundos),"
          "pero no guarda EN QUÉ MOMENTO exacto ocurrió cada una. Por qué esto importa más adelante: "
          "en la parte del proyecto donde: simulamos cómo cambian los datos con el tiempo,"
          "no podemos ordenar las conexiones por fecha real porque no existe esa información. "
          "Vamos a usar el orden en que aparecen las filas en el archivo como si fuera el orden en que ocurrieron.")
    return {"applicable": False, "reason": "dataset sin columna de timestamp"}


def check_impossible_values(df: pd.DataFrame) -> dict:
    section("8. Datos imposibles")
    result = {}
    for col in NON_NEGATIVE_COLS:
        if col in df.columns:
            negatives = int((df[col] < 0).sum())
            if negatives:
                result[f"{col}_negativos"] = negatives
    for col in RATE_COLS:
        if col in df.columns:
            out_of_range = int(((df[col] < 0) | (df[col] > 1)).sum())
            if out_of_range:
                result[f"{col}_fuera_de_[0,1]"] = out_of_range
    if result:
        print(f"Valores imposibles detectados: {result}")
    else:
        print("No se detectaron negativos en columnas de conteo/bytes, ni tasas fuera "
              "del rango [0,1]. Esto es consistente y esperado dado el dominio del problema.")
    return result


def check_outliers(df: pd.DataFrame) -> dict:
    section("9. Valores extremos (outliers)")
    result = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [c for c in numeric_cols if c not in ("is_anomaly",)]
    for col in numeric_cols:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower, upper = q1 - 3 * iqr, q3 + 3 * iqr
        outliers = int(((df[col] < lower) | (df[col] > upper)).sum())
        if outliers > 0:
            result[col] = {"outliers": outliers, "pct": round(outliers / len(df) * 100, 2)}
    top5 = sorted(result.items(), key=lambda kv: kv[1]["pct"], reverse=True)[:5]
# Las 5 columnas donde encontramos más valores "raros" o extremos
# (usamos un método matemático estricto(con 3*IQR) para asegurarnos de que sean
# valores realmente exagerados, no solo un poco más altos de lo normal)
    print("Top 5 columnas con más outliers (método IQR*3, extremo severo):")
    for col, stats in top5:
        print(f"  {col}: {stats['outliers']:,} filas ({stats['pct']}%)")
    print("Justificación: no eliminamos valores extremos, en datos en tráfico de red, un valor 'raro' y muy alto en columnas como " \
        "los outliers severos en 'src_bytes', 'dst_bytes'(cantidad de datos enviados/recibidos) o "
        "'duration'(duración de la conexión) muchas veces NO es un error de los datos, "
        "puede ser justo la señal de que algo sospechoso está pasando. Por eso, en vez de borrar estos valores o 'suavizarlos' hacia un límite"
        "más razonable, simplemente los anotamos y seguimos revisando si tienen relación con el tipo de ataque, antes de decidir qué hacer con ellos.")
    return result


def check_cardinality(df: pd.DataFrame) -> dict:
    section("10. Cardinalidad")
    result = {}
    for col in df.select_dtypes(include="object").columns:
        result[col] = int(df[col].nunique())
    print(result)
    print("Justificación: 'service' con alta cardinalidad (66 nombres de servicios de red) "
          "Si la convirtiéramos a números creando una columna nueva por cada valor posible (66 columnas nuevas), " 
          "el dataset se volvería innecesariamente pesado y difícil de manejar. "
          "Por eso, se recomienda convertirla a números de otra forma: reemplazando" \
          "cada servicio por UN SOLO número que indique, por ejemplo, qué tan seguido ese " \
          "servicio está asociado a un ataque. Esto da la misma información útil, " \
          "pero sin crear tantas columnas nuevas.")
    return result


def check_skewness(df: pd.DataFrame) -> dict:
    section("11. Skewness")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    skew = df[numeric_cols].skew().sort_values(ascending=False)
    high_skew = skew[skew.abs() > 3]
    print(f"Columnas con |skewness|(Asimetría) > 3 (candidatas a transformación log/Box-Cox):\n{high_skew}")
    return {"high_skew_columns": high_skew.to_dict()}

# Recomendación de transformación logarítmica (log1p), basada en el
# patrón real de skewness observado:
#
# SÍ transformar (son cantidades continuas que pueden crecer mucho):
# src_bytes, dst_bytes, duration, num_compromised, num_root,
# num_file_creations, num_failed_logins, num_access_files, num_shells, hot
#
# NO transformar — son prácticamente banderas de sí/no (0 o 1), el
# logaritmo no aporta nada útil aquí:
# land, su_attempted, root_shell, is_guest_login
#
# NO transformar — ya son proporciones entre 0 y 1 por definición
# matemática, no tiene sentido aplicarles logaritmo:
# rerror_rate, srv_rerror_rate, dst_host_rerror_rate,
# dst_host_srv_rerror_rate, diff_srv_rate, dst_host_diff_srv_rate,
# srv_diff_host_rate, dst_host_srv_diff_host_rate
#
# REVISAR antes de decidir (casi siempre son 0, verificar cuántos valores
# distintos tienen realmente antes de transformar):
# urgent, wrong_fragment

# Recomendación de mayor prioridad: eliminar por completo (no transformar)

#num_outbound_cmds -> constante en todo el dataset (1 solo valor)
#is_host_login      -> constante en todo el dataset (1 solo valor)

#Justificación: una columna sin ninguna variación no aporta información
#para distinguir entre clases. Mantenerla no daña el modelo, pero es peso
#muerto — ocupa espacio y tiempo de cómputo sin ningún beneficio.



def check_unit_errors(df: pd.DataFrame) -> dict:
    section("12. Errores de unidad")
    print("Revisión conceptual: 'duration' está en segundos y 'src_bytes'/'dst_bytes' en "
          "bytes, según la documentación original. No encontramos evidencia de que se "
          "mezclen distintas unidades de medida dentro del dataset. "
          "columnas comparables Por ejemplo: no hay una columna que mida algo en "
          "bytes junto a otra columna que mida lo mismo pero en kilobytes (KB)")
    return {"applicable": True, "finding": "sin evidencia de mezcla de unidades"}


def check_leakage(df: pd.DataFrame) -> dict:
    section("13. Leakage")
    result = {}
    if "attack_category" in df.columns and "label" in df.columns:
        print("'attack_category' y 'is_anomaly' se derivan de 'label'. Ambas deben "
              "EXCLUIRSE del set de features de entrenamiento (son post-hoc del "
              "target) y usarse únicamente como target o para EDA.")
        result["derived_target_columns"] = ["attack_category", "is_anomaly", "label"]

    # Dos ventanas de cálculo distintas conviven en el dataset (ver task.html,
    # Stolfo et al.): 2 segundos (Tabla 3) vs. 100 conexiones al mismo host
    # (columnas dst_host_*, no tabuladas en el paper original pero descritas
    # conceptualmente). Ambas deben poder recalcularse igual en producción.
    time_window_cols = [c for c in df.columns if c in (
        "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate",
        "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    )]
    host_window_cols = [c for c in df.columns if c.startswith("dst_host_")]
    result["features_ventana_2_segundos"] = time_window_cols
    result["features_ventana_100_conexiones"] = host_window_cols

    print(f"Features con ventana de 2 segundos (Tabla 3, task.html): {len(time_window_cols)} columnas")
    print(f"Features con ventana de 100 conexiones al mismo host (dst_host_*): {len(host_window_cols)} columnas")
    print("Ambos grupos deben recalcularse de forma IDÉNTICA en el API de inferencia "
          "(sección M) respecto a como se calcularon en el histórico, o habrá "
          "leakage temporal / inconsistencia train-serving.")
    return result


def check_imbalance(df: pd.DataFrame) -> dict:
    section("14. Imbalance de clases")
    result = {}
    if "attack_category" in df.columns:
        counts = df["attack_category"].value_counts()
        pct = (counts / len(df) * 100).round(3)
        print(pd.DataFrame({"count": counts, "pct": pct}))
        result["attack_category_distribution"] = counts.to_dict()
        minority = counts.idxmin()
        majority = counts.idxmax()
        ratio = counts.max() / counts.min()
        print(f"\nRatio de desbalance mayoría/minoría: {ratio:,.0f}:1 "
              f"({majority} vs {minority}).")
        print("Justificación: con este ratio, accuracy es una métrica engañosa. El "
              "proyecto debe reportar F1/AUC por clase (sección J) y considerar "
              "class_weight, SMOTE controlado o umbral ajustado — nunca oversampling "
              "ciego que duplique masivamente clases con solo 3-8 instancias (u2r).")
    return result


def check_temporal_gaps() -> dict:
    section("15. Gaps temporales")
    print("No aplica de forma directa: no existe timestamp absoluto (ver sección 7). "
          "El orden de filas se usará como proxy temporal para la simulación de "
          "producción (sección P), asumiendo que el archivo preserva el orden de "
          "captura original de la competencia.")
    return {"applicable": False}




def check_excessive_correlation(df: pd.DataFrame) -> dict:
    section("16. Correlación excesiva")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [c for c in numeric_cols if c != "is_anomaly"]
    corr = df[numeric_cols].corr().abs()
    pairs = []
    for i, col_i in enumerate(corr.columns):
        for col_j in corr.columns[i + 1:]:
            val = corr.loc[col_i, col_j]
            if val > 0.9:
                pairs.append((col_i, col_j, round(float(val), 3)))
    pairs.sort(key=lambda x: x[2], reverse=True)
    for a, b, v in pairs[:10]:
        print(f"  {a} <-> {b}: r={v}")
    if not pairs:
        print("No se encontraron pares con correlación > 0.9.")
    print("Justificación: pares altamente correlacionados (ej. srv_serror_rate vs "
          "serror_rate) son candidatos a eliminar uno de los dos en Feature Engineering "
          "para reducir redundancia y multicolinealidad en modelos lineales.")
    return {"high_correlation_pairs": pairs}

# No aplica directamente: como ya vimos antes, este dataset no tiene una
# fecha u hora exacta guardada para cada conexión.
#
#"Proxy" = algo que usamos como sustituto o reemplazo de otra cosa que no tenemos. 
# En este caso, no tenemos la fecha/hora real de cada conexión, así que vamos a usar 
# "el orden en que aparecen las filas" como si fuera esa fecha/hora.
#
# Qué vamos a hacer en su lugar: en la parte del proyecto donde simulamos
# cómo cambiarían los datos con el tiempo (sección P), vamos a usar el
# orden en que aparecen las filas en el archivo como si fuera el orden
# real en que ocurrieron las conexiones. Esto asume que el archivo guardó
# las conexiones en el mismo orden en que se capturaron originalmente
# durante la competencia.


def check_statistical_anomalies(df: pd.DataFrame) -> dict:
    section("17. Anomalías estadísticas (z-score global)")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [c for c in numeric_cols if c != "is_anomaly"]
    z = (df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std(ddof=0)
    extreme_rows = (z.abs() > 5).any(axis=1)
    n_extreme = int(extreme_rows.sum())
    pct = round(n_extreme / len(df) * 100, 2)
    print(f"Filas con al menos una columna con |z-score| > 5: {n_extreme:,} ({pct}%).")
    if "is_anomaly" in df.columns:
        overlap = int((extreme_rows & (df["is_anomaly"] == 1)).sum())
        print(f"De esas, {overlap:,} ya están etiquetadas como ataque "
              f"({round(overlap / max(n_extreme, 1) * 100, 1)}% de solapamiento).")
        print("Nota: el solapamiento parcial (no total) confirma que el z-score global "
              "por sí solo NO es un detector de anomalías suficiente para este problema "
              "— hay ataques con features dentro de rango 'normal' y normales con "
              "outliers legítimos (ej. transferencias grandes de archivos).")
    return {"extreme_rows": n_extreme, "extreme_pct": pct}


# Nota importante:
#
# Encontramos que el z-score (una forma de detectar valores raros) solo
# coincide PARCIALMENTE con los ataques reales — no coincide del todo.
#
#"z-score" = una forma de medir qué tan lejos está un valor del promedio. Un z-score alto significa "este valor es raro comparado con el resto".
#"Solapamiento" = cuánto se cruzan o coinciden dos grupos distintos (en este caso: las filas "raras" según el z-score,
#  y las filas que sabemos que son ataques reales).
#
# Esto nos dice algo importante: un método simple de "buscar valores raros"
# NO es suficiente para detectar ataques en este problema. Hay dos tipos de
# casos que confunden a este método:
#
# 1. Ataques reales que NO se ven raros (sus valores están dentro de lo
#    que parecería "normal").
# 2. Conexiones normales que SÍ se ven raras, pero no son ataques (por
#    ejemplo, alguien transfiriendo un archivo muy grande de forma legítima).
#
# Por eso se necesita un modelo más inteligente (entrenado con IA), no solo
# una regla matemática simple que busque valores extremos.

def main():
    parser = argparse.ArgumentParser(description="Diagnóstico de calidad de datos KDD Cup 1999")
    parser.add_argument("--input", default=None,
                         help="Ruta al CSV/parquet ya ingerido. Si se omite, se busca "
                              "automáticamente en <raíz_del_proyecto>/data/raw/kddcup_10_percent.csv")
    parser.add_argument("--output", default=None,
                         help="Ruta donde guardar el reporte JSON. Si se omite, se usa "
                              "<raíz_del_proyecto>/reports/data_quality_report.json")
    args = parser.parse_args()

    raiz = encontrar_raiz_proyecto()
    input_path = Path(args.input) if args.input else raiz / "data/raw/kddcup_10_percent.csv"
    output_path = Path(args.output) if args.output else raiz / "reports/data_quality_report.json"

    if input_path.suffix == ".parquet":
        df = pd.read_parquet(input_path)
    else:
        df = pd.read_csv(input_path)

    df = asegurar_attack_category(df)

    print(f"Dataset cargado: {df.shape[0]:,} filas x {df.shape[1]} columnas desde {input_path}")

    report = {
        "input_file": str(input_path),
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "missing_values": check_missing_values(df),
        "encoded_missing": check_encoded_missing(df),
        "duplicates": check_duplicates(df),
        "inconsistent_records": check_inconsistent_records(df),
        "incorrect_types": check_incorrect_types(df),
        "inconsistent_categories": check_inconsistent_categories(df),
        "invalid_dates": check_invalid_dates(),
        "impossible_values": check_impossible_values(df),
        "outliers": check_outliers(df),
        "cardinality": check_cardinality(df),
        "skewness": check_skewness(df),
        "unit_errors": check_unit_errors(df),
        "leakage": check_leakage(df),
        "imbalance": check_imbalance(df),
        "temporal_gaps": check_temporal_gaps(),
        "excessive_correlation": check_excessive_correlation(df),
        "statistical_anomalies": check_statistical_anomalies(df),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    section("Reporte guardado")
    print(f"JSON estructurado guardado en: {output_path}")


if __name__ == "__main__":
    main()
