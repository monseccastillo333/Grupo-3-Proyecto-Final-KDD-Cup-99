import os
import pandas as pd
import pytest

DATA_PATH = "data/processed/clean_kdd99.csv"


@pytest.fixture
def dataset():
    assert os.path.exists(DATA_PATH), f"No se encontró el archivo {DATA_PATH}"
    return pd.read_csv(DATA_PATH)


def test_schema_and_mandatory_columns(dataset):
    """Verifica presencia de columnas obligatorias."""
    required = ["duration", "src_bytes", "dst_bytes", "logged_in", "label"]
    for col in required:
        assert col in dataset.columns, f"Columna requerida ausente: {col}"


def test_data_types_and_nulls(dataset):
    """Verifica tipos numéricos y ausencia de nulos."""
    assert dataset.isnull().sum().sum() == 0, "El dataset contiene valores nulos"
    assert pd.api.types.is_numeric_dtype(dataset["duration"])
    assert pd.api.types.is_numeric_dtype(dataset["src_bytes"])


def test_value_ranges(dataset):
    """Verifica límites de rangos numéricos y binarios."""
    assert (dataset["duration"] >= 0).all()
    assert (dataset["src_bytes"] >= 0).all()
    assert set(dataset["logged_in"].unique()).issubset({0, 1})