import glob
import pandas as pd
import pytest
import xgboost as xgb


@pytest.fixture
def booster():
    files = (
        glob.glob("models/*.xgb")
        + glob.glob("models/*.json")
        + glob.glob("mlruns/**/artifacts/model/model.xgb", recursive=True)
        + glob.glob("mlruns/**/artifacts/model/model.json", recursive=True)
    )
    assert len(files) > 0, "No hay artefactos de modelo disponibles para probar."
    model = xgb.Booster()
    model.load_model(files[0])
    return model


def test_model_valid_input(booster):
    """Input válido -> Predicción con score en rango [0, 1]."""
    sample = pd.DataFrame([{"duration": 0, "src_bytes": 181, "dst_bytes": 5450, "logged_in": 1}])
    if booster.feature_names:
        sample = sample.reindex(columns=booster.feature_names, fill_value=0)

    dmatrix = xgb.DMatrix(sample)
    preds = booster.predict(dmatrix)

    assert len(preds) == 1
    assert 0.0 <= float(preds[0]) <= 1.0


def test_model_determinism(booster):
    """Misma entrada -> Misma salida exactas en múltiples ejecuciones."""
    sample = pd.DataFrame([{"duration": 0, "src_bytes": 100, "dst_bytes": 200, "logged_in": 0}])
    if booster.feature_names:
        sample = sample.reindex(columns=booster.feature_names, fill_value=0)

    dm = xgb.DMatrix(sample)
    pred1 = float(booster.predict(dm)[0])
    pred2 = float(booster.predict(dm)[0])

    assert pred1 == pred2