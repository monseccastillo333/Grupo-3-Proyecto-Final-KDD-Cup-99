from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_health_endpoint():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] in ["healthy", "unhealthy"]


def test_predict_success_schema():
    """Request Válido -> HTTP 200 -> Response Schema Válido."""
    payload = {
        "duration": 0,
        "protocol_type": "tcp",
        "service": "http",
        "flag": "SF",
        "src_bytes": 215,
        "dst_bytes": 45076,
        "logged_in": 1,
    }
    res = client.post("/predict", json=payload)
    assert res.status_code == 200

    body = res.json()
    assert "anomaly" in body
    assert "anomaly_score" in body
    assert "model_version" in body
    assert isinstance(body["anomaly"], bool)


def test_predict_invalid_input():
    """Demuestra el rechazo (HTTP 422) frente a un payload sintácticamente inválido."""
    res = client.post(
        "/predict",
        data="cadena_no_json",
        headers={"Content-Type": "application/json"},
    )
    assert res.status_code == 422