import json
import pytest
from fastapi.testclient import TestClient

from demo.api.server import app, SAMPLE_CATALOG


@pytest.fixture
def client():
    return TestClient(app)


def test_api_health(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data


def test_api_samples(client):
    response = client.get("/api/samples")
    assert response.status_code == 200
    samples = response.json()
    assert len(samples) >= 6
    for s in samples:
        assert "id" in s
        assert "name" in s
        assert "path" in s
        assert "expected" in s


def test_api_audio_download(client):
    response = client.get("/api/audio/sample_bonafide")
    assert response.status_code == 200
    assert response.headers["content-type"] in ["audio/wav", "audio/x-wav"]
    assert len(response.content) > 1000


def test_api_analyze_preset_sample(client):
    response = client.post(
        "/api/analyze",
        data={"sample_id": "sample_bonafide"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "spoof_probability" in data
    assert 0.0 <= data["spoof_probability"] <= 1.0
    assert "confidence_interval_95" in data
    assert len(data["confidence_interval_95"]) == 2
    assert "operational_decision" in data
    assert data["operational_decision"]["decision"] in ["ALLOW_VERIFIED", "STEP_UP_AUTH", "BLOCK_AND_ALERT"]
    assert "audio_sha256" in data
    assert len(data["audio_sha256"]) == 64
    assert "explanations" in data
    assert isinstance(data["explanations"], list)


def test_api_export_certificate(client):
    # Perform analysis first to get real payload
    analysis_resp = client.post(
        "/api/analyze",
        data={"sample_id": "sample_cloned"},
    )
    assert analysis_resp.status_code == 200
    payload = analysis_resp.json()

    cert_resp = client.post(
        "/api/export_certificate",
        json={"analysis_result": payload, "call_id": "CALL-TEST-9921"},
    )
    assert cert_resp.status_code == 200
    assert cert_resp.headers["content-type"] == "application/pdf"
    assert cert_resp.content.startswith(b"%PDF")
    assert len(cert_resp.content) > 1000



def test_api_webhook_dispatch_simulation(client):
    req_body = {
        "call_id": "CALL-98231",
        "decision": "BLOCK_AND_ALERT",
        "spoof_probability": 0.88,
        "target_webhook_url": None,
    }
    response = client.post("/api/webhook/dispatch", json=req_body)
    assert response.status_code == 200
    res = response.json()
    assert res["status"] == "success"
    assert res["mode"] == "simulation"
    assert res["dispatched_payload"]["action"] == "TERMINATE_CALL"
    assert res["dispatched_payload"]["delivery_status"] == "SIMULATED_LOCAL_DISPATCH"
