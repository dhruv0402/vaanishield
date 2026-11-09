"""A model-free checkout is live, but must never claim inference readiness."""
import hashlib
import io
import wave

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from demo.api import server
from vaanishield.fusion.classifier import FusionClassifier


@pytest.fixture
def model_directory(monkeypatch, tmp_path):
    # Exercise the real loader with absent or invalid files, not a mocked pipeline.
    config = tmp_path / "default.yaml"
    config.write_text("{}")
    monkeypatch.setattr(server, "CONFIG_PATH", str(config))
    monkeypatch.setattr(server, "MODEL", None)
    monkeypatch.setattr(server, "PROJECTOR", None)
    monkeypatch.setattr(server, "CFG", None)
    monkeypatch.chdir(tmp_path)
    # The optional dashboard may already be mounted by the imported app.
    (tmp_path / "web/dist").mkdir(parents=True)
    path = tmp_path / "checkpoints/onnx/model_parameters.npz"
    path.parent.mkdir(parents=True)
    return path


def wav_payload():
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(8000)
        wav.writeframes(b"\x00\x00" * 8000)
    return output.getvalue()


def test_liveness_does_not_require_a_model(model_directory):
    with TestClient(server.app) as client:
        assert client.get("/api/health").status_code == 200
    assert server.MODEL is None


@pytest.mark.parametrize("artifact", ["absent", "unverified", "corrupt", "malformed"])
def test_readiness_rejects_unavailable_model(model_directory, artifact):
    if artifact != "absent":
        model_directory.write_bytes(b"invalid model")
    if artifact == "corrupt":
        model_directory.with_suffix(".npz.sha256").write_text("0" * 64)
    if artifact == "malformed":
        model_directory.with_suffix(".npz.sha256").write_text(
            hashlib.sha256(model_directory.read_bytes()).hexdigest()
        )
    with TestClient(server.app) as client:
        response = client.get("/api/ready")
    assert response.status_code == 503
    assert response.json()["status"] == "unavailable"
    assert response.json()["code"] == "model_unavailable"
    assert server.MODEL is None


def test_readiness_recovers_after_model_is_provisioned(model_directory, fitted_pipeline):
    with TestClient(server.app) as client:
        assert client.get("/api/ready").status_code == 503
        model, projector = fitted_pipeline
        # This tiny test-only model verifies loading, not detector accuracy.
        FusionClassifier.save_parameters_npz(model, projector, str(model_directory))
        response = client.get("/api/ready")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"
    assert server.MODEL is not None
    assert server.PROJECTOR is not None


def test_invalid_npz_does_not_fall_back_to_pickle(model_directory, fitted_pipeline):
    model, projector = fitted_pipeline
    FusionClassifier.save_checkpoint({"model": model, "projector": projector},
                                     "checkpoints/fusion_checkpoint.pkl")
    model_directory.write_bytes(b"unverified model")
    with TestClient(server.app) as client:
        assert client.get("/api/ready").status_code == 503
    assert server.MODEL is None


def test_analyze_rejects_missing_model_without_a_prediction(model_directory):
    with TestClient(server.app) as client:
        response = client.post("/api/analyze", files={
            "file": ("test.wav", wav_payload(), "audio/wav"),
        })
    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "model_unavailable"
    assert "spoof_probability" not in response.json()


def test_stream_closes_cleanly_without_a_model(model_directory):
    with TestClient(server.app) as client:
        with client.websocket_connect("/ws/stream_analyze") as websocket:
            message = websocket.receive_json()
            assert message["type"] == "error"
            assert message["code"] == "model_unavailable"
            assert "spoof_probability" not in message
            with pytest.raises(WebSocketDisconnect) as exc:
                websocket.receive_json()
            assert exc.value.code == 1013
