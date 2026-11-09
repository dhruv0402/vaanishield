"""Deterministic test-only models; never substitute these for research checkpoints."""
import numpy as np
import pytest


def pytest_addoption(parser):
    parser.addoption("--run-artifact-tests", action="store_true", default=False,
                     help="Require local trained checkpoints and test export parity.")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-artifact-tests"):
        marker = pytest.mark.skip(reason="requires trained artifacts; use --run-artifact-tests")
        for item in items:
            if "artifact" in item.keywords:
                item.add_marker(marker)


@pytest.fixture(scope="session")
def fitted_pipeline():
    from vaanishield.fusion.classifier import FusionClassifier
    from vaanishield.speaker_null.snap_projection import SpeakerNullingProjector

    rng = np.random.RandomState(2026)
    boundary = rng.normal(size=(20, 10)).astype(np.float32)
    raw = rng.normal(size=(20, 768)).astype(np.float32)
    raw /= np.linalg.norm(raw, axis=1, keepdims=True)
    projector = SpeakerNullingProjector()
    projector.fit_speaker_subspace(raw, n_components=3)
    model = FusionClassifier(calibration="sigmoid")
    model.fit(boundary, projector.project(raw), np.tile([0, 1], 10))
    return model, projector


@pytest.fixture
def api_pipeline(monkeypatch, tmp_path, fitted_pipeline):
    """Exercise real DSP/inference with a tiny fitted model and temporary audio."""
    from demo.api import server
    from vaanishield.utils.audio_io import save_audio

    model, projector = fitted_pipeline
    monkeypatch.setattr(server, "MODEL", model)
    monkeypatch.setattr(server, "PROJECTOR", projector)
    monkeypatch.setattr(server, "CFG", {})
    sr = 16000
    t = np.arange(sr * 3) / sr
    audio = (0.3 * np.sin(2 * np.pi * 200 * t) +
             0.1 * np.sin(2 * np.pi * 370 * t)).astype(np.float32)
    path = tmp_path / "synthetic_test_only.wav"
    save_audio(str(path), audio, sr)
    for sample_id in ("sample_bonafide", "sample_cloned"):
        monkeypatch.setitem(server.SAMPLE_MAP, sample_id, str(path))
    return model, projector
