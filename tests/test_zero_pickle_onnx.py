"""
Tests for VaaniShield Zero-Pickle Security & Pure ONNX Inference Parity.
"""
import os
import numpy as np
import pytest

from vaanishield.boundary.features import extract_boundary_features, aggregate_boundary_features
from vaanishield.boundary.localization import find_acoustic_change_points
from vaanishield.fusion.classifier import FusionClassifier, ZeroPicklePredictor
from vaanishield.fusion.onnx_engine import VaaniShieldONNXEngine, verify_sha256_sidecar
from vaanishield.speaker_null.snap_projection import SpeakerNullingProjector
from vaanishield.utils.audio_io import load_audio


def test_onnx_engine_initialization_and_parity():
    """Verify that pure ONNX engine initializes with SHA-256 verification and matches PKL model."""
    onnx_path = "checkpoints/onnx/vaanishield_end_to_end.onnx"
    assert os.path.exists(onnx_path), f"ONNX model missing at {onnx_path}"

    engine = VaaniShieldONNXEngine(onnx_path, verify_checksum=True)

    pkl_data = FusionClassifier.load_checkpoint("checkpoints/fusion_checkpoint.pkl")
    pkl_model = pkl_data["model"]
    pkl_projector: SpeakerNullingProjector = pkl_data["projector"]

    np.random.seed(42)
    test_b = np.random.randn(8, 10).astype(np.float32)
    test_raw = np.random.randn(8, 768).astype(np.float32)
    test_raw /= np.linalg.norm(test_raw, axis=1, keepdims=True)

    # 1. Prediction via ONNX Engine (which computes SNAP projection internally)
    onnx_probs = engine.predict_proba(test_b, test_raw)

    # 2. Prediction via PyTorch / scikit-learn pipeline
    snap_emb = pkl_projector.project(test_raw)
    pkl_probs = pkl_model.predict_proba(test_b, snap_emb)

    max_diff = np.max(np.abs(onnx_probs - pkl_probs))
    assert max_diff < 1e-4, f"Parity mismatch between ONNX and PKL model: {max_diff}"

    # 3. Operational decision mapping
    for p in [0.10, 0.49, 0.85]:
        decision = engine.predict_operational_tier(p)
        assert decision in ["ALLOW_VERIFIED", "STEP_UP_AUTH", "BLOCK_AND_ALERT"]


def test_zero_pickle_npz_loading():
    """Verify loading from .npz parameter container without Python pickle."""
    npz_path = "checkpoints/onnx/model_parameters.npz"
    assert os.path.exists(npz_path), f"NPZ model missing at {npz_path}"

    data = FusionClassifier.load_checkpoint(npz_path, verify_checksum=True)
    assert data["format"] == "zero_pickle"

    model: ZeroPicklePredictor = data["model"]
    projector: SpeakerNullingProjector = data["projector"]

    np.random.seed(42)
    test_b = np.random.randn(5, 10).astype(np.float32)
    test_raw = np.random.randn(5, 768).astype(np.float32)
    test_raw /= np.linalg.norm(test_raw, axis=1, keepdims=True)

    snap_emb = projector.project(test_raw)
    probs = model.predict_proba(test_b, snap_emb)
    assert probs.shape == (5, 2)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
    assert np.allclose(np.sum(probs, axis=1), 1.0)

    point_est, ci_low, ci_high = model.predict_with_ci(test_b, snap_emb)
    assert len(point_est) == 5
    assert len(ci_low) == 5
    assert len(ci_high) == 5
    assert np.all(ci_low <= point_est) and np.all(point_est <= ci_high)


def test_zero_pickle_checksum_tamper_detection(tmp_path):
    """Verify that tampering with an archive triggers cryptographic integrity failure."""
    tampered_file = tmp_path / "tampered_model.npz"
    tampered_file.write_bytes(b"corrupted binary payload")

    sha_file = tmp_path / "tampered_model.npz.sha256"
    sha_file.write_text("0000000000000000000000000000000000000000000000000000000000000000  tampered_model.npz\n")

    with pytest.raises(ValueError, match="integrity check failed|Cryptographic integrity mismatch"):
        verify_sha256_sidecar(str(tampered_file))


def test_in_memory_audio_feature_parity():
    """Verify that in-memory audio array passing yields exact results as disk-based path."""
    wav_path = "data/raw/spk_1119_1119_020.wav"
    if not os.path.exists(wav_path):
        pytest.skip(f"Audio file not found: {wav_path}")

    audio, sr = load_audio(wav_path, target_sr=16000)

    # 1. Changepoints from path vs in-memory
    pts_path = find_acoustic_change_points(wav_path, max_points=3)
    pts_mem = find_acoustic_change_points(audio, sr=sr, max_points=3)
    assert len(pts_path) == len(pts_mem)
    for p1, p2 in zip(pts_path, pts_mem):
        assert abs(p1.timestamp_sec - p2.timestamp_sec) < 1e-4

    # 2. Features from path vs in-memory
    feats_path = extract_boundary_features(wav_path, pts_path)
    feats_mem = extract_boundary_features(audio, pts_path, sr=sr)
    agg_path = aggregate_boundary_features(feats_path)
    agg_mem = aggregate_boundary_features(feats_mem)
    assert np.allclose(agg_path, agg_mem, atol=1e-5)

    # 3. Projector embed from path vs in-memory
    proj = SpeakerNullingProjector()
    emb_path = proj.embed(wav_path)
    emb_mem = proj.embed(audio, sr=sr)
    assert np.allclose(emb_path, emb_mem, atol=1e-5)
