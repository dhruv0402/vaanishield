import numpy as np
import pytest
from pathlib import Path

from vaanishield.boundary.localization import find_codeswitch_points
from vaanishield.boundary.features import extract_boundary_features, aggregate_boundary_features
from vaanishield.speaker_null.snap_projection import SpeakerNullingProjector
from vaanishield.fusion.classifier import FusionClassifier
from vaanishield.fusion.explain import explain_prediction


def test_full_pipeline_real_audio_integration(tmp_path):
    """End-to-end integration test taking an audio fixture through:
    localization -> boundary features -> speaker nulling embedding -> fusion -> explanation.
    Asserts no stage returns degenerate/constant output.
    """
    raw_audio_path = Path("data/raw/spk_1119_1119_020.wav")
    if raw_audio_path.exists():
        audio_path = str(raw_audio_path)
    else:
        # Self-contained fixture when raw audio is excluded from git checkout
        sr = 16000
        duration = 3.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        # Create non-trivial speech-like harmonic signal
        sig = 0.5 * np.sin(2 * np.pi * 180.0 * t) + 0.25 * np.sin(2 * np.pi * 360.0 * t)
        sig = sig.astype(np.float32)
        fixture_path = tmp_path / "ci_fixture.wav"
        from vaanishield.utils.audio_io import save_audio
        save_audio(str(fixture_path), sig, sr)
        audio_path = str(fixture_path)

    tokens = ["Mera", "laptop", "start", "nahi", "ho", "raha,", "can", "you", "check", "the", "battery?"]

    # 1. Localization
    cs_points = find_codeswitch_points(
        audio_path=audio_path,
        transcript_tokens=tokens,
        window_ms_before=300,
        window_ms_after=300,
    )
    assert len(cs_points) > 0, "Localization must find at least one boundary point"
    timestamps = [p.timestamp_sec for p in cs_points]
    assert len(set(timestamps)) == len(timestamps), "Boundary timestamps must not be identical"

    # 2. Boundary Features
    b_feats = extract_boundary_features(audio_path, cs_points)
    assert len(b_feats) == len(cs_points)
    agg_b = aggregate_boundary_features(b_feats, aggregation="mean_and_max")
    assert agg_b.shape == (10,), f"Expected 10-dim aggregated boundary feature, got {agg_b.shape}"
    assert not np.all(agg_b == 0.0), "Boundary features must not be all zeros"
    assert not np.all(agg_b == agg_b[0]), "Boundary features must not be constant"

    # 3. Speaker Nulling Embedding
    projector = SpeakerNullingProjector()
    raw_emb = projector.embed(audio_path)
    assert raw_emb.shape == (768,), f"Expected 768-dim raw embedding, got {raw_emb.shape}"
    assert not np.all(raw_emb == 0.0), "Raw embedding must not be all zeros"
    assert not np.all(raw_emb == raw_emb[0]), "Raw embedding must not be constant"

    # Fit dummy subspace with varied components
    dummy_ref = np.random.RandomState(42).normal(0, 1, size=(10, 768)).astype(np.float32)
    projector.fit_speaker_subspace(dummy_ref, n_components=4)
    nulled_emb = projector.project(raw_emb)
    assert nulled_emb.shape == (768,)
    assert not np.allclose(nulled_emb, raw_emb), "Projected embedding must differ from raw embedding"

    # 4. Fusion Classifier
    classifier = FusionClassifier(classifier_type="logistic_regression", calibration="isotonic")
    # Fit with synthetic small variation
    X_b = np.vstack([agg_b, agg_b * 0.5, agg_b * 1.5, agg_b * 0.2])
    X_sn = np.vstack([nulled_emb, nulled_emb * 0.9, -nulled_emb, -nulled_emb * 0.5])
    y_dummy = np.array([0, 0, 1, 1], dtype=np.int32)
    classifier.fit(X_b, X_sn, y_dummy)

    prob = classifier.predict_proba(agg_b.reshape(1, -1), nulled_emb.reshape(1, -1))
    assert prob.shape == (1, 2)
    assert 0.0 <= prob[0, 1] <= 1.0

    # 5. Acoustic Explainability
    fused_vec = np.concatenate([agg_b, nulled_emb])
    explanations = explain_prediction(fused_vec, classifier, cs_points)
    assert len(explanations) == len(cs_points)
    for exp in explanations:
        assert "description" in exp
        assert len(exp["description"]) > 5
        # Ensure acoustic terminology is present
        assert any(term in exp["description"].lower() for term in ["f0", "pitch", "energy", "mfcc", "formant", "transition", "boundary"])
