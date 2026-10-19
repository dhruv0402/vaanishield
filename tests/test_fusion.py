"""
Unit tests for vaanishield.fusion.

Critically, includes a PRIVACY test (see docs/PRIVACY_AND_ETHICS.md section 6):
explain_prediction() output must never contain transcript content, only descriptions of
acoustic patterns. Do not remove or weaken this test -- it enforces a stated project
constraint, not an implementation preference.
"""

import numpy as np
import pytest
from vaanishield.fusion.classifier import FusionClassifier
from vaanishield.fusion.explain import explain_prediction
from vaanishield.boundary.localization import CodeSwitchPoint


def test_explanation_does_not_contain_transcript_text():
    """explain_prediction() output must never contain words or transcript content.

    Strictly tests the non-negotiable privacy guarantee from docs/PRIVACY_AND_ETHICS.md.
    """
    sample_transcript_words = [
        "account", "police", "rupees", "twenty", "thousand", "hospital",
        "emergency", "password", "otp", "aadhaar", "money", "kidnap"
    ]

    boundary_pts = [
        CodeSwitchPoint(
            timestamp_sec=2.45,
            from_lang="hi",
            to_lang="en",
            window_start_sec=2.15,
            window_end_sec=2.75,
        ),
        CodeSwitchPoint(
            timestamp_sec=4.12,
            from_lang="en",
            to_lang="hi",
            window_start_sec=3.82,
            window_end_sec=4.42,
        ),
    ]

    # Synthetic fused feature vector: 10 boundary features + 768 speaker-nulled dims
    fused_dim = 10 + 768
    fused_vector = np.random.uniform(0.1, 1.0, size=fused_dim).astype(np.float32)

    explanations = explain_prediction(
        fused_feature_vector=fused_vector,
        model=None,
        codeswitch_points=boundary_pts,
    )

    assert len(explanations) == 2
    for exp in explanations:
        assert "description" in exp
        desc = exp["description"].lower()
        # Verify that no transcript words leak into the explanation string
        for word in sample_transcript_words:
            assert word not in desc, f"Privacy violation: explanation leaked transcript word '{word}'"

        # Verify it contains acoustic pattern terms
        assert any(cue in desc for cue in ["pitch", "energy", "phonetic", "formant", "spectral", "acoustic"])


def test_fusion_classifier_predict_with_ci_returns_valid_interval():
    # Train dummy fusion classifier
    n_samples = 30
    bf = np.random.uniform(0.0, 2.0, size=(n_samples, 10)).astype(np.float32)
    sn = np.random.normal(0.0, 1.0, size=(n_samples, 768)).astype(np.float32)
    labels = np.array([0] * 15 + [1] * 15)

    clf = FusionClassifier(classifier_type="logistic_regression", calibration="isotonic")
    clf.fit(bf, sn, labels)

    # Test test samples
    test_bf = bf[:5]
    test_sn = sn[:5]
    probs, lower, upper = clf.predict_with_ci(test_bf, test_sn, n_bootstrap=200, ci=0.95)

    assert len(probs) == 5
    assert len(lower) == 5
    assert len(upper) == 5

    for p, l, u in zip(probs, lower, upper):
        assert 0.0 <= p <= 1.0
        assert 0.0 <= l <= 1.0
        assert 0.0 <= u <= 1.0
        assert l <= p <= u or np.isclose(l, p, atol=1e-3) or np.isclose(u, p, atol=1e-3)


def test_predict_operational_tier_decision_tiers():
    n_samples = 30
    bf = np.random.uniform(0.0, 2.0, size=(n_samples, 10)).astype(np.float32)
    sn = np.random.normal(0.0, 1.0, size=(n_samples, 768)).astype(np.float32)
    labels = np.array([0] * 15 + [1] * 15)

    clf = FusionClassifier(classifier_type="logistic_regression", calibration="isotonic")
    clf.fit(bf, sn, labels)

    tiers = clf.predict_operational_tier(bf[:3], sn[:3], tau_review=0.48, tau_block=0.52)
    assert len(tiers) == 3
    for t in tiers:
        assert "decision" in t
        assert t["decision"] in ["ALLOW_VERIFIED", "STEP_UP_AUTH", "BLOCK_AND_ALERT", "PASS", "STEP_UP_REVIEW", "BLOCK"]
        assert "risk_tier" in t
        assert t["risk_tier"] in ["SECURE", "ELEVATED", "CRITICAL", "LOW", "SUSPICIOUS"]
        assert "ci_95" in t
        assert len(t["ci_95"]) == 2


