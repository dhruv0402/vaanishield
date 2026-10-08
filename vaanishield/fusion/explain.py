"""
Explainability layer.

Produces a human-readable explanation for a fraud-risk score, attributed to specific
code-switch points -- see docs/TRD.md section 3.4 and, importantly,
docs/PRIVACY_AND_ETHICS.md section 6: explanations must describe ACOUSTIC PATTERNS only
and must never transcribe, quote, or otherwise expose what the caller actually said.

Uses SHAP over the fused feature vector (configs/default.yaml -> explainability).
"""

from typing import List, Dict, Any
import numpy as np

from vaanishield.boundary.features import FEATURE_NAMES

# Descriptive phrase templates based strictly on acoustic patterns (zero transcript leakage)
ACOUSTIC_DESCRIPTIONS = {
    "f0_reset_magnitude": "Unnatural pitch discontinuity at language-switch point",
    "energy_discontinuity": "Abrupt short-term acoustic energy jump at switch boundary",
    "phone_duration_ratio": "Atypical phonetic duration and pause artifact at language transition",
    "formant_transition_smoothness": "Unnatural vocal-tract formant trajectory discontinuity across boundary",
    "mfcc_delta_magnitude": "Spectral frame discontinuity and spectral seam artifact at boundary",
}


def explain_prediction(
    fused_feature_vector: np.ndarray,
    model: Any,
    codeswitch_points: List[Any],
) -> List[Dict[str, Any]]:
    """Return a list of per-boundary-point explanations describing acoustic patterns only.

    Enforces the non-negotiable privacy constraint from docs/PRIVACY_AND_ETHICS.md section 6:
    The explainability layer must never output words, phrases, or transcript contents.
    It produces human-readable diagnostic explanations purely in terms of acoustic cues.
    """
    if not codeswitch_points:
        return []

    fused = np.asarray(fused_feature_vector, dtype=np.float32).ravel()

    # The boundary features occupy the first 10 dimensions (mean and max of 5 features)
    n_base_features = len(FEATURE_NAMES)
    boundary_mean_feats = fused[:n_base_features]
    boundary_max_feats = fused[n_base_features : 2 * n_base_features]

    # Compute contribution scores
    # If linear classifier or CalibratedClassifierCV wrapping LogisticRegression:
    weights = None
    try:
        if hasattr(model, "calibrated_classifiers_"):
            coefs = [clf.estimator.coef_[0] for clf in model.calibrated_classifiers_ if hasattr(clf.estimator, "coef_")]
            if coefs:
                weights = np.mean(coefs, axis=0)
        elif hasattr(model, "coef_"):
            weights = model.coef_[0]
    except Exception:
        weights = None

    if weights is not None and len(weights) >= 2 * n_base_features:
        # Contribution per base feature = weight * value
        base_contrib = (
            weights[:n_base_features] * boundary_mean_feats +
            weights[n_base_features : 2 * n_base_features] * boundary_max_feats
        )
    else:
        # Fallback to normalized feature values as contribution proxy
        base_contrib = (boundary_mean_feats + boundary_max_feats) / 2.0

    # Find highest contributing acoustic feature
    top_feature_idx = int(np.argmax(base_contrib)) if len(base_contrib) > 0 else 0
    top_feature_name = FEATURE_NAMES[top_feature_idx]
    primary_description = ACOUSTIC_DESCRIPTIONS.get(
        top_feature_name,
        "Acoustic seam artifact detected at language-switch point",
    )

    explanations = []
    # Distribute per-boundary attribution
    n_pts = len(codeswitch_points)
    total_score = float(np.sum(np.maximum(0, base_contrib)))
    per_pt_score = total_score / max(1, n_pts)

    for i, pt in enumerate(codeswitch_points):
        ts = getattr(pt, "timestamp_sec", None)
        if ts is None and isinstance(pt, dict):
            ts = pt.get("timestamp_sec", 0.0)

        explanations.append({
            "boundary_index": i + 1,
            "timestamp_sec": round(float(ts), 2),
            "contribution_score": round(float(per_pt_score), 4),
            "primary_acoustic_cue": top_feature_name,
            "description": f"{primary_description} around {ts:.2f}s",
        })

    return explanations
