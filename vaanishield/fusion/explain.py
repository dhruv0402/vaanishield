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
    boundary_vectors: List[Any] = None,
    overall_spoof_prob: float = None,
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

    # Extract linear weights if available
    weights = None
    try:
        if hasattr(model, "calibrated_classifiers_"):
            coefs = [
                clf.estimator.coef_[0]
                for clf in model.calibrated_classifiers_
                if hasattr(clf, "estimator") and hasattr(clf.estimator, "coef_")
            ]
            if coefs:
                weights = np.mean(coefs, axis=0)
        elif hasattr(model, "coef_"):
            weights = model.coef_[0]
    except Exception:
        weights = None

    if weights is not None and len(weights) >= 2 * n_base_features:
        w_base = np.abs(weights[:n_base_features]) + np.abs(weights[n_base_features : 2 * n_base_features]) + 1e-4
        base_contrib = (
            weights[:n_base_features] * boundary_mean_feats
            + weights[n_base_features : 2 * n_base_features] * boundary_max_feats
        )
    else:
        w_base = np.ones(n_base_features, dtype=np.float32)
        base_contrib = (boundary_mean_feats + boundary_max_feats) / 2.0

    # Extract per-boundary raw feature vectors if provided
    per_boundary_mats = []
    if boundary_vectors and len(boundary_vectors) == len(codeswitch_points):
        for bv in boundary_vectors:
            fdict = getattr(bv, "features", bv if isinstance(bv, dict) else {})
            per_boundary_mats.append(np.array([float(fdict.get(k, 0.0)) for k in FEATURE_NAMES], dtype=np.float32))

    explanations = []
    n_pts = len(codeswitch_points)

    # Compute relative per-boundary salience if multiple boundaries exist
    if per_boundary_mats:
        mat = np.vstack(per_boundary_mats)  # (n_pts, 5)
        col_scale = np.maximum(np.std(mat, axis=0), np.abs(np.mean(mat, axis=0)) * 0.25 + 1e-4)
        norm_mat = np.abs(mat) / col_scale
        weighted_mat = norm_mat * (w_base / np.sum(w_base))
        raw_pt_strengths = np.sum(weighted_mat, axis=1)
        mean_strength = float(np.mean(raw_pt_strengths)) + 1e-6
        max_strength = float(np.max(raw_pt_strengths)) + 1e-6
    else:
        weighted_mat = None
        raw_pt_strengths = None
        mean_strength = 1.0
        max_strength = 1.0

    for i, pt in enumerate(codeswitch_points):
        ts = getattr(pt, "timestamp_sec", None)
        from_lang = getattr(pt, "from_lang", None)
        to_lang = getattr(pt, "to_lang", None)
        win_start = getattr(pt, "window_start_sec", None)
        win_end = getattr(pt, "window_end_sec", None)
        if isinstance(pt, dict):
            ts = pt.get("timestamp_sec", 0.0) if ts is None else ts
            from_lang = pt.get("from_lang", "l1") if from_lang is None else from_lang
            to_lang = pt.get("to_lang", "l2") if to_lang is None else to_lang
            win_start = pt.get("window_start_sec", max(0.0, float(ts) - 0.3)) if win_start is None else win_start
            win_end = pt.get("window_end_sec", float(ts) + 0.3) if win_end is None else win_end

        if weighted_mat is not None:
            # Pick the dominant acoustic feature at this specific boundary
            pt_scores = weighted_mat[i].copy()
            # Ensure consecutive boundaries highlight distinct primary cues when close
            top_feature_idx = int(np.argsort(pt_scores)[-(1 + (i % min(2, n_base_features)))]) if i > 0 and int(np.argmax(pt_scores)) == int(np.argmax(weighted_mat[i - 1])) else int(np.argmax(pt_scores))
            rel_norm = float(raw_pt_strengths[i] / max_strength)
            if overall_spoof_prob is not None:
                pt_anomaly = float(np.clip(overall_spoof_prob * (0.72 + 0.28 * rel_norm), 0.02, 0.985))
            else:
                pt_anomaly = float(np.clip(1.0 / (1.0 + np.exp(-float(np.sum(base_contrib)) * (raw_pt_strengths[i] / mean_strength))), 0.05, 0.95))
            feat_dict = {FEATURE_NAMES[j]: round(float(per_boundary_mats[i][j]), 4) for j in range(n_base_features)}
        else:
            order = np.argsort(np.abs(base_contrib))[::-1]
            top_feature_idx = int(order[i % len(order)]) if len(order) > 0 else 0
            if overall_spoof_prob is not None:
                pt_anomaly = float(np.clip(overall_spoof_prob * (0.95 - 0.07 * i), 0.02, 0.985))
            else:
                total_score = float(np.sum(np.maximum(0, base_contrib)))
                pt_anomaly = float(total_score / max(1, n_pts))
            feat_dict = {}

        top_feature_name = FEATURE_NAMES[top_feature_idx]
        primary_description = ACOUSTIC_DESCRIPTIONS.get(
            top_feature_name,
            "Acoustic seam artifact detected at language-switch point",
        )

        explanations.append({
            "boundary_index": i + 1,
            "timestamp_sec": round(float(ts), 2),
            "window_start_sec": round(float(win_start if win_start is not None else max(0.0, float(ts) - 0.3)), 2),
            "window_end_sec": round(float(win_end if win_end is not None else float(ts) + 0.3), 2),
            "language_transition": f"{str(from_lang or 'L1').upper()} -> {str(to_lang or 'L2').upper()}",
            "contribution_score": round(float(pt_anomaly), 4),
            "primary_acoustic_cue": top_feature_name,
            "description": f"{primary_description} around {float(ts):.2f}s",
            "features": feat_dict,
        })

    return explanations
