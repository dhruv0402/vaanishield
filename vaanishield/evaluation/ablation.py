"""
Ablation harness.

Runs the fusion model with (a) boundary-local stream only, (b) speaker-nulled global
stream only, and (c) both fused, so that each stream's independent contribution can be
reported -- this is a required deliverable per docs/PRD.md (Goals & Success Metrics) and
docs/TRD.md section 5, not optional polish.
"""

from typing import Dict


from typing import Dict, Any, List
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

from vaanishield.boundary.features import extract_boundary_features, aggregate_boundary_features
from vaanishield.boundary.localization import CodeSwitchPoint
from vaanishield.speaker_null.snap_projection import SpeakerNullingProjector
from vaanishield.evaluation.metrics import evaluate_by_breakdown
from vaanishield.utils.logging_utils import get_logger

logger = get_logger("ablation")


def run_ablation_suite(
    train_records: List[dict],
    test_records: List[dict],
    heldout_records: List[dict],
    config: dict,
) -> Dict[str, Any]:
    """Train and evaluate 3 model variants:
    1. Boundary-local stream only
    2. Speaker-nulled global stream only
    3. Full fused model

    Also evaluates generalization against the held-out voice-cloning system (OpenVoice).
    """
    logger.info(
        f"Starting ablation suite: {len(train_records)} train, {len(test_records)} test, "
        f"{len(heldout_records)} held-out synthesizer samples."
    )

    # 1. Fit SNAP Speaker Nulling Projector on train set
    projector = SpeakerNullingProjector(
        ssl_backbone=config.get("speaker_null", {}).get("ssl_backbone", "microsoft/wavlm-base-plus")
    )
    train_raw_embs = np.array([projector.embed(r["audio_path"]) for r in train_records], dtype=np.float32)
    projector.fit_speaker_subspace(train_raw_embs, n_components=4)

    train_nulled_embs = projector.project(train_raw_embs)
    test_nulled_embs = projector.project(
        np.array([projector.embed(r["audio_path"]) for r in test_records], dtype=np.float32)
    )
    heldout_nulled_embs = projector.project(
        np.array([projector.embed(r["audio_path"]) for r in heldout_records], dtype=np.float32)
    )

    # 2. Extract Boundary features
    def parse_cs_points(rec):
        pts = []
        for p in rec.get("codeswitch_points", []):
            if isinstance(p, dict):
                pts.append(CodeSwitchPoint(
                    p["timestamp_sec"], p["from_lang"], p["to_lang"],
                    p["window_start_sec"], p["window_end_sec"]
                ))
            else:
                pts.append(p)
        return pts

    def extract_boundary_matrix(records):
        mat = []
        for r in records:
            pts = parse_cs_points(r)
            b_feats = extract_boundary_features(r["audio_path"], pts)
            agg = aggregate_boundary_features(b_feats, aggregation="mean_and_max")
            mat.append(agg)
        return np.array(mat, dtype=np.float32)

    train_boundary_mat = extract_boundary_matrix(train_records)
    test_boundary_mat = extract_boundary_matrix(test_records)
    heldout_boundary_mat = extract_boundary_matrix(heldout_records)

    # Labels: 0 for bona_fide, 1 for spoof
    y_train = np.array([1 if r.get("label") == "spoof" else 0 for r in train_records], dtype=np.int32)
    y_test = np.array([1 if r.get("label") == "spoof" else 0 for r in test_records], dtype=np.int32)
    y_heldout = np.array([1 if r.get("label") == "spoof" else 0 for r in heldout_records], dtype=np.int32)

    # Metadata breakdowns for test set
    test_breakdowns = {
        "noise_condition": [r.get("noise_condition", "clean") for r in test_records],
        "language_pair": [r.get("language_pair", "hi-en") for r in test_records],
    }
    heldout_breakdowns = {
        "noise_condition": [r.get("noise_condition", "clean") for r in heldout_records],
    }

    # Variant 1: Boundary-only
    clf_boundary = CalibratedClassifierCV(
        estimator=LogisticRegression(max_iter=500, class_weight="balanced", random_state=42),
        method="isotonic",
        cv=2,
    )
    clf_boundary.fit(train_boundary_mat, y_train)
    preds_test_boundary = clf_boundary.predict_proba(test_boundary_mat)[:, 1]
    preds_heldout_boundary = clf_boundary.predict_proba(heldout_boundary_mat)[:, 1]

    # Variant 2: Speaker-null only
    clf_sn = CalibratedClassifierCV(
        estimator=LogisticRegression(max_iter=500, class_weight="balanced", random_state=42),
        method="isotonic",
        cv=2,
    )
    clf_sn.fit(train_nulled_embs, y_train)
    preds_test_sn = clf_sn.predict_proba(test_nulled_embs)[:, 1]
    preds_heldout_sn = clf_sn.predict_proba(heldout_nulled_embs)[:, 1]

    # Variant 3: Fused
    train_fused = np.concatenate([train_boundary_mat, train_nulled_embs], axis=1)
    test_fused = np.concatenate([test_boundary_mat, test_nulled_embs], axis=1)
    heldout_fused = np.concatenate([heldout_boundary_mat, heldout_nulled_embs], axis=1)

    clf_fused = CalibratedClassifierCV(
        estimator=LogisticRegression(max_iter=500, class_weight="balanced", random_state=42),
        method="isotonic",
        cv=2,
    )
    clf_fused.fit(train_fused, y_train)
    preds_test_fused = clf_fused.predict_proba(test_fused)[:, 1]
    preds_heldout_fused = clf_fused.predict_proba(heldout_fused)[:, 1]

    results: Dict[str, Any] = {
        "boundary_only": {
            "test_eval": evaluate_by_breakdown(preds_test_boundary, y_test, test_breakdowns),
            "heldout_generalization": evaluate_by_breakdown(preds_heldout_boundary, y_heldout, heldout_breakdowns),
        },
        "speaker_null_only": {
            "test_eval": evaluate_by_breakdown(preds_test_sn, y_test, test_breakdowns),
            "heldout_generalization": evaluate_by_breakdown(preds_heldout_sn, y_heldout, heldout_breakdowns),
        },
        "fused": {
            "test_eval": evaluate_by_breakdown(preds_test_fused, y_test, test_breakdowns),
            "heldout_generalization": evaluate_by_breakdown(preds_heldout_fused, y_heldout, heldout_breakdowns),
        },
    }

    logger.info("Ablation suite completed successfully.")
    return results

