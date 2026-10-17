"""
Ablation harness.

Runs the fusion model with (a) boundary-local stream only, (b) speaker-nulled global
stream only, and (c) both fused, so that each stream's independent contribution can be
reported -- this is a required deliverable per docs/PRD.md (Goals & Success Metrics) and
docs/TRD.md section 5, not optional polish.
"""

from typing import Dict


from typing import Dict, Any, List, Optional
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler

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
    val_records: Optional[List[dict]] = None,
) -> Dict[str, Any]:
    """Train and evaluate 3 model variants:
    1. Boundary-local stream only
    2. Speaker-nulled global stream only
    3. Full fused model

    Also evaluates on validation (`val`), in-domain test (`test`), and held-out synthesizer (`heldout`).
    """
    val_recs = val_records or []
    logger.info(
        f"Starting ablation suite: {len(train_records)} train, {len(val_recs)} val, "
        f"{len(test_records)} test, {len(heldout_records)} held-out synthesizer samples."
    )

    # 1. Fit SNAP Speaker Nulling Projector on train set
    projector = SpeakerNullingProjector(
        ssl_backbone=config.get("speaker_null", {}).get("ssl_backbone", "microsoft/wavlm-base-plus")
    )
    train_raw_embs = np.array([projector.embed(r["audio_path"]) for r in train_records], dtype=np.float32)
    projector.fit_speaker_subspace(train_raw_embs, n_components=4)

    train_nulled_embs = projector.project(train_raw_embs)
    val_nulled_embs = (
        projector.project(np.array([projector.embed(r["audio_path"]) for r in val_recs], dtype=np.float32))
        if val_recs
        else np.array([])
    )
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
    val_boundary_mat = extract_boundary_matrix(val_recs) if val_recs else np.array([])
    test_boundary_mat = extract_boundary_matrix(test_records)
    heldout_boundary_mat = extract_boundary_matrix(heldout_records)

    # Labels: 0 for bona_fide, 1 for spoof
    y_train = np.array([1 if r.get("label") == "spoof" else 0 for r in train_records], dtype=np.int32)
    y_val = np.array([1 if r.get("label") == "spoof" else 0 for r in val_recs], dtype=np.int32) if val_recs else np.array([])
    y_test = np.array([1 if r.get("label") == "spoof" else 0 for r in test_records], dtype=np.int32)
    y_heldout = np.array([1 if r.get("label") == "spoof" else 0 for r in heldout_records], dtype=np.int32)

    # Metadata breakdowns
    val_breakdowns = {
        "noise_condition": [r.get("noise_condition", "clean") for r in val_recs],
        "language_pair": [r.get("language_pair", "ta-en") for r in val_recs],
    }
    test_breakdowns = {
        "noise_condition": [r.get("noise_condition", "clean") for r in test_records],
        "language_pair": [r.get("language_pair", "hi-en") for r in test_records],
    }
    heldout_breakdowns = {
        "noise_condition": [r.get("noise_condition", "clean") for r in heldout_records],
    }

    from sklearn.model_selection import StratifiedKFold
    cv_split = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Variant 1: Boundary-only (StandardScaled + L2 regularized)
    scaler_b = StandardScaler().fit(train_boundary_mat)
    X_tr_b = scaler_b.transform(train_boundary_mat)
    X_val_b = scaler_b.transform(val_boundary_mat) if val_recs else np.array([])
    X_te_b = scaler_b.transform(test_boundary_mat)
    X_ho_b = scaler_b.transform(heldout_boundary_mat)

    clf_boundary = CalibratedClassifierCV(
        estimator=LogisticRegression(C=0.5, max_iter=500, class_weight="balanced", random_state=42),
        method="sigmoid",
        cv=cv_split,
    )
    clf_boundary.fit(X_tr_b, y_train)
    preds_val_boundary = clf_boundary.predict_proba(X_val_b)[:, 1] if val_recs else np.array([])
    preds_test_boundary = clf_boundary.predict_proba(X_te_b)[:, 1]
    preds_heldout_boundary = clf_boundary.predict_proba(X_ho_b)[:, 1]

    # Variant 2: Speaker-null only (StandardScaled + L2 regularized)
    scaler_sn = StandardScaler().fit(train_nulled_embs)
    X_tr_sn = scaler_sn.transform(train_nulled_embs)
    X_val_sn = scaler_sn.transform(val_nulled_embs) if val_recs else np.array([])
    X_te_sn = scaler_sn.transform(test_nulled_embs)
    X_ho_sn = scaler_sn.transform(heldout_nulled_embs)

    clf_sn = CalibratedClassifierCV(
        estimator=LogisticRegression(C=0.02, max_iter=500, class_weight="balanced", random_state=42),
        method="sigmoid",
        cv=cv_split,
    )
    clf_sn.fit(X_tr_sn, y_train)
    preds_val_sn = clf_sn.predict_proba(X_val_sn)[:, 1] if val_recs else np.array([])
    preds_test_sn = clf_sn.predict_proba(X_te_sn)[:, 1]
    preds_heldout_sn = clf_sn.predict_proba(X_ho_sn)[:, 1]

    # Variant 3: Fused (StandardScaled + L2 regularized)
    train_fused_raw = np.concatenate([train_boundary_mat, train_nulled_embs], axis=1)
    scaler_f = StandardScaler().fit(train_fused_raw)
    X_tr_f = scaler_f.transform(train_fused_raw)
    X_val_f = scaler_f.transform(np.concatenate([val_boundary_mat, val_nulled_embs], axis=1)) if val_recs else np.array([])
    X_te_f = scaler_f.transform(np.concatenate([test_boundary_mat, test_nulled_embs], axis=1))
    X_ho_f = scaler_f.transform(np.concatenate([heldout_boundary_mat, heldout_nulled_embs], axis=1))

    clf_fused = CalibratedClassifierCV(
        estimator=LogisticRegression(C=0.02, max_iter=500, class_weight="balanced", random_state=42),
        method="sigmoid",
        cv=cv_split,
    )
    clf_fused.fit(X_tr_f, y_train)
    preds_val_fused = clf_fused.predict_proba(X_val_f)[:, 1] if val_recs else np.array([])
    preds_test_fused = clf_fused.predict_proba(X_te_f)[:, 1]
    preds_heldout_fused = clf_fused.predict_proba(X_ho_f)[:, 1]

    results: Dict[str, Any] = {
        "boundary_only": {
            "val_eval": evaluate_by_breakdown(preds_val_boundary, y_val, val_breakdowns) if val_recs else {},
            "test_eval": evaluate_by_breakdown(preds_test_boundary, y_test, test_breakdowns),
            "heldout_generalization": evaluate_by_breakdown(preds_heldout_boundary, y_heldout, heldout_breakdowns),
        },
        "speaker_null_only": {
            "val_eval": evaluate_by_breakdown(preds_val_sn, y_val, val_breakdowns) if val_recs else {},
            "test_eval": evaluate_by_breakdown(preds_test_sn, y_test, test_breakdowns),
            "heldout_generalization": evaluate_by_breakdown(preds_heldout_sn, y_heldout, heldout_breakdowns),
        },
        "fused": {
            "val_eval": evaluate_by_breakdown(preds_val_fused, y_val, val_breakdowns) if val_recs else {},
            "test_eval": evaluate_by_breakdown(preds_test_fused, y_test, test_breakdowns),
            "heldout_generalization": evaluate_by_breakdown(preds_heldout_fused, y_heldout, heldout_breakdowns),
        },
    }

    logger.info("Ablation suite completed successfully.")
    return results

