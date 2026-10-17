"""
Evaluation metrics: EER, precision/recall, bootstrap confidence intervals.

All reported results in this project must include a baseline comparison and a
confidence interval -- a single aggregate accuracy number is explicitly insufficient
per docs/PRD.md (Non-Functional Requirements) and docs/TRD.md section 5.
"""

from typing import Dict, Tuple, Any, List, Optional
import numpy as np
from sklearn.metrics import precision_score, recall_score, roc_curve


def compute_eer(genuine_scores: np.ndarray, spoof_scores: np.ndarray) -> float:
    """Compute Equal Error Rate (EER) between genuine (bona_fide) and spoof scores.

    In spoof detection conventions:
    - genuine scores represent bona-fide likeness (or lower spoof score)
    - spoof scores represent higher spoof likelihood.
    """
    g_scores = np.asarray(genuine_scores, dtype=np.float64).ravel()
    s_scores = np.asarray(spoof_scores, dtype=np.float64).ravel()

    if len(g_scores) == 0 or len(s_scores) == 0:
        return 0.0

    y_true = np.concatenate([np.zeros_like(g_scores), np.ones_like(s_scores)])
    scores = np.concatenate([g_scores, s_scores])

    # Compute ROC curve (FPR = FAR, TPR = 1 - FRR)
    fpr, tpr, thresholds = roc_curve(y_true, scores, pos_label=1)
    fnr = 1.0 - tpr

    # Find the threshold where FAR and FRR are closest (Equal Error Rate)
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    return float(np.clip(eer, 0.0, 1.0))


def bootstrap_confidence_interval(
    scores: np.ndarray,
    labels: np.ndarray,
    clusters: Optional[np.ndarray] = None,
    n_resamples: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Compute (point_estimate, ci_lower, ci_upper) via cluster or sample bootstrap of EER.

    Resamples the scores by speaker cluster (if provided) or by sample index,
    recomputes EER for each resample, and returns empirical percentile bounds.
    """
    sc = np.asarray(scores, dtype=np.float64).ravel()
    lb = np.asarray(labels, dtype=np.int32).ravel()

    if len(sc) == 0 or len(lb) == 0:
        return 0.0, 0.0, 0.0

    gen_mask = (lb == 0)
    spf_mask = (lb == 1)
    if not np.any(gen_mask) or not np.any(spf_mask):
        return 0.0, 0.0, 0.0

    point_estimate = compute_eer(sc[gen_mask], sc[spf_mask])

    rng = np.random.RandomState(seed)
    boot_eers = []

    if clusters is not None:
        cl = np.asarray(clusters)
        unique_clusters = np.unique(cl)
        n_clusters = len(unique_clusters)
        if n_clusters > 1:
            for _ in range(n_resamples):
                sampled_clusters = rng.choice(unique_clusters, size=n_clusters, replace=True)
                sample_indices = []
                for c in sampled_clusters:
                    c_idx = np.where(cl == c)[0]
                    sample_indices.extend(rng.choice(c_idx, size=len(c_idx), replace=True))
                sample_indices = np.array(sample_indices, dtype=int)

                b_sc = sc[sample_indices]
                b_lb = lb[sample_indices]
                b_gen = b_sc[b_lb == 0]
                b_spf = b_sc[b_lb == 1]
                if len(b_gen) > 0 and len(b_spf) > 0:
                    boot_eers.append(compute_eer(b_gen, b_spf))
        else:
            # Fall back to standard sample bootstrap
            n_samples = len(sc)
            for _ in range(n_resamples):
                idx = rng.choice(n_samples, size=n_samples, replace=True)
                b_sc = sc[idx]
                b_lb = lb[idx]
                b_gen = b_sc[b_lb == 0]
                b_spf = b_sc[b_lb == 1]
                if len(b_gen) > 0 and len(b_spf) > 0:
                    boot_eers.append(compute_eer(b_gen, b_spf))
    else:
        n_samples = len(sc)
        for _ in range(n_resamples):
            idx = rng.choice(n_samples, size=n_samples, replace=True)
            b_sc = sc[idx]
            b_lb = lb[idx]
            b_gen = b_sc[b_lb == 0]
            b_spf = b_sc[b_lb == 1]
            if len(b_gen) > 0 and len(b_spf) > 0:
                boot_eers.append(compute_eer(b_gen, b_spf))

    if len(boot_eers) >= 10:
        alpha = (1.0 - ci) / 2.0
        ci_lower = float(np.percentile(boot_eers, alpha * 100))
        ci_upper = float(np.percentile(boot_eers, (1.0 - alpha) * 100))
    else:
        ci_lower, ci_upper = point_estimate, point_estimate

    return point_estimate, ci_lower, ci_upper


def evaluate_by_breakdown(
    predictions: np.ndarray,
    labels: np.ndarray,
    breakdown_keys: Dict[str, List[Any]],
    clusters: Optional[np.ndarray] = None,
    n_resamples: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Compute EER, precision, recall, and bootstrap CIs broken down by conditions.

    Computes:
    - Global aggregate metrics
    - Disaggregated metrics for each condition (e.g. language_pair x noise_condition).
    """
    preds = np.asarray(predictions, dtype=np.float64).ravel()
    y_true = np.asarray(labels, dtype=np.int32).ravel()

    def calc_metrics(sub_preds, sub_true, sub_clusters=None):
        if len(sub_true) == 0:
            return {"count": 0, "eer": 0.0, "eer_ci": [0.0, 0.0], "precision": 0.0, "recall": 0.0}

        gen_scores = sub_preds[sub_true == 0]
        spf_scores = sub_preds[sub_true == 1]

        if len(gen_scores) == 0 or len(spf_scores) == 0:
            return {
                "count": int(len(sub_true)),
                "eer": 0.0,
                "eer_ci": [0.0, 0.0],
                "precision": float(precision_score(sub_true, (sub_preds >= 0.5).astype(int), zero_division=0)),
                "recall": float(recall_score(sub_true, (sub_preds >= 0.5).astype(int), zero_division=0)),
            }

        eer, ci_low, ci_high = bootstrap_confidence_interval(
            sub_preds, sub_true, clusters=sub_clusters, n_resamples=n_resamples, seed=seed
        )
        prec = float(precision_score(sub_true, (sub_preds >= 0.5).astype(int), zero_division=0))
        rec = float(recall_score(sub_true, (sub_preds >= 0.5).astype(int), zero_division=0))

        return {
            "count": int(len(sub_true)),
            "eer": round(eer, 4),
            "eer_ci": [round(ci_low, 4), round(ci_high, 4)],
            "precision": round(prec, 4),
            "recall": round(rec, 4),
        }

    results: Dict[str, Any] = {
        "overall": calc_metrics(preds, y_true, clusters),
        "breakdowns": {},
    }

    # Iterate through breakdown keys (e.g. noise_condition, language_pair)
    for key_name, key_values in breakdown_keys.items():
        results["breakdowns"][key_name] = {}
        unique_vals = set(key_values)
        for val in unique_vals:
            mask = np.array([v == val for v in key_values])
            if np.any(mask):
                sub_cl = clusters[mask] if clusters is not None else None
                results["breakdowns"][key_name][str(val)] = calc_metrics(preds[mask], y_true[mask], sub_cl)

    return results
