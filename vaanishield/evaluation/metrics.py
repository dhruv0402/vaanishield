"""
Evaluation metrics: EER, precision/recall, bootstrap confidence intervals.

All reported results in this project must include a baseline comparison and a
confidence interval -- a single aggregate accuracy number is explicitly insufficient
per docs/PRD.md (Non-Functional Requirements) and docs/TRD.md section 5.
"""

from numbers import Integral, Real
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import precision_score, recall_score, roc_curve


def _finite_vector(values: np.ndarray, name: str) -> np.ndarray:
    """Validate before conversion so shapes or complex values cannot be lost."""
    array = np.asarray(values)
    if array.ndim != 1 or array.dtype.kind not in "biuf":
        raise ValueError(f"{name} must be a one-dimensional array of real numbers")
    array = array.astype(np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _validate_scores_labels(scores, labels) -> Tuple[np.ndarray, np.ndarray]:
    sc = _finite_vector(scores, "scores")
    lb = _finite_vector(labels, "labels")
    if len(sc) != len(lb):
        raise ValueError("scores and labels must have the same length")
    if not np.all((lb == 0) | (lb == 1)):
        raise ValueError("labels must contain only 0 (bona fide) and 1 (spoof)")
    return sc, lb.astype(np.int32)


def _metadata_vector(values, name: str, count: int) -> np.ndarray:
    # Object dtype preserves mixed identifiers rather than coercing 1 and "1"
    # into the same cluster. Each observation needs a known, scalar identifier.
    array = np.asarray(values, dtype=object)
    if array.ndim != 1 or len(array) != count:
        raise ValueError(f"{name} must be one-dimensional and match the score count")
    for value in array:
        try:
            hash(value)
            missing = value is None or bool(value != value)
        except (TypeError, ValueError):
            raise ValueError(f"{name} must contain nonmissing, hashable scalar values") from None
        if missing or (isinstance(value, Real) and not np.isfinite(value)):
            raise ValueError(f"{name} must contain nonmissing, finite identifiers")
    return array


def _validate_bootstrap_options(n_resamples: int, ci: float) -> None:
    if isinstance(n_resamples, (bool, np.bool_)) or not isinstance(n_resamples, Integral) or n_resamples <= 0:
        raise ValueError("n_resamples must be a positive integer")
    if isinstance(ci, (bool, np.bool_)) or not isinstance(ci, Real) or not np.isfinite(ci) or not 0 < ci < 1:
        raise ValueError("ci must be a finite confidence level strictly between 0 and 1")


def compute_eer(genuine_scores: np.ndarray, spoof_scores: np.ndarray) -> float:
    """Compute EER at the linearly interpolated ROC crossing of FPR and FNR.

    Both arrays contain the same kind of score: higher values mean more likely
    spoof, including for the bona-fide observations in ``genuine_scores``.
    Returns NaN when either class is absent. Nonfinite, nonreal, or non-vector
    scores raise ValueError instead of being silently dropped or flattened.
    """
    g_scores = _finite_vector(genuine_scores, "genuine_scores")
    s_scores = _finite_vector(spoof_scores, "spoof_scores")
    if len(g_scores) == 0 or len(s_scores) == 0:
        return float("nan")

    y_true = np.concatenate([np.zeros(len(g_scores)), np.ones(len(s_scores))])
    scores = np.concatenate([g_scores, s_scores])
    fpr, tpr, _ = roc_curve(y_true, scores, pos_label=1, drop_intermediate=False)
    difference = fpr - (1.0 - tpr)
    right = int(np.searchsorted(difference, 0.0))
    if difference[right] == 0.0:
        return float(fpr[right])

    left = right - 1
    fraction = -difference[left] / (difference[right] - difference[left])
    return float(fpr[left] + fraction * (fpr[right] - fpr[left]))


def bootstrap_confidence_interval(
    scores: np.ndarray,
    labels: np.ndarray,
    clusters: Optional[np.ndarray] = None,
    n_resamples: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Return (EER, percentile lower bound, percentile upper bound).

    With clusters, sample the original number of clusters with replacement and
    retain every observation in each draw, including repeated cluster draws.
    Without clusters, sample observations with replacement (an IID assumption).
    Resamples lacking either class are discarded; bounds are conditional on
    valid resamples and require at least ten of them. Use many more for reporting.

    A missing class makes all three values NaN. A single supplied cluster or
    fewer than ten valid resamples leaves the point estimate intact but returns
    NaN bounds: neither case silently falls back to a different sampling model
    or a zero-width interval. Invalid inputs/options raise ValueError.
    """
    sc, lb = _validate_scores_labels(scores, labels)
    _validate_bootstrap_options(n_resamples, ci)
    cl = _metadata_vector(clusters, "clusters", len(sc)) if clusters is not None else None
    point_estimate = compute_eer(sc[lb == 0], sc[lb == 1])
    undefined = (point_estimate, float("nan"), float("nan"))
    if np.isnan(point_estimate):
        return undefined

    cluster_indices = None
    if cl is not None:
        grouped: Dict[Any, List[int]] = {}
        for index, cluster in enumerate(cl):
            grouped.setdefault(cluster, []).append(index)
        if len(grouped) < 2:
            return undefined
        cluster_indices = [np.asarray(indices, dtype=int) for indices in grouped.values()]

    rng = np.random.RandomState(seed)
    boot_eers = []
    for _ in range(n_resamples):
        if cluster_indices is None:
            indices = rng.choice(len(sc), size=len(sc), replace=True)
        else:
            draws = rng.choice(len(cluster_indices), size=len(cluster_indices), replace=True)
            indices = np.concatenate([cluster_indices[draw] for draw in draws])
        sample_scores, sample_labels = sc[indices], lb[indices]
        eer = compute_eer(sample_scores[sample_labels == 0], sample_scores[sample_labels == 1])
        if np.isfinite(eer):
            boot_eers.append(eer)

    if len(boot_eers) < 10:
        return undefined
    alpha = (1.0 - ci) / 2.0
    ci_lower, ci_upper = np.percentile(boot_eers, [alpha * 100, (1.0 - alpha) * 100])
    return point_estimate, float(ci_lower), float(ci_upper)


def evaluate_by_breakdown(
    predictions: np.ndarray,
    labels: np.ndarray,
    breakdown_keys: Dict[str, List[Any]],
    clusters: Optional[np.ndarray] = None,
    n_resamples: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Compute aggregate and per-condition EER, precision, recall, and EER CIs.

    All metadata must align with the one-dimensional prediction/label arrays.
    Undefined EER or CI values are JSON-safe None, never perfect scores. Empty
    datasets also have undefined precision/recall; otherwise those two metrics
    use the existing 0.5 threshold and zero_division=0 convention.
    """
    preds, y_true = _validate_scores_labels(predictions, labels)
    _validate_bootstrap_options(n_resamples, 0.95)
    cl = _metadata_vector(clusters, "clusters", len(preds)) if clusters is not None else None
    breakdowns = {
        name: _metadata_vector(values, f"breakdown {name!r}", len(preds))
        for name, values in breakdown_keys.items()
    }
    for name, values in breakdowns.items():
        unique_values = dict.fromkeys(values)
        if len({str(value) for value in unique_values}) != len(unique_values):
            raise ValueError(f"breakdown {name!r} contains values with identical output keys")

    def rounded_or_none(value):
        return round(float(value), 4) if np.isfinite(value) else None

    def calc_metrics(sub_preds, sub_true, sub_clusters=None):
        if len(sub_true) == 0:
            return {"count": 0, "eer": None, "eer_ci": [None, None], "precision": None, "recall": None}

        eer, ci_low, ci_high = bootstrap_confidence_interval(
            sub_preds, sub_true, clusters=sub_clusters, n_resamples=n_resamples, seed=seed
        )
        predicted_labels = (sub_preds >= 0.5).astype(int)
        prec = precision_score(sub_true, predicted_labels, zero_division=0)
        rec = recall_score(sub_true, predicted_labels, zero_division=0)
        return {
            "count": int(len(sub_true)),
            "eer": rounded_or_none(eer),
            "eer_ci": [rounded_or_none(ci_low), rounded_or_none(ci_high)],
            "precision": rounded_or_none(prec),
            "recall": rounded_or_none(rec),
        }

    results: Dict[str, Any] = {
        "overall": calc_metrics(preds, y_true, cl),
        "breakdowns": {},
    }
    for key_name, key_values in breakdowns.items():
        results["breakdowns"][key_name] = {}
        for value in dict.fromkeys(key_values):
            mask = key_values == value
            sub_cl = cl[mask] if cl is not None else None
            results["breakdowns"][key_name][str(value)] = calc_metrics(preds[mask], y_true[mask], sub_cl)
    return results
