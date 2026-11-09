"""Regression tests for honest EER estimates and cluster bootstrap intervals."""

import json

import numpy as np
import pytest

from vaanishield.evaluation.metrics import (
    bootstrap_confidence_interval,
    compute_eer,
    evaluate_by_breakdown,
)


@pytest.mark.parametrize("genuine,spoof", [([], []), ([], [0.8]), ([0.2], [])])
def test_eer_is_undefined_without_both_classes(genuine, spoof):
    assert np.isnan(compute_eer(genuine, spoof))


@pytest.mark.parametrize(
    "genuine,spoof,expected",
    [
        ([0.1, 0.2], [0.8, 0.9], 0.0),
        ([0.8, 0.9], [0.1, 0.2], 1.0),
        ([0.5, 0.5], [0.5, 0.5], 0.5),
        # The ROC crosses FPR = FNR on the vertical segment at FPR = 1/3.
        ([0.1, 0.2, 0.4], [0.3, 0.5], 1.0 / 3.0),
        # A tied-score segment must be interpolated, not rounded to an endpoint.
        ([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 0.4),
    ],
)
def test_eer_uses_interpolated_roc_crossing(genuine, spoof, expected):
    assert compute_eer(genuine, spoof) == pytest.approx(expected)


@pytest.mark.parametrize("bad_scores", [[np.nan], [np.inf], [-np.inf], [[0.2]], [0.2j]])
def test_eer_rejects_nonfinite_nonreal_or_nonvector_scores(bad_scores):
    with pytest.raises(ValueError):
        compute_eer(bad_scores, [0.8])
    with pytest.raises(ValueError):
        compute_eer([0.2], bad_scores)


@pytest.mark.parametrize("scores,labels", [([], []), ([0.2, 0.3], [0, 0]), ([0.8], [1])])
def test_bootstrap_is_undefined_without_both_classes(scores, labels):
    assert all(np.isnan(value) for value in bootstrap_confidence_interval(scores, labels))


@pytest.mark.parametrize(
    "scores,labels",
    [
        ([0.2, 0.8], [0]),
        ([0.2], [0, 1]),
        ([0.2, 0.8], [0, 1.5]),
        ([0.2, 0.8], [0, 2]),
        ([0.2, 0.8], [0, -1]),
        ([0.2, 0.8], [0, np.nan]),
        ([0.2, 0.8], [0, np.inf]),
        ([0.2, np.inf], [0, 1]),
        ([np.nan, 0.8], [0, 1]),
        ([[0.2, 0.8]], [0, 1]),
        ([0.2, 0.8], [[0, 1]]),
        ([0.2, 0.8], [0, 1 + 1j]),
    ],
)
def test_bootstrap_rejects_invalid_aligned_input(scores, labels):
    with pytest.raises(ValueError):
        bootstrap_confidence_interval(scores, labels, n_resamples=10)


@pytest.mark.parametrize("clusters", [["one"], ["a", "b", "c"], [["a", "b"]], [None, "b"], [np.nan, "b"]])
def test_bootstrap_rejects_invalid_cluster_membership(clusters):
    with pytest.raises(ValueError):
        bootstrap_confidence_interval([0.2, 0.8], [0, 1], clusters=clusters, n_resamples=10)


@pytest.mark.parametrize("n_resamples", [0, -1, 1.5, True])
def test_bootstrap_rejects_invalid_resample_counts(n_resamples):
    with pytest.raises(ValueError):
        bootstrap_confidence_interval([0.2, 0.8], [0, 1], n_resamples=n_resamples)


@pytest.mark.parametrize("ci", [0.0, 1.0, -0.1, 1.1, np.nan, np.inf, True])
def test_bootstrap_rejects_invalid_confidence_levels(ci):
    with pytest.raises(ValueError):
        bootstrap_confidence_interval([0.2, 0.8], [0, 1], n_resamples=10, ci=ci)


def test_cluster_bootstrap_preserves_all_observations_in_each_draw():
    # Both independent clusters have exactly the same empirical distribution.
    # A whole-cluster bootstrap cannot change it. Resampling inside each cluster can.
    result = bootstrap_confidence_interval(
        [0.1, 0.8, 0.2, 0.9] * 2,
        [0, 0, 1, 1] * 2,
        clusters=["speaker-a"] * 4 + ["speaker-b"] * 4,
        n_resamples=100,
    )
    assert result == pytest.approx((0.5, 0.5, 0.5))


def test_single_cluster_does_not_fall_back_to_independent_sample_bootstrap():
    point, low, high = bootstrap_confidence_interval(
        [0.1, 0.8, 0.2, 0.9], [0, 0, 1, 1], clusters=["one-speaker"] * 4,
        n_resamples=100,
    )
    assert point == pytest.approx(0.5)
    assert np.isnan(low)
    assert np.isnan(high)


def test_insufficient_valid_resamples_do_not_become_point_intervals():
    point, low, high = bootstrap_confidence_interval([0.1, 0.8, 0.2, 0.9], [0, 0, 1, 1], n_resamples=5)
    assert point == pytest.approx(0.5)
    assert np.isnan(low)
    assert np.isnan(high)


def test_bootstrap_is_reproducible_for_a_seed():
    scores = [0.1, 0.8, 0.2, 0.9, 0.3, 0.7]
    labels = [0, 0, 1, 1, 0, 1]
    first = bootstrap_confidence_interval(scores, labels, seed=10, n_resamples=100)
    second = bootstrap_confidence_interval(scores, labels, seed=10, n_resamples=100)
    assert first == second
    assert 0 <= first[1] <= first[2] <= 1


def test_empty_breakdown_reports_undefined_metrics_as_json_null():
    result = evaluate_by_breakdown([], [], {"condition": []})
    assert result["overall"] == {
        "count": 0, "eer": None, "eer_ci": [None, None], "precision": None, "recall": None,
    }
    assert result["breakdowns"] == {"condition": {}}
    json.dumps(result, allow_nan=False)


def test_single_class_breakdown_preserves_count_without_claiming_perfect_eer():
    result = evaluate_by_breakdown([0.2, 0.8], [0, 1], {"condition": ["genuine", "spoof"]}, n_resamples=10)
    for metrics in result["breakdowns"]["condition"].values():
        assert metrics["count"] == 1
        assert metrics["eer"] is None
        assert metrics["eer_ci"] == [None, None]
    json.dumps(result, allow_nan=False)


def test_breakdown_accepts_cluster_lists_and_keeps_unavailable_intervals_json_safe():
    result = evaluate_by_breakdown(
        [0.1, 0.8, 0.2, 0.9] * 2, [0, 0, 1, 1] * 2,
        {"condition": ["clean"] * 4 + ["noisy"] * 4},
        clusters=["speaker-a"] * 4 + ["speaker-b"] * 4, n_resamples=100,
    )
    assert result["overall"]["eer_ci"] == [0.5, 0.5]
    for metrics in result["breakdowns"]["condition"].values():
        assert metrics["eer"] == 0.5
        assert metrics["eer_ci"] == [None, None]
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize(
    "predictions,labels,breakdowns,clusters",
    [
        ([0.2, 0.8], [0, 0.9], {}, None),
        ([0.2, np.inf], [0, 0], {}, None),
        ([0.2, 0.8], [0, 1], {"condition": ["clean"]}, None),
        ([0.2, 0.8], [0, 1], {"condition": [["clean", "noisy"]]}, None),
        ([0.2, 0.8], [0, 0], {}, ["short"]),
    ],
)
def test_breakdown_validates_inputs_even_when_eer_is_undefined(predictions, labels, breakdowns, clusters):
    with pytest.raises(ValueError):
        evaluate_by_breakdown(predictions, labels, breakdowns, clusters=clusters, n_resamples=10)


def test_repeated_cluster_draws_keep_their_full_multiplicity():
    # Three paired clusters, two correctly ranked and one reversed. Each draw's
    # EER is the fraction of reversed clusters, hence {0, 1/3, 2/3, 1}. The
    # central 80% interval is [0, 2/3]; deduplicating cluster draws gives [0, 1/2].
    point, low, high = bootstrap_confidence_interval(
        [0.0, 1.0, 0.0, 1.0, 1.0, 0.0], [0, 1, 0, 1, 0, 1],
        clusters=["a", "a", "b", "b", "c", "c"],
        n_resamples=1000, ci=0.8, seed=42,
    )
    assert (point, low, high) == pytest.approx((1.0 / 3.0, 0.0, 2.0 / 3.0))


def test_distinct_numeric_and_string_cluster_ids_are_not_merged():
    point, low, high = bootstrap_confidence_interval(
        [0.1, 0.8, 0.2, 0.9] * 2, [0, 0, 1, 1] * 2,
        clusters=[1] * 4 + ["1"] * 4, n_resamples=100,
    )
    assert (point, low, high) == pytest.approx((0.5, 0.5, 0.5))


def test_breakdown_rejects_colliding_output_keys_instead_of_losing_a_group():
    with pytest.raises(ValueError):
        evaluate_by_breakdown([0.2, 0.8], [0, 1], {"condition": [1, "1"]}, n_resamples=10)


@pytest.mark.parametrize(
    "fused,base_a,base_b,bounds",
    [
        (None, 0.4, 0.5, [None, None]),
        (0.1, None, 0.5, [0.05, 0.2]),
        (0.1, 0.4, np.nan, [0.05, 0.2]),
        (0.1, 0.4, 0.5, [None, None]),
        (0.1, 0.4, 0.5, [np.nan, np.nan]),
        (0.1, 0.4, 0.5, [0.3, 0.2]),
        (0.1, 0.4, 0.5, [0.05, np.inf]),
    ],
)
def test_verdict_does_not_claim_superiority_when_metrics_are_unavailable(fused, base_a, base_b, bounds):
    from scripts.evaluate import generate_verdict

    verdict = generate_verdict(fused, base_a, base_b, bounds).lower()
    assert "unavailable" in verdict
    assert "statistically outperforms" not in verdict
    assert "p <" not in verdict


def test_fused_confidence_interval_alone_does_not_prove_baseline_significance():
    from scripts.evaluate import generate_verdict

    verdict = generate_verdict(0.1, 0.4, 0.5, [0.05, 0.2]).lower()
    assert "0.1000" in verdict
    assert "0.4000" in verdict
    assert "paired" in verdict
    assert "statistically outperforms" not in verdict
    assert "p <" not in verdict


def _write_small_evaluation_inputs(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("{}\n", encoding="utf-8")
    records = [
        {
            "split": split, "audio_path": str(index), "utterance_id": f"{split}-{index}",
            "source_utterance_id": f"{split}-only-cluster", "speaker_id": f"{split}-speaker",
            "label": "bona_fide" if index < 2 else "spoof",
            "noise_condition": "clean" if index < 2 else "noisy", "language_pair": "hi-en",
        }
        for split in ("val", "test", "heldout") for index in range(4)
    ]
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    return config, manifest, records


def test_evaluation_reports_render_undefined_metrics_without_crashing(tmp_path, monkeypatch):
    from scripts import evaluate

    config, manifest, _ = _write_small_evaluation_inputs(tmp_path)
    checkpoint = tmp_path / "checkpoint.pkl"
    checkpoint.write_bytes(b"test model placeholder")

    class Predictions:
        def predict_proba(self, *inputs):
            probabilities = np.array([0.1, 0.8, 0.2, 0.9])
            return np.column_stack([1 - probabilities, probabilities])

    class Projector:
        def embed(self, path):
            return np.array([0.0, 1.0])

        def project(self, values):
            return values

    # Isolate unavailable external model/audio artifacts; exercise real metrics,
    # verdict generation, JSON serialization, and every Markdown table section.
    model = Predictions()
    checkpoint_contents = iter([
        {"model": model, "projector": Projector()}, {"model": model}, {"model": model},
    ])
    monkeypatch.setattr(evaluate.pickle, "load", lambda file: next(checkpoint_contents))
    monkeypatch.setattr(evaluate, "extract_boundary_features", lambda *args: np.array([0.0]))
    monkeypatch.setattr(evaluate, "aggregate_boundary_features", lambda values, **kwargs: values)
    output = tmp_path / "evaluation"
    report = evaluate.evaluate_models(
        str(config), str(checkpoint), str(checkpoint), str(checkpoint), str(manifest), str(output),
    )
    for metrics in report["models"].values():
        for split in ("val", "test", "heldout_generalization"):
            assert metrics[split]["overall"]["eer"] == 0.5
            assert metrics[split]["overall"]["eer_ci"] == [None, None]
    text = (output / "evaluation_report.md").read_text(encoding="utf-8")
    assert "N/A" in text
    assert "nan" not in text.lower()
    assert "p <" not in text
    json.dumps(json.loads((output / "evaluation_report.json").read_text()), allow_nan=False)


def test_ablation_reports_render_missing_intervals(tmp_path, monkeypatch):
    from scripts import run_ablation

    config, manifest, _ = _write_small_evaluation_inputs(tmp_path)
    unavailable = evaluate_by_breakdown([], [], {})
    single_cluster = evaluate_by_breakdown([0.2, 0.8], [0, 1], {}, clusters=["one", "one"])
    results = {
        variant: {"val_eval": unavailable, "test_eval": single_cluster, "heldout_generalization": unavailable}
        for variant in ("boundary_only", "speaker_null_only", "fused")
    }
    # Training is separate from the report renderer exercised here.
    monkeypatch.setattr(run_ablation, "run_ablation_suite", lambda **kwargs: results)
    output = tmp_path / "ablation"
    run_ablation.run_ablation(str(config), dataset_manifest=str(manifest), output_dir=str(output))
    text = (output / "ablation_report.md").read_text(encoding="utf-8")
    assert "N/A" in text
    assert "nan" not in text.lower()
    assert "Early Feature Fusion (Boundary + SNAP)" in text
    assert "Full VaaniShield Fused Model" not in text
    json.dumps(json.loads((output / "ablation_report.json").read_text()), allow_nan=False)


def test_ablation_uses_cluster_intervals_for_every_variant_and_split(monkeypatch):
    from vaanishield.evaluation import ablation

    class Projector:
        def __init__(self, **kwargs):
            pass

        def embed(self, path):
            return np.array([float(path), 1.0])

        def fit_speaker_subspace(self, values, **kwargs):
            pass

        def project(self, values):
            return values

    def records(count, split):
        return [
            {
                "audio_path": str(index), "speaker_id": f"{split}-speaker",
                "source_utterance_id": f"{split}-only-cluster",
                "label": "spoof" if index % 2 else "bona_fide",
            }
            for index in range(count)
        ]

    # Only audio extraction is substituted; fitting, calibration, and evaluation
    # run for all three variants against a split with one independent cluster.
    monkeypatch.setattr(ablation, "SpeakerNullingProjector", Projector)
    monkeypatch.setattr(ablation, "extract_boundary_features", lambda path, _: np.array([float(path)]))
    monkeypatch.setattr(ablation, "aggregate_boundary_features", lambda values, **kwargs: values)
    results = ablation.run_ablation_suite(
        records(20, "train"), records(4, "test"), records(4, "heldout"), {},
        val_records=records(4, "val"),
    )
    assert results["fused"]["architecture"] == "early_feature_fusion"
    for variant in results.values():
        for split in ("val_eval", "test_eval", "heldout_generalization"):
            assert np.isfinite(variant[split]["overall"]["eer"])
            assert variant[split]["overall"]["eer_ci"] == [None, None]


def test_ablation_does_not_replace_fresh_metrics_with_an_existing_evaluation_report(tmp_path, monkeypatch):
    from scripts import run_ablation

    config, manifest, _ = _write_small_evaluation_inputs(tmp_path)
    fresh = evaluate_by_breakdown([0.1, 0.8, 0.2, 0.9], [0, 0, 1, 1], {}, clusters=["one"] * 4)
    results = {
        variant: {split: fresh for split in ("val_eval", "test_eval", "heldout_generalization")}
        for variant in ("boundary_only", "speaker_null_only", "fused")
    }
    output = tmp_path / "ablation"
    output.mkdir()
    stale = {"overall": {"count": 500, "eer": 0.0, "eer_ci": [0.0, 0.0], "precision": 1.0, "recall": 1.0}}
    historical_report = output / "evaluation_report.json"
    historical_contents = json.dumps({
        "models": {"vaanishield_fusion": {split: stale for split in ("val", "test", "heldout_generalization")}}
    })
    historical_report.write_text(historical_contents, encoding="utf-8")
    monkeypatch.setattr(run_ablation, "run_ablation_suite", lambda **kwargs: results)
    report = run_ablation.run_ablation(str(config), dataset_manifest=str(manifest), output_dir=str(output))
    written = json.loads((output / "ablation_report.json").read_text())
    for source in (report, written):
        for split in ("val_eval", "test_eval", "heldout_generalization"):
            assert source["fused"][split]["overall"]["count"] == 4
            assert source["fused"][split]["overall"]["eer"] == 0.5
            assert source["fused"][split]["overall"]["eer_ci"] == [None, None]
    assert historical_report.read_text(encoding="utf-8") == historical_contents
