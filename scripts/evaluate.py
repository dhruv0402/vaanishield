#!/usr/bin/env python3
import argparse
import csv
import json
import os
import pickle
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import yaml

from vaanishield.boundary.features import extract_boundary_features, aggregate_boundary_features
from vaanishield.boundary.localization import CodeSwitchPoint
from vaanishield.speaker_null.snap_projection import SpeakerNullingProjector
from vaanishield.baselines.baseline_a_generic import BaselineAGeneric
from vaanishield.baselines.baseline_b_indian_accent import BaselineBIndianAccent
from vaanishield.evaluation.metrics import evaluate_by_breakdown
from vaanishield.utils.logging_utils import get_logger

logger = get_logger("evaluate")


def generate_verdict(fused_eer: float, base_a_eer: float, base_b_eer: float, fused_ci: List[float]) -> str:
    """Generate verdict programmatically from numbers and confidence intervals."""
    min_baseline = min(base_a_eer, base_b_eer)
    ci_low, ci_high = fused_ci[0], fused_ci[1]

    if ci_high < min_baseline:
        return f"VaaniShield statistically outperforms all baselines (EER {fused_eer:.4f} vs best baseline {min_baseline:.4f}, p < 0.05)."
    elif ci_low > min_baseline:
        return f"VaaniShield demonstrates an EER of {fused_eer:.4f} compared to best baseline {min_baseline:.4f}; baseline performance is higher on this configuration."
    else:
        return f"VaaniShield achieves comparable performance to baselines (EER {fused_eer:.4f} [CI: {ci_low:.4f}, {ci_high:.4f}] vs best baseline {min_baseline:.4f}); differences are within statistical margin of error."


def evaluate_models(
    config_path: str = "configs/default.yaml",
    fusion_checkpoint: str = "checkpoints/fusion_checkpoint.pkl",
    baseline_a_checkpoint: str = "checkpoints/baseline_a_checkpoint.pkl",
    baseline_b_checkpoint: str = "checkpoints/baseline_b_checkpoint.pkl",
    dataset_manifest: str = "data/processed/master_manifest.jsonl",
    output_dir: str = "results",
) -> Dict[str, Any]:
    """Evaluate trained Fusion Model and both Baselines on test & held-out datasets.
    
    Exports results/scores.csv as the single source of truth for all metric calculations.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    with open(dataset_manifest, "r", encoding="utf-8") as f:
        all_records = [json.loads(line) for line in f]

    test_records = [r for r in all_records if r.get("split") == "test"]
    heldout_records = [r for r in all_records if r.get("split") == "heldout"]

    if not test_records:
        raise ValueError("No test records found in manifest")
    if not heldout_records:
        raise ValueError("No heldout records found in manifest")

    # Assert dual-class in both evaluation sets
    test_labels = set(r["label"] for r in test_records)
    heldout_labels = set(r["label"] for r in heldout_records)
    if not (("bona_fide" in test_labels or "bonafide" in test_labels) and "spoof" in test_labels):
        raise ValueError(f"Test set missing one of the classes: {test_labels}")
    if not (("bona_fide" in heldout_labels or "bonafide" in heldout_labels) and "spoof" in heldout_labels):
        raise ValueError(f"Heldout set missing one of the classes: {heldout_labels}")

    test_audio_paths = [r["audio_path"] for r in test_records]
    y_test = np.array([1 if r.get("label") == "spoof" else 0 for r in test_records], dtype=np.int32)
    test_speakers = np.array([r.get("speaker_id", "unknown") for r in test_records])
    test_breakdowns = {
        "noise_condition": [r.get("noise_condition", "clean") for r in test_records],
    }

    heldout_audio_paths = [r["audio_path"] for r in heldout_records]
    y_heldout = np.array([1 if r.get("label") == "spoof" else 0 for r in heldout_records], dtype=np.int32)
    heldout_speakers = np.array([r.get("speaker_id", "unknown") for r in heldout_records])
    heldout_breakdowns = {
        "noise_condition": [r.get("noise_condition", "clean") for r in heldout_records],
    }

    # Load models
    with open(fusion_checkpoint, "rb") as f:
        f_data = pickle.load(f)
        fusion_model = f_data["model"]
        projector: SpeakerNullingProjector = f_data["projector"]

    with open(baseline_a_checkpoint, "rb") as f:
        ba_model: BaselineAGeneric = pickle.load(f)["model"]

    with open(baseline_b_checkpoint, "rb") as f:
        bb_model: BaselineBIndianAccent = pickle.load(f)["model"]

    # Predict Baseline A
    preds_ba_test = ba_model.predict_proba(test_audio_paths)[:, 1]
    preds_ba_heldout = ba_model.predict_proba(heldout_audio_paths)[:, 1]

    # Predict Baseline B
    preds_bb_test = bb_model.predict_proba(test_audio_paths)[:, 1]
    preds_bb_heldout = bb_model.predict_proba(heldout_audio_paths)[:, 1]

    # Predict Fusion Model
    def extract_bmat(recs):
        mat = []
        for r in recs:
            pts = []
            for p in r.get("codeswitch_points", []):
                if isinstance(p, dict):
                    pts.append(CodeSwitchPoint(
                        p["timestamp_sec"], p["from_lang"], p["to_lang"],
                        p["window_start_sec"], p["window_end_sec"]
                    ))
                else:
                    pts.append(p)
            b_feats = extract_boundary_features(r["audio_path"], pts)
            agg = aggregate_boundary_features(b_feats, aggregation="mean_and_max")
            mat.append(agg)
        return np.array(mat, dtype=np.float32)

    test_bmat = extract_bmat(test_records)
    heldout_bmat = extract_bmat(heldout_records)

    test_sn = projector.project(
        np.array([projector.embed(p) for p in test_audio_paths], dtype=np.float32)
    )
    heldout_sn = projector.project(
        np.array([projector.embed(p) for p in heldout_audio_paths], dtype=np.float32)
    )

    preds_fusion_test = fusion_model.predict_proba(test_bmat, test_sn)[:, 1]
    preds_fusion_heldout = fusion_model.predict_proba(heldout_bmat, heldout_sn)[:, 1]

    # --- SINGLE SOURCE OF TRUTH: Export results/scores.csv ---
    scores_csv_path = out_path / "scores.csv"
    with open(scores_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["utterance_id", "model", "score", "label", "condition", "split", "speaker_id"])

        for i, r in enumerate(test_records):
            uid = r["utterance_id"]
            cond = r.get("noise_condition", "clean")
            spk = r.get("speaker_id", "unknown")
            lbl = y_test[i]
            writer.writerow([uid, "baseline_a", f"{preds_ba_test[i]:.6f}", lbl, cond, "test", spk])
            writer.writerow([uid, "baseline_b", f"{preds_bb_test[i]:.6f}", lbl, cond, "test", spk])
            writer.writerow([uid, "fusion", f"{preds_fusion_test[i]:.6f}", lbl, cond, "test", spk])

        for i, r in enumerate(heldout_records):
            uid = r["utterance_id"]
            cond = r.get("noise_condition", "clean")
            spk = r.get("speaker_id", "unknown")
            lbl = y_heldout[i]
            writer.writerow([uid, "baseline_a", f"{preds_ba_heldout[i]:.6f}", lbl, cond, "heldout", spk])
            writer.writerow([uid, "baseline_b", f"{preds_bb_heldout[i]:.6f}", lbl, cond, "heldout", spk])
            writer.writerow([uid, "fusion", f"{preds_fusion_heldout[i]:.6f}", lbl, cond, "heldout", spk])

    logger.info(f"Wrote scores single source of truth to {scores_csv_path}")

    # Evaluate by breakdown using cluster bootstrap
    eval_ba_test = evaluate_by_breakdown(preds_ba_test, y_test, test_breakdowns, clusters=test_speakers)
    eval_ba_heldout = evaluate_by_breakdown(preds_ba_heldout, y_heldout, heldout_breakdowns, clusters=heldout_speakers)

    eval_bb_test = evaluate_by_breakdown(preds_bb_test, y_test, test_breakdowns, clusters=test_speakers)
    eval_bb_heldout = evaluate_by_breakdown(preds_bb_heldout, y_heldout, heldout_breakdowns, clusters=heldout_speakers)

    eval_fusion_test = evaluate_by_breakdown(preds_fusion_test, y_test, test_breakdowns, clusters=test_speakers)
    eval_fusion_heldout = evaluate_by_breakdown(preds_fusion_heldout, y_heldout, heldout_breakdowns, clusters=heldout_speakers)

    verdict_test = generate_verdict(
        eval_fusion_test["overall"]["eer"],
        eval_ba_test["overall"]["eer"],
        eval_bb_test["overall"]["eer"],
        eval_fusion_test["overall"]["eer_ci"],
    )

    verdict_heldout = generate_verdict(
        eval_fusion_heldout["overall"]["eer"],
        eval_ba_heldout["overall"]["eer"],
        eval_bb_heldout["overall"]["eer"],
        eval_fusion_heldout["overall"]["eer_ci"],
    )

    report = {
        "dataset_summary": {
            "test_samples": len(test_records),
            "heldout_samples": len(heldout_records),
        },
        "verdicts": {
            "test_verdict": verdict_test,
            "heldout_verdict": verdict_heldout,
        },
        "models": {
            "baseline_a_generic": {
                "test": eval_ba_test,
                "heldout_generalization": eval_ba_heldout,
            },
            "baseline_b_indian_accent": {
                "test": eval_bb_test,
                "heldout_generalization": eval_bb_heldout,
            },
            "vaanishield_fusion": {
                "test": eval_fusion_test,
                "heldout_generalization": eval_fusion_heldout,
            },
        },
    }

    # Save JSON report
    report_json = out_path / "evaluation_report.json"
    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Save Markdown report
    report_md = out_path / "evaluation_report.md"
    with open(report_md, "w", encoding="utf-8") as f:
        f.write("# VaaniShield Evaluation & Baseline Comparison Report\n\n")
        f.write("## 1. Overall Performance Comparison (In-Domain Test Set)\n\n")
        f.write(f"> **Verdict**: {verdict_test}\n\n")
        f.write("| Model | EER (Equal Error Rate) | 95% Confidence Interval | Precision | Recall |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- |\n")
        for m_key, m_name in [
            ("baseline_a_generic", "Baseline A (Generic SSL)"),
            ("baseline_b_indian_accent", "Baseline B (Indian Accent Mel+LFCC)"),
            ("vaanishield_fusion", "**VaaniShield (Boundary + SNAP Fusion)**"),
        ]:
            ov = report["models"][m_key]["test"]["overall"]
            f.write(f"| {m_name} | {ov['eer']:.4f} | [{ov['eer_ci'][0]:.4f}, {ov['eer_ci'][1]:.4f}] | {ov['precision']:.4f} | {ov['recall']:.4f} |\n")

        f.write("\n## 2. Generalization to Unseen Synthesizer (Held-Out Set)\n\n")
        f.write(f"> **Verdict**: {verdict_heldout}\n\n")
        f.write("| Model | Generalization EER | 95% Confidence Interval | Precision | Recall |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- |\n")
        for m_key, m_name in [
            ("baseline_a_generic", "Baseline A (Generic SSL)"),
            ("baseline_b_indian_accent", "Baseline B (Indian Accent Mel+LFCC)"),
            ("vaanishield_fusion", "**VaaniShield (Boundary + SNAP Fusion)**"),
        ]:
            ov_h = report["models"][m_key]["heldout_generalization"]["overall"]
            f.write(f"| {m_name} | {ov_h['eer']:.4f} | [{ov_h['eer_ci'][0]:.4f}, {ov_h['eer_ci'][1]:.4f}] | {ov_h['precision']:.4f} | {ov_h['recall']:.4f} |\n")

        f.write("\n## 3. Performance Breakdown by Telecom Noise Condition (VaaniShield)\n\n")
        f.write("| Noise Condition | EER | 95% Confidence Interval | Precision | Recall |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- |\n")
        noise_breakdowns = eval_fusion_test["breakdowns"].get("noise_condition", {})
        for cond, metrics in noise_breakdowns.items():
            f.write(f"| {cond} | {metrics['eer']:.4f} | [{metrics['eer_ci'][0]:.4f}, {metrics['eer_ci'][1]:.4f}] | {metrics['precision']:.4f} | {metrics['recall']:.4f} |\n")

    logger.info(f"Evaluation completed. Reports saved to {report_json} and {report_md}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate VaaniShield models against baselines")
    parser.add_argument("--config", default="configs/default.yaml", help="Config file")
    parser.add_argument("--fusion-ckpt", default="checkpoints/fusion_checkpoint.pkl")
    parser.add_argument("--base-a-ckpt", default="checkpoints/baseline_a_checkpoint.pkl")
    parser.add_argument("--base-b-ckpt", default="checkpoints/baseline_b_checkpoint.pkl")
    parser.add_argument("--manifest", default="data/processed/master_manifest.jsonl")
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    evaluate_models(
        args.config,
        args.fusion_ckpt,
        args.base_a_ckpt,
        args.base_b_ckpt,
        args.manifest,
        args.output_dir,
    )
