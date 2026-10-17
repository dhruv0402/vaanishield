#!/usr/bin/env python3
import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, Any
import numpy as np
import yaml

from vaanishield.evaluation.ablation import run_ablation_suite
from vaanishield.evaluation.metrics import evaluate_by_breakdown
from vaanishield.utils.logging_utils import get_logger

logger = get_logger("run_ablation")


def run_ablation(
    config_path: str = "configs/default.yaml",
    scores_csv: str = "results/scores.csv",
    dataset_manifest: str = "data/processed/master_manifest.jsonl",
    output_dir: str = "results",
) -> Dict[str, Any]:
    """Run full three-way ablation suite.
    
    Reads master_manifest.jsonl for splits and runs ablation streams.
    Aligns metrics with results/scores.csv for consistency.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    with open(dataset_manifest, "r", encoding="utf-8") as f:
        all_records = [json.loads(line) for line in f]

    train_recs = [r for r in all_records if r.get("split") == "train"]
    val_recs = [r for r in all_records if r.get("split") == "val"]
    test_recs = [r for r in all_records if r.get("split") == "test"]
    heldout_recs = [r for r in all_records if r.get("split") == "heldout"]

    results = run_ablation_suite(
        train_records=train_recs,
        test_records=test_recs,
        heldout_records=heldout_recs,
        config=cfg,
        val_records=val_recs,
    )

    # If scores.csv exists, synchronize the fused model metric directly from evaluation_report.json
    eval_report_json = out_path / "evaluation_report.json"
    if eval_report_json.exists():
        with open(eval_report_json, "r", encoding="utf-8") as f:
            eval_data = json.load(f)
        if "models" in eval_data and "vaanishield_fusion" in eval_data["models"]:
            if "val" in eval_data["models"]["vaanishield_fusion"]:
                results["fused"]["val_eval"] = eval_data["models"]["vaanishield_fusion"]["val"]
            results["fused"]["test_eval"] = eval_data["models"]["vaanishield_fusion"]["test"]
            results["fused"]["heldout_generalization"] = eval_data["models"]["vaanishield_fusion"]["heldout_generalization"]

    # Save JSON report
    report_json = out_path / "ablation_report.json"
    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Save Markdown report
    report_md = out_path / "ablation_report.md"
    with open(report_md, "w", encoding="utf-8") as f:
        f.write("# VaaniShield Three-Way Ablation Study\n\n")
        f.write("Ablation evaluating individual and combined stream contributions across validation, in-domain test, and held-out sets.\n\n")
        if val_recs and results["fused"].get("val_eval"):
            f.write("## 1. Ablation Variant Performance (Validation Split)\n\n")
            f.write("| Stream / Variant | Validation EER | 95% Confidence Interval | Precision | Recall |\n")
            f.write("| :--- | :--- | :--- | :--- | :--- |\n")
            for v_key, v_name in [
                ("boundary_only", "Boundary-Local Stream Only"),
                ("speaker_null_only", "Speaker-Nulled Global Stream Only"),
                ("fused", "**Full VaaniShield Fused Model**"),
            ]:
                ov_v = results[v_key]["val_eval"]["overall"]
                f.write(f"| {v_name} | {ov_v['eer']:.4f} | [{ov_v['eer_ci'][0]:.4f}, {ov_v['eer_ci'][1]:.4f}] | {ov_v['precision']:.4f} | {ov_v['recall']:.4f} |\n")
            f.write("\n")

        f.write("## 2. Ablation Variant Performance (In-Domain Test Set)\n\n")
        f.write("| Stream / Variant | EER | 95% Confidence Interval | Precision | Recall |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- |\n")
        for v_key, v_name in [
            ("boundary_only", "Boundary-Local Stream Only"),
            ("speaker_null_only", "Speaker-Nulled Global Stream Only"),
            ("fused", "**Full VaaniShield Fused Model**"),
        ]:
            ov = results[v_key]["test_eval"]["overall"]
            f.write(f"| {v_name} | {ov['eer']:.4f} | [{ov['eer_ci'][0]:.4f}, {ov['eer_ci'][1]:.4f}] | {ov['precision']:.4f} | {ov['recall']:.4f} |\n")

        f.write("\n## 3. Generalization to Held-Out Synthesizer\n\n")
        f.write("| Stream / Variant | Held-Out Generalization EER | 95% Confidence Interval | Precision | Recall |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- |\n")
        for v_key, v_name in [
            ("boundary_only", "Boundary-Local Stream Only"),
            ("speaker_null_only", "Speaker-Nulled Global Stream Only"),
            ("fused", "**Full VaaniShield Fused Model**"),
        ]:
            ov_h = results[v_key]["heldout_generalization"]["overall"]
            f.write(f"| {v_name} | {ov_h['eer']:.4f} | [{ov_h['eer_ci'][0]:.4f}, {ov_h['eer_ci'][1]:.4f}] | {ov_h['precision']:.4f} | {ov_h['recall']:.4f} |\n")

    logger.info(f"Ablation study completed. Saved reports to {report_json} and {report_md}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run VaaniShield ablation suite")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--scores", default="results/scores.csv")
    parser.add_argument("--manifest", default="data/processed/master_manifest.jsonl")
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    run_ablation(args.config, args.scores, args.manifest, args.output_dir)
