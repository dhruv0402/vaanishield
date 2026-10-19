#!/usr/bin/env python3
import argparse
import json
import os
import pickle
from pathlib import Path
import numpy as np
import yaml

from vaanishield.boundary.features import extract_boundary_features, aggregate_boundary_features
from vaanishield.boundary.localization import CodeSwitchPoint
from vaanishield.speaker_null.snap_projection import SpeakerNullingProjector
from vaanishield.fusion.classifier import FusionClassifier
from vaanishield.baselines.baseline_a_generic import BaselineAGeneric
from vaanishield.baselines.baseline_b_indian_accent import BaselineBIndianAccent
from vaanishield.utils.logging_utils import get_logger

logger = get_logger("train")


def train(
    config_path: str = "configs/default.yaml",
    model_type: str = "fusion",
    dataset_manifest: str = "data/processed/master_manifest.jsonl",
) -> str:
    """Train the requested model (fusion, baseline_a, baseline_b) using the train split of master_manifest.jsonl."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    checkpoints_dir = Path(cfg.get("paths", {}).get("checkpoints_dir", "checkpoints"))
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    with open(dataset_manifest, "r", encoding="utf-8") as f:
        all_records = [json.loads(line) for line in f]

    # Select records belonging to the "train" and "val" splits
    train_records = [r for r in all_records if r.get("split") == "train"]
    val_records = [r for r in all_records if r.get("split") == "val"]
    logger.info(
        f"Loaded {len(train_records)} training records (train split) and {len(val_records)} validation records (val split)."
    )

    audio_paths = [r["audio_path"] for r in train_records]
    labels = np.array([1 if r.get("label") == "spoof" else 0 for r in train_records], dtype=np.int32)

    saved_checkpoint_path = str(checkpoints_dir / f"{model_type}_checkpoint.pkl")

    if model_type == "baseline_a":
        model = BaselineAGeneric(
            ssl_backbone=cfg.get("speaker_null", {}).get("ssl_backbone", "microsoft/wavlm-base-plus")
        )
        model.fit(audio_paths, labels)
        with open(saved_checkpoint_path, "wb") as f:
            pickle.dump({"model": model, "config": cfg, "model_type": model_type}, f)

    elif model_type == "baseline_b":
        model = BaselineBIndianAccent()
        model.fit(audio_paths, labels)
        with open(saved_checkpoint_path, "wb") as f:
            pickle.dump({"model": model, "config": cfg, "model_type": model_type}, f)

    elif model_type == "fusion":
        # 1. Fit SNAP Speaker Nulling Projector strictly on train split
        projector = SpeakerNullingProjector(
            ssl_backbone=cfg.get("speaker_null", {}).get("ssl_backbone", "microsoft/wavlm-base-plus")
        )
        speaker_ids = [r["speaker_id"] for r in train_records]
        raw_embs = np.array([projector.embed(p) for p in audio_paths], dtype=np.float32)
        projector.fit_speaker_subspace(raw_embs, speaker_ids=speaker_ids, n_components=4)
        nulled_embs = projector.project(raw_embs)

        # 2. Extract boundary features
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

        def build_boundary_mat(recs):
            mat = []
            for r in recs:
                pts = parse_cs_points(r)
                b_feats = extract_boundary_features(r["audio_path"], pts)
                agg = aggregate_boundary_features(
                    b_feats,
                    aggregation=cfg.get("boundary_features", {}).get("aggregation", "mean_and_max"),
                )
                mat.append(agg)
            return np.array(mat, dtype=np.float32)

        boundary_mat = build_boundary_mat(train_records)

        val_bmat, val_sn, val_labels = None, None, None
        if val_records:
            val_audio_paths = [r["audio_path"] for r in val_records]
            val_labels = np.array([1 if r.get("label") == "spoof" else 0 for r in val_records], dtype=np.int32)
            val_raw_embs = np.array([projector.embed(p) for p in val_audio_paths], dtype=np.float32)
            val_sn = projector.project(val_raw_embs)
            val_bmat = build_boundary_mat(val_records)

        # 3. Fit calibrated FusionClassifier using train split and tune/calibrate on val split
        fusion_cfg = cfg.get("fusion", {})
        model = FusionClassifier(
            classifier_type=fusion_cfg.get("classifier", "logistic_regression"),
            mlp_hidden_dim=fusion_cfg.get("mlp_hidden_dim", 64),
            calibration=fusion_cfg.get("calibration", "isotonic"),
        )
        model.fit(
            boundary_mat,
            nulled_embs,
            labels,
            val_boundary_features=val_bmat,
            val_speaker_nulled_embeddings=val_sn,
            val_labels=val_labels,
        )

        with open(saved_checkpoint_path, "wb") as f:
            pickle.dump({
                "model": model,
                "projector": projector,
                "config": cfg,
                "model_type": model_type,
            }, f)

        # Write SHA-256 integrity sidecar
        import hashlib
        with open(saved_checkpoint_path, "rb") as f:
            sha256 = hashlib.sha256(f.read()).hexdigest()
        with open(Path(saved_checkpoint_path).with_suffix(".sha256"), "w") as f:
            f.write(f"{sha256}  {Path(saved_checkpoint_path).name}\n")

        # Export zero-pickle container
        npz_target = Path("checkpoints/onnx/model_parameters.npz")
        if npz_target.parent.exists():
            FusionClassifier.save_parameters_npz(model, projector, str(npz_target))

    else:
        raise ValueError(f"Unknown model_type: {model_type}. Choose from 'fusion', 'baseline_a', 'baseline_b'")

    logger.info(f"Successfully trained {model_type} and saved checkpoint to {saved_checkpoint_path}")
    return saved_checkpoint_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train VaaniShield models")
    parser.add_argument("--config", default="configs/default.yaml", help="Config file")
    parser.add_argument("--model", default="fusion", choices=["fusion", "baseline_a", "baseline_b"], help="Model to train")
    parser.add_argument("--manifest", default="data/processed/master_manifest.jsonl", help="Dataset manifest")
    args = parser.parse_args()

    train(args.config, args.model, args.manifest)
