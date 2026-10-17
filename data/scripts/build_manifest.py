#!/usr/bin/env python3
"""
Build Unified Master Dataset Manifest for VaaniShield.

Adheres strictly to docs/DATA_SCHEMA.md:
- Resolves raw recordings, train clones (XTTS v2), and heldout clones.
- Links to codeswitch scripts, token sequences, and aligned boundary timestamps.
- Groups splits strictly by speaker_id and source_utterance_id to eliminate data leakage.
- Assigns:
    * Train: spk_1119, spk_2131, spk_ta_01, spk_kn_01 (200 records)
    * Validation (val): spk_ta_02 (50 records, dual-class bona-fide + spoof)
    * In-domain Test (test): spk_3701, spk_kn_02 (100 records, dual-class bona-fide + spoof)
    * Held-out Synthesizer (heldout): spk_5945 (50 records, dual-class bona-fide + heldout spoof)
- Derives deterministic per-record seed from utterance_id.
- Extracts duration, sample rate, and normalized loudness.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Any
import soundfile as sf
import numpy as np

from vaanishield.utils.audio_io import load_audio
from vaanishield.utils.logging_utils import get_logger

logger = get_logger("build_manifest")


def derive_seed(utterance_id: str) -> int:
    """Derive deterministic non-negative 31-bit integer seed from utterance_id."""
    return int(hashlib.sha256(utterance_id.encode("utf-8")).hexdigest(), 16) % (2**31)


def load_consented_speakers(manifest_path: str = "data/consent_forms/consented_speakers.csv") -> Dict[str, dict]:
    """Load consented speakers and verify signed consent forms exist."""
    path = Path(manifest_path)
    if not path.exists():
        raise PermissionError(f"Consent registry missing: {manifest_path}")

    consented = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            spk_id = row.get("speaker_id", "").strip()
            form_id = row.get("form_id", "").strip()
            signed_status = row.get("consent_signed", "").strip().lower()
            if spk_id and signed_status in ("true", "1", "yes"):
                form_file = Path(f"data/consent_forms/signed/{form_id}.md")
                if not form_file.exists():
                    raise PermissionError(f"Missing signed consent form for speaker {spk_id}: {form_file}")
                consented[spk_id] = row

    if len(consented) < 4:
        raise ValueError(f"Fewer than 4 consented speakers found ({len(consented)})")
    return consented


def determine_split(speaker_id: str, is_heldout_system: bool) -> str:
    """Strict 4-way speaker-isolated partition:
    - spk_1119, spk_2131, spk_ta_01, spk_kn_01 -> train
    - spk_ta_02 -> val (validation split for calibration & threshold selection)
    - spk_3701, spk_kn_02 -> test (in-domain evaluation split)
    - spk_5945 -> heldout (contains bona_fide and heldout clones)
    """
    if speaker_id in ("spk_1119", "spk_2131", "spk_ta_01", "spk_kn_01"):
        return "train"
    elif speaker_id == "spk_ta_02":
        return "val"
    elif speaker_id in ("spk_3701", "spk_kn_02"):
        return "test"
    elif speaker_id == "spk_5945":
        return "heldout"
    else:
        raise ValueError(f"Unknown speaker ID for split assignment: {speaker_id}")


def build_manifest(
    raw_manifest_path: str = "data/raw/raw_manifest.jsonl",
    clean_manifest_path: str = "data/processed/master_clean_manifest.jsonl",
    augmented_manifest_path: str = "data/processed/augmented_dataset_manifest.jsonl",
    output_path: str = "data/processed/master_manifest.jsonl",
    consent_path: str = "data/consent_forms/consented_speakers.csv",
) -> List[Dict[str, Any]]:
    """Build standardized master manifest joining bona-fide, cloned, and augmented records."""
    consented = load_consented_speakers(consent_path)
    logger.info(f"Verified {len(consented)} consented speakers.")

    with open(clean_manifest_path, "r", encoding="utf-8") as f:
        clean_records = [json.loads(line) for line in f]
    clean_map = {r["utterance_id"]: r for r in clean_records}

    with open(augmented_manifest_path, "r", encoding="utf-8") as f:
        augmented_records = [json.loads(line) for line in f]

    # Map raw text to sentence_ids if available
    sentence_map = {}
    scripts_file = Path("data/processed/codeswitch_scripts.jsonl")
    if scripts_file.exists():
        with open(scripts_file, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                sentence_map[rec["text"]] = rec["sentence_id"]

    master_records = []
    logger.info(f"Processing {len(augmented_records)} augmented records into master manifest...")

    for aug in augmented_records:
        utt_id = aug["utterance_id"]
        # Find base clean record
        base_rec = None
        if utt_id in clean_map:
            base_rec = clean_map[utt_id]
        else:
            base_id = utt_id.rsplit("_", 1)[0]
            if base_id in clean_map:
                base_rec = clean_map[base_id]
            else:
                raise KeyError(f"Cannot trace base clean record for {utt_id}")

        speaker_id = aug["speaker_id"]
        label = "bona_fide" if aug.get("label") == "bonafide" or aug.get("label") == "bona_fide" else "spoof"
        cloning_sys = aug.get("cloning_system", base_rec.get("cloning_system", "none"))
        is_heldout_sys = bool(aug.get("is_heldout_system", base_rec.get("is_heldout_system", False)))

        # Assign source_utterance_id
        if label == "bona_fide":
            # For bona fide, the source utterance ID is the clean recording utterance ID
            source_utt_id = base_rec["utterance_id"]
        else:
            source_utt_id = base_rec.get("source_utterance_id")
            if not source_utt_id:
                # Remove cloning suffix to find original source bona-fide ID
                b_id = base_rec["utterance_id"]
                for suff in ["_xtts_heldout", "_xtts", "_bark", "_openvoice"]:
                    if b_id.endswith(suff):
                        source_utt_id = b_id[:-len(suff)]
                        break
                if not source_utt_id:
                    source_utt_id = b_id

        # Determine split
        split = determine_split(speaker_id, is_heldout_sys)

        # In train, val, and test splits, we exclude heldout systems from in-domain pools
        if split in ("train", "val", "test") and is_heldout_sys:
            continue
        # In heldout split (spk_5945), we evaluate heldout synthesizer and bona-fide
        if split == "heldout" and (label == "spoof" and not is_heldout_sys):
            continue

        audio_path = aug["audio_path"]
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file missing: {audio_path}")

        info = sf.info(audio_path)
        duration_sec = round(float(info.duration), 4)
        sample_rate_hz = int(info.samplerate)

        if duration_sec < 0.5:
            raise ValueError(f"Utterance {utt_id} shorter than 0.5s: {duration_sec}s")

        # Measured loudness
        audio_arr, _ = load_audio(audio_path)
        rms = float(np.sqrt(np.mean(audio_arr ** 2)))
        if rms < 0.01:
            raise ValueError(f"Utterance {utt_id} is near-silent: RMS = {rms}")

        text = aug.get("text", base_rec.get("text", ""))
        sent_id = sentence_map.get(text, "sent_unknown")
        cs_points = aug.get("codeswitch_points", base_rec.get("codeswitch_points", []))
        lang_pair = aug.get("language_pair", base_rec.get("language_pair", "hi-en"))

        master_row = {
            "utterance_id": utt_id,
            "source_utterance_id": source_utt_id,
            "sentence_id": sent_id,
            "speaker_id": speaker_id,
            "language_pair": lang_pair,
            "audio_path": audio_path,
            "label": label,
            "cloning_system": cloning_sys,
            "is_heldout_system": is_heldout_sys,
            "noise_condition": aug.get("noise_condition", "clean"),
            "split": split,
            "seed": derive_seed(utt_id),
            "duration_sec": duration_sec,
            "sample_rate_hz": sample_rate_hz,
            "loudness_lufs_or_rms": round(rms, 4),
            "codeswitch_points": cs_points,
            "text": text,
        }
        master_records.append(master_row)

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        for r in master_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    logger.info(f"Built master manifest with {len(master_records)} records -> {output_path}")
    return master_records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build unified master manifest for VaaniShield")
    parser.add_argument("--output", default="data/processed/master_manifest.jsonl", help="Output path")
    args = parser.parse_args()

    build_manifest(output_path=args.output)
