import argparse
import json
import sys
from pathlib import Path
from typing import List
import yaml

from vaanishield.boundary.localization import find_codeswitch_points, CodeSwitchPoint
from vaanishield.utils.logging_utils import get_logger

logger = get_logger("align_codeswitch_points")


def run_batch_alignment(
    input_manifest: str,
    config_path: str = "configs/default.yaml",
    output_manifest: str = "data/processed/aligned_manifest.jsonl",
) -> List[dict]:
    """Run code-switch boundary localization on all utterances in manifest.
    
    Validates required keys up front (audio_path, text).
    Reports errors separately from legitimate zero-boundary cases.
    Fails (exits non-zero) if error rate exceeds 2%.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    b_cfg = cfg.get("boundary_localization", {})
    lid_model = b_cfg.get("language_id_model", "fasttext-lid176")
    alignment_backend = b_cfg.get("alignment_backend", "whisper")
    window_before = b_cfg.get("window_ms_before", 300)
    window_after = b_cfg.get("window_ms_after", 300)

    in_path = Path(input_manifest)
    if not in_path.exists():
        raise FileNotFoundError(f"Input manifest not found: {input_manifest}")

    with open(in_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    out_records = []
    error_count = 0
    zero_boundary_count = 0

    for rec in records:
        if "audio_path" not in rec or "text" not in rec:
            raise KeyError(f"Manifest record missing mandatory 'audio_path' or 'text': {rec}")

        audio_path = rec["audio_path"]
        tokens = rec.get("tokens", rec["text"].split())

        points: List[CodeSwitchPoint] = []
        try:
            points = find_codeswitch_points(
                audio_path=audio_path,
                transcript_tokens=tokens,
                lid_model=lid_model,
                alignment_backend=alignment_backend,
                window_ms_before=window_before,
                window_ms_after=window_after,
            )
        except Exception as e:
            error_count += 1
            logger.error(f"Alignment failure for {audio_path}: {e}")
            points = []

        pts_dicts = [
            {
                "timestamp_sec": p.timestamp_sec,
                "from_lang": p.from_lang,
                "to_lang": p.to_lang,
                "window_start_sec": p.window_start_sec,
                "window_end_sec": p.window_end_sec,
            }
            for p in points
        ]

        if not pts_dicts and error_count == 0:
            zero_boundary_count += 1
            logger.warning(
                f"Zero code-switch points found for utterance {rec.get('utterance_id', audio_path)}: "
                f"'{rec.get('text')}' (requires manual review if expected to be code-switched)."
            )

        updated_rec = dict(rec)
        updated_rec["codeswitch_points"] = pts_dicts
        updated_rec["num_detected_boundaries"] = len(pts_dicts)
        out_records.append(updated_rec)

    total_records = len(records)
    error_rate = error_count / total_records if total_records > 0 else 0.0

    logger.info(
        f"Completed batch alignment for {total_records} utterances -> {output_manifest}. "
        f"Failures/Errors: {error_count} ({error_rate:.1%}), Zero-boundary: {zero_boundary_count}."
    )

    if error_rate > 0.02:
        logger.error(f"Alignment failure rate {error_rate:.1%} exceeds strict threshold of 2.0%!")
        sys.exit(1)

    out_p = Path(output_manifest)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        for r in out_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return out_records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch code-switch boundary alignment")
    parser.add_argument("--input-manifest", default="data/raw/raw_manifest.jsonl")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output-manifest", default="data/processed/aligned_manifest.jsonl")
    args = parser.parse_args()

    run_batch_alignment(args.input_manifest, args.config, args.output_manifest)
