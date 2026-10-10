import json
import pytest
from pathlib import Path


def test_split_speaker_and_utterance_isolation():
    """Assert no speaker_id and no source_utterance_id appears in multiple splits."""
    manifest_path = Path("data/processed/master_manifest.jsonl")
    if not manifest_path.exists():
        pytest.skip(f"Master manifest not on disk (local audio artifact): {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    splits = {}
    for r in records:
        sp = r["split"]
        splits.setdefault(sp, []).append(r)

    assert "train" in splits, "train split missing"
    assert "test" in splits, "test split missing"
    assert "heldout" in splits, "heldout split missing"

    train_speakers = set(r["speaker_id"] for r in splits["train"])
    test_speakers = set(r["speaker_id"] for r in splits["test"])
    heldout_speakers = set(r["speaker_id"] for r in splits["heldout"])

    # 1. No speaker overlap across splits
    assert not (train_speakers & test_speakers), f"Speaker leak train vs test: {train_speakers & test_speakers}"
    assert not (train_speakers & heldout_speakers), f"Speaker leak train vs heldout: {train_speakers & heldout_speakers}"
    assert not (test_speakers & heldout_speakers), f"Speaker leak test vs heldout: {test_speakers & heldout_speakers}"

    # 2. No source_utterance_id overlap across splits
    train_src_utts = set(r["source_utterance_id"] for r in splits["train"] if r.get("source_utterance_id"))
    test_src_utts = set(r["source_utterance_id"] for r in splits["test"] if r.get("source_utterance_id"))
    heldout_src_utts = set(r["source_utterance_id"] for r in splits["heldout"] if r.get("source_utterance_id"))

    assert not (train_src_utts & test_src_utts), f"Source utterance leak train vs test: {train_src_utts & test_src_utts}"
    assert not (train_src_utts & heldout_src_utts), f"Source utterance leak train vs heldout: {train_src_utts & heldout_src_utts}"
    assert not (test_src_utts & heldout_src_utts), f"Source utterance leak test vs heldout: {test_src_utts & heldout_src_utts}"

    # 3. Dual-class representation in test and held-out splits
    for split_name in ("test", "heldout"):
        labels = set(r["label"] for r in splits[split_name])
        assert "bona_fide" in labels and "spoof" in labels, (
            f"Split '{split_name}' must contain both bona_fide and spoof, found {labels}"
        )


def test_split_cohesion_of_noise_variants():
    """Assert all noise conditions for each source utterance travel together in the exact same split."""
    manifest_path = Path("data/processed/master_manifest.jsonl")
    if not manifest_path.exists():
        pytest.skip(f"Master manifest not on disk (local audio artifact): {manifest_path}")
    with open(manifest_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    utt_to_splits = {}
    for r in records:
        src_id = r["source_utterance_id"]
        utt_to_splits.setdefault(src_id, set()).add(r["split"])

    for src_id, sp_set in utt_to_splits.items():
        assert len(sp_set) == 1, f"Source utterance {src_id} fractured across splits: {sp_set}"
