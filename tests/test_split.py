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
    assert "val" in splits, "val split missing"
    assert "test" in splits, "test split missing"
    assert "heldout" in splits, "heldout split missing"

    split_names = ["train", "val", "test", "heldout"]
    for i in range(len(split_names)):
        for j in range(i + 1, len(split_names)):
            s1, s2 = split_names[i], split_names[j]
            spk1 = set(r["speaker_id"] for r in splits[s1])
            spk2 = set(r["speaker_id"] for r in splits[s2])
            assert not (spk1 & spk2), f"Speaker leak {s1} vs {s2}: {spk1 & spk2}"

            utt1 = set(r["source_utterance_id"] for r in splits[s1] if r.get("source_utterance_id"))
            utt2 = set(r["source_utterance_id"] for r in splits[s2] if r.get("source_utterance_id"))
            assert not (utt1 & utt2), f"Source utterance leak {s1} vs {s2}: {utt1 & utt2}"

    # 3. Dual-class representation in val, test, and held-out splits
    for split_name in ("val", "test", "heldout"):
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
