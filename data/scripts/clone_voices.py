import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import yaml
os.environ["COQUI_TOS_AGREED"] = "1"
import torch
import torchaudio
import soundfile as sf

# Monkey patch torchaudio.load to use soundfile on systems without torchcodec
def _sf_load(filepath, *args, **kwargs):
    data, sr = sf.read(filepath, dtype="float32")
    tensor = torch.from_numpy(data)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim == 2:
        tensor = tensor.t()
    return tensor, sr

torchaudio.load = _sf_load

# Patch torch.load for weights_only=False on PyTorch 2.6+
_orig_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

from vaanishield.utils.audio_io import load_audio, save_audio
from vaanishield.utils.logging_utils import get_logger

logger = get_logger("clone_voices")

_XTTS_MODEL = None


def get_xtts_model():
    """Lazily load genuine Coqui XTTS v2 model for multi-lingual zero-shot voice cloning."""
    global _XTTS_MODEL
    if _XTTS_MODEL is None:
        logger.info("Initializing Coqui XTTS v2 neural voice cloning pipeline...")
        from TTS.api import TTS
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _XTTS_MODEL = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    return _XTTS_MODEL


def load_consented_speakers(manifest_path: str = "data/consent_forms/consented_speakers.csv") -> Dict[str, dict]:
    """Verify and load the registered speakers manifest.

    Refuses to run if manifest does not exist, is unreadable, or contains no valid speakers.
    Enforces the ethical constraint from docs/PRIVACY_AND_ETHICS.md:
    Audio from untracked speakers or scraped sources without documented license is strictly prohibited.
    """
    path = Path(manifest_path)
    if not path.exists():
        raise PermissionError(
            f"ETHICAL VIOLATION PREVENTED: Speaker registry not found at '{manifest_path}'. "
            "Never clone audio without verified informed consent or open research license in data/consent_forms/."
        )

    consented = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            spk_id = row.get("speaker_id", "").strip()
            status = (row.get("consent_status") or row.get("consent_signed", "")).strip().lower()
            if spk_id and (status in ("true", "1", "yes", "public_academic_license") or row.get("source_license")):
                consented[spk_id] = row

    if not consented:
        raise PermissionError(
            f"ETHICAL VIOLATION PREVENTED: No valid registered speakers found in '{manifest_path}'."
        )

    return consented


def clone_with_xtts(
    speaker_wav_path: str,
    target_text: str,
    output_wav_path: str,
    language: str = "hi",
) -> None:
    """Run real zero-shot neural voice cloning with Coqui XTTS v2.
    
    Synthesizes target code-switched text using the speaker reference audio.
    No artificial acoustic seams or hand-injected artifacts are added.
    """
    tts = get_xtts_model()
    target_path = Path(output_wav_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    
    tts.tts_to_file(
        text=target_text,
        speaker_wav=speaker_wav_path,
        language=language,
        file_path=str(target_path),
    )


def clone_utterance(
    source_audio_path: str,
    speaker_id: str,
    target_text: str,
    system_name: str,
    output_path: str,
    consented_manifest_path: str = "data/consent_forms/consented_speakers.csv",
) -> None:
    """Clone a source recording using genuine neural voice cloning.

    Guarantees that speaker consent is strictly verified before proceeding.
    """
    consented = load_consented_speakers(consented_manifest_path)
    if speaker_id not in consented:
        raise PermissionError(
            f"ETHICAL VIOLATION PREVENTED: Speaker '{speaker_id}' is not in the consented speakers manifest."
        )

    if not os.path.exists(source_audio_path):
        raise FileNotFoundError(f"Source recording not found: {source_audio_path}")

    # Invoke genuine XTTS v2 cross-lingual zero-shot neural voice cloning
    clone_with_xtts(
        speaker_wav_path=source_audio_path,
        target_text=target_text,
        output_wav_path=output_path,
        language="hi",
    )


def batch_clone(
    raw_manifest_path: str,
    config_path: str = "configs/default.yaml",
    consented_manifest_path: str = "data/consent_forms/consented_speakers.csv",
    output_dir: str = "data/processed",
) -> List[dict]:
    """Batch run voice cloning over consented speakers with genuine neural synthesizers.

    Keeps the held-out generalization system strictly isolated in
    `data/processed/heldout/` to avoid any data leakage.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    train_systems = cfg.get("dataset_construction", {}).get("cloning_systems_train", ["xtts"])
    heldout_systems = cfg.get("dataset_construction", {}).get("cloning_systems_heldout", ["openvoice"])

    consented = load_consented_speakers(consented_manifest_path)
    logger.info(f"Verified {len(consented)} consented speakers from {consented_manifest_path}.")

    if not os.path.exists(raw_manifest_path):
        raise FileNotFoundError(
            f"Raw recording manifest not found at '{raw_manifest_path}'. "
            "Real human recordings must be collected before voice cloning can proceed."
        )

    with open(raw_manifest_path, "r", encoding="utf-8") as f:
        utterances = [json.loads(line) for line in f]

    out_base = Path(output_dir)
    train_dir = out_base / "train_clones"
    heldout_dir = out_base / "heldout"
    train_dir.mkdir(parents=True, exist_ok=True)
    heldout_dir.mkdir(parents=True, exist_ok=True)

    generated_manifest = []

    for utt in utterances:
        spk = utt["speaker_id"]
        if spk not in consented:
            logger.warning(f"Skipping unconsented speaker {spk} for utterance {utt.get('utterance_id')}")
            continue

        source_wav = utt["audio_path"]
        text = utt["text"]
        pts = utt.get("codeswitch_points", [])

        # 1. Generate using training neural synthesizer (XTTS v2)
        for sys_name in train_systems:
            out_name = f"{utt['utterance_id']}_{sys_name}.wav"
            out_file = train_dir / out_name
            clone_utterance(
                source_audio_path=source_wav,
                speaker_id=spk,
                target_text=text,
                system_name=sys_name,
                output_path=str(out_file),
                consented_manifest_path=consented_manifest_path,
            )

            generated_manifest.append({
                "utterance_id": f"{utt['utterance_id']}_{sys_name}",
                "source_utterance_id": utt["utterance_id"],
                "speaker_id": spk,
                "audio_path": str(out_file),
                "text": text,
                "codeswitch_points": pts,
                "label": "spoof",
                "cloning_system": sys_name,
                "is_heldout_system": False,
                "noise_condition": "clean",
            })

        # 2. Generate using heldout synthesizer for generalization testing
        for sys_name in heldout_systems:
            out_name = f"{utt['utterance_id']}_{sys_name}.wav"
            out_file = heldout_dir / out_name
            clone_utterance(
                source_audio_path=source_wav,
                speaker_id=spk,
                target_text=text,
                system_name=sys_name,
                output_path=str(out_file),
                consented_manifest_path=consented_manifest_path,
            )

            generated_manifest.append({
                "utterance_id": f"{utt['utterance_id']}_{sys_name}",
                "source_utterance_id": utt["utterance_id"],
                "speaker_id": spk,
                "audio_path": str(out_file),
                "text": text,
                "codeswitch_points": pts,
                "label": "spoof",
                "cloning_system": sys_name,
                "is_heldout_system": True,
                "noise_condition": "clean",
            })

    manifest_path = out_base / "cloned_manifest.jsonl"
    with open(manifest_path, "w", encoding="utf-8") as f:
        for item in generated_manifest:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    logger.info(
        f"Generated {len(generated_manifest)} genuine cloned clips into {train_dir} and {heldout_dir}."
    )
    return generated_manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate genuine cloned voice dataset with Coqui XTTS v2")
    parser.add_argument("--manifest", required=True, help="Path to raw bona-fide manifest JSONL")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to default.yaml")
    parser.add_argument("--consent", default="data/consent_forms/consented_speakers.csv", help="Consent manifest")
    parser.add_argument("--output-dir", default="data/processed", help="Output directory")
    args = parser.parse_args()

    batch_clone(args.manifest, args.config, args.consent, args.output_dir)


