import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import scipy.signal
import yaml

from vaanishield.utils.audio_io import load_audio, save_audio
from vaanishield.utils.logging_utils import get_logger

logger = get_logger("augment_noise")


def derive_seed(key: str) -> int:
    """Derive deterministic non-negative 31-bit integer seed from string key."""
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest(), 16) % (2**31)


def apply_codec_compression(audio: np.ndarray, sr: int) -> np.ndarray:
    """Simulate telecom bandpass and mu-law / A-law codec quantization (300Hz - 3400Hz)."""
    # 1. Butterworth bandpass filter representing PSTN / VoIP telephony band
    nyq = 0.5 * sr
    low = max(0.01, 300.0 / nyq)
    high = min(0.99, 3400.0 / nyq)
    b, a = scipy.signal.butter(4, [low, high], btype="band")
    filtered = scipy.signal.lfilter(b, a, audio)

    # 2. 8-bit mu-law companding & quantization simulation
    mu = 255.0
    x = np.clip(filtered, -1.0, 1.0)
    x_comp = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    x_quant = np.round(x_comp * 128.0) / 128.0
    decompressed = np.sign(x_quant) * ((1.0 + mu) ** np.abs(x_quant) - 1.0) / mu

    return decompressed.astype(np.float32)


def apply_amr_wb_simulation(audio: np.ndarray, sr: int) -> np.ndarray:
    """Simulate Adaptive Multi-Rate Wideband (AMR-WB / G.722.2) 50Hz - 7000Hz mobile codec."""
    nyq = 0.5 * sr
    low = max(0.005, 50.0 / nyq)
    high = min(0.99, 7000.0 / nyq)
    b, a = scipy.signal.butter(4, [low, high], btype="band")
    filtered = scipy.signal.lfilter(b, a, audio)

    bits = 14
    levels = 2 ** (bits - 1)
    quantized = np.round(np.clip(filtered, -1.0, 1.0) * levels) / levels
    return quantized.astype(np.float32)


def apply_packet_loss_simulation(
    audio: np.ndarray,
    sr: int,
    loss_rate: float = 0.05,
    burst_length_ms: int = 40,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Simulate VoIP / cellular packet loss with burst drops and zero-fill/attenuation concealment."""
    rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState(42)
    degraded = audio.copy()
    burst_samples = int((burst_length_ms / 1000.0) * sr)
    num_samples = len(audio)

    if num_samples <= burst_samples:
        return degraded

    num_bursts = max(1, int((num_samples / burst_samples) * loss_rate))
    drop_indices = rng.choice(num_samples - burst_samples, size=num_bursts, replace=False)

    for idx in drop_indices:
        window = np.hanning(burst_samples)
        degraded[idx : idx + burst_samples] *= (1.0 - window) * 0.1

    return degraded.astype(np.float32)


def apply_background_noise_injection(
    audio: np.ndarray,
    sr: int,
    snr_db: float = 15.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Inject stationary + pink phone background noise at a target SNR."""
    rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState(42)
    num_samples = len(audio)
    signal_power = np.mean(audio ** 2)
    if signal_power == 0:
        return audio

    white = rng.normal(0, 1, num_samples)
    b, a = scipy.signal.butter(2, 1500.0 / (0.5 * sr), btype="low")
    shaped_noise = scipy.signal.lfilter(b, a, white)

    noise_power = np.mean(shaped_noise ** 2)
    if noise_power == 0:
        return audio

    target_noise_power = signal_power / (10 ** (snr_db / 10.0))
    scale = np.sqrt(target_noise_power / noise_power)
    noisy_audio = audio + scale * shaped_noise

    max_amp = np.max(np.abs(noisy_audio))
    if max_amp > 1.0:
        noisy_audio = noisy_audio / max_amp * 0.98

    return noisy_audio.astype(np.float32)


def augment_manifest(
    input_manifest: str,
    config_path: str = "configs/default.yaml",
    output_dir: str = "data/processed/augmented",
) -> List[dict]:
    """Batch augment audio records across configured noise conditions with per-utterance seeding."""
    # Guard against re-augmenting an already augmented manifest
    if "augmented" in Path(input_manifest).name:
        raise ValueError(f"Refusing to re-augment an already augmented manifest: {input_manifest}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    noise_augmentations = cfg.get("dataset_construction", {}).get(
        "noise_augmentations",
        ["codec_compression", "packet_loss_simulation", "background_noise_injection"],
    )

    out_base = Path(output_dir)
    out_base.mkdir(parents=True, exist_ok=True)

    with open(input_manifest, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    augmented_records = []

    for rec in records:
        clean_rec = dict(rec)
        if "noise_condition" not in clean_rec:
            clean_rec["noise_condition"] = "clean"
        augmented_records.append(clean_rec)

        audio_path = rec["audio_path"]
        try:
            audio, sr = load_audio(audio_path)
        except Exception as e:
            logger.error(f"Failed to load {audio_path}: {e}")
            continue

        utt_id = rec.get("utterance_id", Path(audio_path).stem)

        if "codec_compression" in noise_augmentations:
            deg_audio = apply_codec_compression(audio, sr)
            out_file = out_base / f"{utt_id}_codec.wav"
            save_audio(str(out_file), deg_audio, sr)
            r = dict(rec)
            r["utterance_id"] = f"{utt_id}_codec"
            r["audio_path"] = str(out_file)
            r["noise_condition"] = "codec_compression"
            r["seed"] = derive_seed(r["utterance_id"])
            augmented_records.append(r)

        if "amr_wb_compression" in noise_augmentations:
            deg_audio = apply_amr_wb_simulation(audio, sr)
            out_file = out_base / f"{utt_id}_amrwb.wav"
            save_audio(str(out_file), deg_audio, sr)
            r = dict(rec)
            r["utterance_id"] = f"{utt_id}_amrwb"
            r["audio_path"] = str(out_file)
            r["noise_condition"] = "amr_wb_compression"
            r["seed"] = derive_seed(r["utterance_id"])
            augmented_records.append(r)

        if "packet_loss_simulation" in noise_augmentations:
            seed = derive_seed(f"{utt_id}_packetloss")
            deg_audio = apply_packet_loss_simulation(audio, sr, loss_rate=0.05, seed=seed)
            out_file = out_base / f"{utt_id}_packetloss.wav"
            save_audio(str(out_file), deg_audio, sr)
            r = dict(rec)
            r["utterance_id"] = f"{utt_id}_packetloss"
            r["audio_path"] = str(out_file)
            r["noise_condition"] = "packet_loss_simulation"
            r["seed"] = seed
            augmented_records.append(r)

        if "background_noise_injection" in noise_augmentations:
            seed = derive_seed(f"{utt_id}_bgnoise")
            deg_audio = apply_background_noise_injection(audio, sr, snr_db=15.0, seed=seed)
            out_file = out_base / f"{utt_id}_bgnoise.wav"
            save_audio(str(out_file), deg_audio, sr)
            r = dict(rec)
            r["utterance_id"] = f"{utt_id}_bgnoise"
            r["audio_path"] = str(out_file)
            r["noise_condition"] = "background_noise_injection"
            r["seed"] = seed
            augmented_records.append(r)

    manifest_file = Path(input_manifest).parent / "augmented_dataset_manifest.jsonl"
    with open(manifest_file, "w", encoding="utf-8") as f:
        for r in augmented_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    logger.info(
        f"Generated {len(augmented_records)} records across conditions "
        f"(clean, codec, packet_loss, bg_noise) -> {manifest_file}"
    )
    return augmented_records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply telecom channel degradation augmentations")
    parser.add_argument("--input-manifest", required=True, help="Input dataset manifest JSONL")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to default.yaml")
    parser.add_argument("--output-dir", default="data/processed/augmented", help="Augmented audio output dir")
    args = parser.parse_args()

    augment_manifest(args.input_manifest, args.config, args.output_dir)
