"""Shared audio I/O helpers. Keep every module reading/writing audio through these
functions rather than calling librosa/soundfile directly, so sample-rate handling and
mono/stereo conventions stay consistent project-wide."""

import os
from pathlib import Path
from typing import Tuple
import numpy as np
import soundfile as sf
import librosa


def load_audio(path: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """Load audio file via soundfile/librosa, resample to target_sr, and force mono.

    Args:
        path: Path to the audio file.
        target_sr: Desired sample rate (default: 16000).

    Returns:
        Tuple of (audio_signal as 1D float32 numpy array, sample_rate).

    Raises:
        FileNotFoundError: If the audio file does not exist.
        ValueError: If the file is corrupt, empty, or unreadable.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Audio file not found: {path}")

    try:
        data, sr = sf.read(path, dtype="float32")
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        if target_sr is not None and sr != target_sr:
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=DeprecationWarning)
                warnings.filterwarnings("ignore", message=".*pkg_resources.*")
                audio = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
        else:
            audio = data
    except Exception as e:
        raise ValueError(f"Corrupt or unreadable audio file: {path}. Error: {e}") from e

    if audio is None or len(audio) == 0:
        raise ValueError(f"Empty or silent audio read from: {path}")

    if np.isnan(audio).any() or np.isinf(audio).any():
        raise ValueError(f"Corrupt audio (NaN or Inf values detected) in: {path}")

    # C3 Fix: Apply identical target RMS loudness normalization to all audio upon load
    # Target RMS of 0.1 (-20 dBFS) prevents peak-normalization leaks
    target_rms = 0.1
    current_rms = float(np.sqrt(np.mean(audio ** 2)))
    if current_rms > 1e-6:
        gain = target_rms / current_rms
        audio = audio * gain
        # Soft-clip to [-0.98, 0.98] to prevent digital clipping while avoiding artificial flat peaks
        max_val = np.max(np.abs(audio))
        if max_val > 0.98:
            audio = audio / max_val * 0.98

    return audio.astype(np.float32), sr


def save_audio(path: str, audio: np.ndarray, sr: int) -> None:
    """Save audio array to file via soundfile, creating parent directories if needed.

    Args:
        path: Destination audio file path.
        audio: 1D or 2D numpy array containing audio samples.
        sr: Sample rate in Hz.

    Raises:
        ValueError: If audio is empty or contains non-finite values.
    """
    if audio is None or len(audio) == 0:
        raise ValueError(f"Cannot save empty audio array to {path}")

    if np.isnan(audio).any() or np.isinf(audio).any():
        raise ValueError(f"Cannot save audio with NaN/Inf values to {path}")

    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    sf.write(str(target_path), audio, sr)
