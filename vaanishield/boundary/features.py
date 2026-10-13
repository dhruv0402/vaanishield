"""
Boundary-local feature extraction.

For each CodeSwitchPoint (see localization.py), extract fine-grained prosodic and
spectral discontinuity features from the audio window around it. This is the project's
core novel signal -- see docs/TRD.md section 3.2 and docs/VaaniShield_Abstract_ProblemStatement.pdf
for why this specific signal (rather than generic global artifacts) is the hypothesis
under test.

Features to implement (configs/default.yaml -> boundary_features.features):
    - f0_reset_magnitude: discontinuity in pitch contour across the boundary
    - energy_discontinuity: jump in short-term energy across the boundary
    - phone_duration_ratio: duration of phones near the boundary vs. the speaker's own
      baseline phone-duration distribution (requires per-speaker baseline stats)
    - formant_transition_smoothness: how smoothly formants move across the boundary
      (vs. an abrupt, unnatural transition)
    - mfcc_delta_magnitude: frame-to-frame MFCC delta magnitude in the window

Use librosa for spectral features and parselmouth (Praat) for pitch/formant-related
features -- parselmouth is generally more reliable for F0 and formant tracking than
librosa's built-in pitch estimation.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import parselmouth
import librosa

from vaanishield.utils.audio_io import load_audio
from .localization import CodeSwitchPoint

FEATURE_NAMES = [
    "f0_reset_magnitude",
    "energy_discontinuity",
    "phone_duration_ratio",
    "formant_transition_smoothness",
    "mfcc_delta_magnitude",
]


@dataclass
class BoundaryFeatureVector:
    codeswitch_point: CodeSwitchPoint
    features: Dict[str, float]


def denoise_audio_window(audio_window: np.ndarray, sr: int, reduction_factor: float = 0.75) -> np.ndarray:
    """Apply spectral subtraction noise reduction to stabilize boundary prosodic features.

    Subtracts stationary background noise floor estimated from minimum magnitude
    frames, preventing pitch tracking failures in noisy telecom conditions while
    preserving speech harmonics.
    """
    if len(audio_window) < int(0.04 * sr):
        return audio_window

    try:
        n_fft = min(512, len(audio_window))
        hop_length = n_fft // 4
        stft = librosa.stft(audio_window, n_fft=n_fft, hop_length=hop_length)
        mag, phase = np.abs(stft), np.angle(stft)

        # Estimate stationary noise profile from bottom 15th percentile energy frames
        noise_profile = np.percentile(mag, 15, axis=1, keepdims=True)
        cleaned_mag = np.maximum(mag - reduction_factor * noise_profile, 0.05 * mag)

        cleaned_stft = cleaned_mag * np.exp(1j * phase)
        cleaned_audio = librosa.istft(cleaned_stft, hop_length=hop_length, length=len(audio_window))
        return cleaned_audio.astype(np.float32)
    except Exception:
        return audio_window


def extract_f0_reset_magnitude(audio_window: np.ndarray, sr: int) -> float:
    """Measure F0 pitch discontinuity across the midpoint of the boundary window.

    Uses Praat Parselmouth to track pitch across the pre-boundary (first half) and
    post-boundary (second half) segments. Computes the absolute jump in median F0
    normalized by mean F0, capturing unnatural pitch resets.
    """
    if len(audio_window) < int(0.04 * sr):
        return 0.0

    try:
        snd = parselmouth.Sound(audio_window, sampling_frequency=sr)
        pitch = snd.to_pitch(time_step=0.01, pitch_floor=75.0, pitch_ceiling=500.0)
        pitch_values = pitch.selected_array["frequency"]
    except Exception:
        # Fallback to librosa pyin if praat throws error
        f0, _, _ = librosa.pyin(audio_window, fmin=75, fmax=500, sr=sr)
        pitch_values = f0 if f0 is not None else np.array([])

    if len(pitch_values) == 0:
        return 0.0

    mid = len(pitch_values) // 2
    pre_f0 = pitch_values[:mid]
    post_f0 = pitch_values[mid:]

    pre_voiced = pre_f0[pre_f0 > 0]
    post_voiced = post_f0[post_f0 > 0]

    if len(pre_voiced) == 0 or len(post_voiced) == 0:
        return 0.0

    med_pre = float(np.median(pre_voiced))
    med_post = float(np.median(post_voiced))
    mean_all = float(np.mean(np.concatenate([pre_voiced, post_voiced])))

    if mean_all <= 0:
        return 0.0

    # Normalized absolute pitch jump across switch boundary
    jump = abs(med_post - med_pre) / mean_all
    return float(np.clip(jump, 0.0, 5.0))


def extract_energy_discontinuity(audio_window: np.ndarray, sr: int) -> float:
    """Compute noise-compensated short-term RMS energy jump across the boundary midpoint."""
    if len(audio_window) < int(0.04 * sr):
        return 0.0

    mid = len(audio_window) // 2
    pre_win = audio_window[:mid]
    post_win = audio_window[mid:]

    # Estimate background noise floor from lowest 10th percentile energy frames
    frame_len = max(16, int(0.01 * sr))
    frame_energies = [
        np.mean(audio_window[i : i + frame_len] ** 2)
        for i in range(0, len(audio_window) - frame_len, frame_len)
    ]
    noise_floor = float(np.percentile(frame_energies, 10)) if frame_energies else 1e-9

    rms_pre = np.sqrt(max(1e-9, np.mean(pre_win ** 2) - noise_floor))
    rms_post = np.sqrt(max(1e-9, np.mean(post_win ** 2) - noise_floor))

    # Relative log energy difference across the boundary compensated for ambient noise
    energy_jump = float(abs(np.log(rms_post) - np.log(rms_pre)))
    return float(np.clip(energy_jump, 0.0, 10.0))


def extract_phone_duration_ratio(
    audio_window: np.ndarray,
    sr: int,
    speaker_baseline: Optional[dict] = None,
) -> float:
    """Compare speech segment duration near boundary vs speaker's baseline phone duration.

    Synthesizers stitching distinct language models often exhibit unnatural pauses
    or elongated transition syllables at code-switch points.
    """
    baseline_duration = 0.08  # Default 80ms baseline phone duration
    if speaker_baseline and "mean_phone_duration_sec" in speaker_baseline:
        baseline_duration = float(speaker_baseline["mean_phone_duration_sec"])

    # Estimate active voicing/sound duration in the central 100ms around boundary
    mid = len(audio_window) // 2
    span = int(0.05 * sr)
    center_seg = audio_window[max(0, mid - span) : min(len(audio_window), mid + span)]

    if len(center_seg) == 0:
        return 1.0

    energy = center_seg ** 2
    threshold = np.mean(energy) * 0.1
    active_samples = np.sum(energy > threshold)
    observed_duration = active_samples / sr

    ratio = observed_duration / max(0.01, baseline_duration)
    return float(np.clip(ratio, 0.0, 10.0))


def extract_formant_transition_smoothness(audio_window: np.ndarray, sr: int) -> float:
    """Quantify formant (F1/F2) trajectory smoothness across the boundary window.

    A genuine human speaker exhibits co-articulation and smooth formant shifts;
    concatenative or cross-lingual TTS switching exhibits sharp discontinuities
    (high second derivative / low smoothness score).
    """
    if len(audio_window) < int(0.05 * sr):
        return 1.0

    try:
        snd = parselmouth.Sound(audio_window, sampling_frequency=sr)
        formants = snd.to_formant_burg(time_step=0.01, max_number_of_formants=5)
        times = formants.ts()
        f1_vals = [formants.get_value_at_time(1, t) for t in times]
        f2_vals = [formants.get_value_at_time(2, t) for t in times]

        f1 = np.array([v for v in f1_vals if not np.isnan(v)])
        f2 = np.array([v for v in f2_vals if not np.isnan(v)])
    except Exception:
        f1, f2 = np.array([]), np.array([])

    if len(f1) < 4 or len(f2) < 4:
        return 1.0

    # Second derivative measures curvature / acceleration
    diff2_f1 = np.diff(f1, n=2)
    diff2_f2 = np.diff(f2, n=2)
    roughness = float(np.mean(np.abs(diff2_f1)) + np.mean(np.abs(diff2_f2)))

    # Smoothness is inverse of curvature/roughness (higher = smoother)
    smoothness = 100.0 / (roughness + 100.0)
    return float(np.clip(smoothness, 0.0, 1.0))


def extract_mfcc_delta_magnitude(audio_window: np.ndarray, sr: int) -> float:
    """Measure the magnitude of MFCC frame-to-frame delta coefficients at boundary frames."""
    if len(audio_window) < int(0.04 * sr):
        return 0.0

    try:
        n_fft = min(512, len(audio_window))
        hop_length = n_fft // 4
        mfcc = librosa.feature.mfcc(y=audio_window, sr=sr, n_mfcc=13, n_fft=n_fft, hop_length=hop_length)
        if mfcc.shape[1] < 3:
            return 0.0
        delta = librosa.feature.delta(mfcc)
        # Center frames around boundary
        mid_frame = delta.shape[1] // 2
        center_delta = delta[:, max(0, mid_frame - 2) : min(delta.shape[1], mid_frame + 3)]
        delta_mag = float(np.mean(np.abs(center_delta)))
        return float(np.clip(delta_mag, 0.0, 50.0))
    except Exception:
        return 0.0


def extract_boundary_features(
    audio_path: str,
    codeswitch_points: List[CodeSwitchPoint],
    speaker_baseline: Optional[dict] = None,
) -> List[BoundaryFeatureVector]:
    """Run all 5 configured feature extractors for every detected boundary in an utterance."""
    if not codeswitch_points:
        return []

    audio, sr = load_audio(audio_path)
    total_samples = len(audio)

    vectors: List[BoundaryFeatureVector] = []
    for pt in codeswitch_points:
        start_samp = max(0, int(pt.window_start_sec * sr))
        end_samp = min(total_samples, int(pt.window_end_sec * sr))
        window = audio[start_samp:end_samp]

        if len(window) == 0:
            feat_dict = {name: 0.0 for name in FEATURE_NAMES}
        else:
            # Apply spectral subtraction noise reduction to stabilize boundary prosodic features
            clean_window = denoise_audio_window(window, sr)
            feat_dict = {
                "f0_reset_magnitude": extract_f0_reset_magnitude(clean_window, sr),
                "energy_discontinuity": extract_energy_discontinuity(clean_window, sr),
                "phone_duration_ratio": extract_phone_duration_ratio(clean_window, sr, speaker_baseline),
                "formant_transition_smoothness": extract_formant_transition_smoothness(clean_window, sr),
                "mfcc_delta_magnitude": extract_mfcc_delta_magnitude(clean_window, sr),
            }

        vectors.append(BoundaryFeatureVector(codeswitch_point=pt, features=feat_dict))

    return vectors


def aggregate_boundary_features(
    boundary_vectors: List[BoundaryFeatureVector],
    aggregation: str = "mean_and_max",
) -> np.ndarray:
    """Combine per-boundary vectors into a fixed-length utterance-level feature vector.

    If aggregation == "mean_and_max":
        Vector dimension = 2 * len(FEATURE_NAMES) = 10 (mean and max of each feature).
    If no boundaries were detected (e.g. monolingual clip), returns a clean zero vector
    of the exact fixed length.
    """
    n_feats = len(FEATURE_NAMES)
    dim = 2 * n_feats if aggregation == "mean_and_max" else n_feats

    if not boundary_vectors:
        return np.zeros(dim, dtype=np.float32)

    matrix = np.array(
        [[vec.features[k] for k in FEATURE_NAMES] for vec in boundary_vectors],
        dtype=np.float32,
    )  # shape: (num_boundaries, 5)

    if aggregation == "mean_and_max":
        mean_vals = np.mean(matrix, axis=0)
        max_vals = np.max(matrix, axis=0)
        return np.concatenate([mean_vals, max_vals]).astype(np.float32)
    else:
        return np.mean(matrix, axis=0).astype(np.float32)

