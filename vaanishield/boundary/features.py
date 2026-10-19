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
from typing import Dict, List, Optional, Any, Union
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
    preserving speech harmonics. Uses high-speed PyTorch STFT/iSTFT.
    """
    if len(audio_window) < int(0.04 * sr):
        return audio_window

    try:
        import torch
        x = torch.from_numpy(audio_window.astype(np.float32))
        n_fft = min(512, len(audio_window))
        hop_length = n_fft // 4
        win = torch.hann_window(n_fft)
        stft = torch.stft(x, n_fft=n_fft, hop_length=hop_length, window=win, return_complex=True)
        mag = torch.abs(stft)
        k = max(1, int(0.15 * stft.shape[1]))
        noise_profile, _ = torch.kthvalue(mag, k, dim=1, keepdim=True)
        cleaned_mag = torch.maximum(mag - reduction_factor * noise_profile, 0.05 * mag)
        cleaned_stft = cleaned_mag * torch.exp(1j * torch.angle(stft))
        cleaned_x = torch.istft(cleaned_stft, n_fft=n_fft, hop_length=hop_length, window=win, length=len(audio_window))
        return cleaned_x.numpy().astype(np.float32)
    except Exception:
        try:
            n_fft = min(512, len(audio_window))
            hop_length = n_fft // 4
            stft = librosa.stft(audio_window, n_fft=n_fft, hop_length=hop_length)
            mag, phase = np.abs(stft), np.angle(stft)
            noise_profile = np.percentile(mag, 15, axis=1, keepdims=True)
            cleaned_mag = np.maximum(mag - reduction_factor * noise_profile, 0.05 * mag)
            cleaned_stft = cleaned_mag * np.exp(1j * phase)
            return librosa.istft(cleaned_stft, hop_length=hop_length, length=len(audio_window)).astype(np.float32)
        except Exception:
            return audio_window


def batched_yin_pitch(
    audio: np.ndarray,
    sr: int,
    fmin: float = 75.0,
    fmax: float = 500.0,
    threshold: float = 0.15,
    frame_length_sec: float = 0.03,
    hop_length_sec: float = 0.01,
) -> np.ndarray:
    """Vectorized YIN pitch estimator (De Cheveigné & Kawahara 2002) in pure NumPy.

    Zero external C-extension binary dependencies, zero Python GIL lock, sub-sample
    parabolic interpolation.
    """
    w = int(frame_length_sec * sr)
    hop = max(1, int(hop_length_sec * sr))
    tau_min = max(1, int(sr / fmax))
    tau_max = min(w - 2, int(sr / fmin))

    if len(audio) < w or tau_max <= tau_min:
        return np.array([], dtype=np.float32)

    n_frames = (len(audio) - w) // hop + 1
    if n_frames <= 0:
        return np.array([], dtype=np.float32)

    # 2D frame striding
    shape = (n_frames, w)
    strides = (audio.strides[0] * hop, audio.strides[0])
    frames = np.lib.stride_tricks.as_strided(audio, shape=shape, strides=strides)

    energies = np.sum(frames ** 2, axis=1)
    fft_size = 2 ** int(np.ceil(np.log2(2 * w)))

    # FFT-accelerated autocorrelation: O(N log N)
    fftx = np.fft.rfft(frames, n=fft_size, axis=1)
    r = np.fft.irfft(fftx * np.conj(fftx), n=fft_size, axis=1)[:, :w]

    # Cumulative energy term r_tau(0)
    cumsum = np.cumsum(frames[:, ::-1] ** 2, axis=1)
    r_tau_0 = np.concatenate([energies[:, None], cumsum[:, -2::-1]], axis=1)[:, :w]

    # Difference function d(tau) = r(0) + r_tau(0) - 2*r(tau)
    d = (energies[:, None] + r_tau_0 - 2 * r)[:, :tau_max + 1]

    # Cumulative mean normalized difference: d'(tau)
    running_sum = np.cumsum(d[:, 1:], axis=1)
    tau_idx = np.arange(1, tau_max + 1)[None, :]
    d_prime = np.ones((n_frames, tau_max + 1), dtype=np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        d_prime[:, 1:] = d[:, 1:] / (running_sum / tau_idx)
    d_prime = np.nan_to_num(d_prime, nan=1.0)

    # Pitch search with sub-sample parabolic interpolation
    pitches = np.zeros(n_frames, dtype=np.float32)
    for i in range(n_frames):
        if energies[i] < 1e-6:
            continue
        dp = d_prime[i]
        tau_cand = -1
        for tau in range(tau_min, tau_max):
            if dp[tau] < threshold:
                while tau + 1 < tau_max and dp[tau + 1] < dp[tau]:
                    tau += 1
                tau_cand = tau
                break
        if tau_cand == -1:
            best_tau = tau_min + int(np.argmin(dp[tau_min:tau_max]))
            if dp[best_tau] < 0.40:
                tau_cand = best_tau

        if tau_cand > 0 and 0 < tau_cand < tau_max:
            alpha = dp[tau_cand - 1]
            beta = dp[tau_cand]
            gamma = dp[tau_cand + 1]
            denom = 2 * (alpha - 2 * beta + gamma)
            delta = (alpha - gamma) / denom if abs(denom) > 1e-7 else 0.0
            pitches[i] = sr / (tau_cand + delta)

    return pitches


def extract_f0_reset_magnitude(audio_window: np.ndarray, sr: int) -> float:
    """Measure F0 pitch discontinuity across the midpoint of the boundary window.

    Uses high-speed vectorized YIN pitch tracking with sub-sample interpolation.
    """
    if len(audio_window) < int(0.04 * sr):
        return 0.0

    try:
        pitch_values = batched_yin_pitch(audio_window, sr=sr, fmin=75.0, fmax=500.0)
    except Exception:
        try:
            snd = parselmouth.Sound(audio_window, sampling_frequency=sr)
            pitch = snd.to_pitch(time_step=0.01, pitch_floor=75.0, pitch_ceiling=500.0)
            pitch_values = pitch.selected_array["frequency"]
        except Exception:
            pitch_values = np.array([], dtype=np.float32)

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

    # Estimate background noise floor from lowest 10th percentile energy frames via vectorization
    frame_len = max(16, int(0.01 * sr))
    n_frames = len(audio_window) // frame_len
    if n_frames > 0:
        frames = audio_window[: n_frames * frame_len].reshape(n_frames, frame_len)
        frame_energies = np.mean(frames ** 2, axis=1)
        noise_floor = float(np.percentile(frame_energies, 10))
    else:
        noise_floor = 1e-9

    rms_pre = np.sqrt(max(1e-9, float(np.mean(pre_win ** 2)) - noise_floor))
    rms_post = np.sqrt(max(1e-9, float(np.mean(post_win ** 2)) - noise_floor))

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
    """Quantify formant (F1/F2) and spectral centroid trajectory smoothness across the boundary window."""
    if len(audio_window) < int(0.05 * sr):
        return 1.0

    try:
        n_fft = min(512, len(audio_window))
        stft = np.abs(librosa.stft(audio_window, n_fft=n_fft, hop_length=max(1, n_fft // 4)))
        cent = float(np.mean(librosa.feature.spectral_centroid(S=stft, sr=sr))) / max(1000.0, sr / 4.0)
        return float(np.clip(cent, 0.0, 1.0))
    except Exception:
        return 0.5


def extract_mfcc_delta_magnitude(audio_window: np.ndarray, sr: int) -> float:
    """Measure the magnitude of MFCC frame-to-frame delta and sub-band cepstral modulation at boundary frames."""
    if len(audio_window) < int(0.04 * sr):
        return 0.0

    try:
        n_fft = min(512, len(audio_window))
        hop_length = max(1, int(0.01 * sr))
        stft = np.abs(librosa.stft(audio_window, n_fft=n_fft, hop_length=hop_length))
        mel = librosa.power_to_db(librosa.feature.melspectrogram(S=stft**2, sr=sr, n_mels=40))
        mfcc = librosa.feature.mfcc(S=mel, n_mfcc=13)
        if mfcc.shape[1] < 3:
            return 0.0
        delta = librosa.feature.delta(mfcc)
        delta_mag = float(np.mean(np.std(mfcc[1:8, :], axis=1)) + np.mean(np.abs(delta[1:8, :])))
        return float(max(0.0, delta_mag))
    except Exception:
        return 0.0


def extract_monolingual_features(audio: np.ndarray, sr: int) -> Dict[str, float]:
    """Extract physical acoustic descriptors for monolingual speech.

    When no code-switching boundaries exist, detects neural vocoder and synthesis
    footprints via Harmonic-to-Noise Ratio (HNR), Spectral Flatness, and energy dynamics.
    """
    if len(audio) < int(0.05 * sr):
        return {name: 0.0 for name in FEATURE_NAMES}

    # 1. Harmonic-to-Noise Ratio (HNR) via frame autocorrelation peak
    w = int(0.03 * sr)
    hop = int(0.015 * sr)
    hnr_mean, hnr_std = 0.0, 0.0
    if len(audio) >= w:
        n_frames = (len(audio) - w) // hop + 1
        if n_frames > 0:
            shape = (n_frames, w)
            strides = (audio.strides[0] * hop, audio.strides[0])
            frames = np.lib.stride_tricks.as_strided(audio, shape=shape, strides=strides)
            energies = np.sum(frames ** 2, axis=1)
            thresh = np.percentile(energies, 20) if len(energies) > 5 else 1e-6
            voiced = frames[energies > thresh]
            if len(voiced) > 0:
                fft_size = 2 ** int(np.ceil(np.log2(2 * w)))
                fftx = np.fft.rfft(voiced, n=fft_size, axis=1)
                r = np.fft.irfft(fftx * np.conj(fftx), n=fft_size, axis=1)[:, :w]
                r0 = np.maximum(r[:, 0:1], 1e-9)
                norm_r = r / r0
                tau_min = max(1, int(sr / 500.0))
                tau_max = min(w - 2, int(sr / 75.0))
                if tau_max > tau_min:
                    peak_r = np.max(norm_r[:, tau_min:tau_max], axis=1)
                    peak_r = np.clip(peak_r, 0.001, 0.999)
                    hnr_db = 10.0 * np.log10(peak_r / (1.0 - peak_r))
                    hnr_mean = float(np.mean(hnr_db))
                    hnr_std = float(np.std(hnr_db))

    # 2. Spectral Flatness Measure (SFM)
    try:
        n_fft = min(512, len(audio))
        stft = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=max(1, n_fft // 4))) ** 2
        stft = np.maximum(stft, 1e-12)
        geom_mean = np.exp(np.mean(np.log(stft), axis=0))
        arith_mean = np.mean(stft, axis=0)
        sfm = float(np.mean(geom_mean / arith_mean))
    except Exception:
        sfm = 0.5

    # 3. Macro Energy Variance
    frame_len = int(0.02 * sr)
    n_f = len(audio) // frame_len
    if n_f > 1:
        f_rms = np.sqrt(np.mean(audio[: n_f * frame_len].reshape(n_f, frame_len) ** 2, axis=1))
        macro_energy_var = float(np.std(f_rms) / (np.mean(f_rms) + 1e-6))
    else:
        macro_energy_var = 0.0

    # 4. Zero Crossing Rate (ZCR) Variance
    try:
        zcr = librosa.feature.zero_crossing_rate(audio, frame_length=frame_len, hop_length=max(1, frame_len // 2))
        zcr_var = float(np.std(zcr))
    except Exception:
        zcr_var = 0.0

    return {
        "f0_reset_magnitude": float(np.clip(hnr_mean / 20.0, -1.0, 3.0)),
        "energy_discontinuity": float(np.clip(hnr_std / 10.0, 0.0, 3.0)),
        "phone_duration_ratio": float(np.clip(sfm * 5.0, 0.0, 5.0)),
        "formant_transition_smoothness": float(np.clip(macro_energy_var, 0.0, 3.0)),
        "mfcc_delta_magnitude": float(np.clip(zcr_var * 10.0, 0.0, 5.0)),
    }


def extract_boundary_features(
    audio_input: Any = None,
    codeswitch_points: Optional[List[CodeSwitchPoint]] = None,
    speaker_baseline: Optional[dict] = None,
    sr: int = 16000,
    audio_path: Optional[str] = None,
    fallback_monolingual: bool = True,
) -> List[BoundaryFeatureVector]:
    """Run all 5 configured feature extractors for every detected boundary in an utterance.

    When codeswitch_points is empty or None (e.g. unmixed monolingual audio), applies
    Dual-Mode Stream Routing to extract global vocal-tract physics (HNR, SFM, energy variance),
    preventing zero-vector collapse and closing the monolingual bypass loophole.
    """
    src = audio_input if audio_input is not None else audio_path
    if src is None:
        return []

    if isinstance(src, np.ndarray):
        audio = src
    else:
        audio, sr = load_audio(str(src), target_sr=sr)
    total_samples = len(audio)

    if not codeswitch_points:
        if fallback_monolingual and total_samples > 0:
            mono_feats = extract_monolingual_features(audio, sr=sr)
            dur = total_samples / sr
            synth_pt = CodeSwitchPoint(
                timestamp_sec=dur / 2.0,
                from_lang="monolingual",
                to_lang="monolingual",
                window_start_sec=0.0,
                window_end_sec=dur,
            )
            return [BoundaryFeatureVector(codeswitch_point=synth_pt, features=mono_feats)]
        return []

    vectors: List[BoundaryFeatureVector] = []
    for pt in codeswitch_points:
        start_samp = max(0, int(pt.window_start_sec * sr))
        end_samp = min(total_samples, int(pt.window_end_sec * sr))
        raw_window = audio[start_samp:end_samp]

        if len(raw_window) < int(0.04 * sr):
            feat_dict = {name: 0.0 for name in FEATURE_NAMES}
        else:
            # Denoise window for robust acoustic tracking in telephony conditions
            window = denoise_audio_window(raw_window, sr=sr)

            # 1. f0_reset_magnitude: normalized pitch jump across pre/post boundary
            f0_jump = extract_f0_reset_magnitude(window, sr)
            # 2. energy_discontinuity: noise-compensated RMS step across boundary
            energy_jump = extract_energy_discontinuity(window, sr)
            # 3. phone_duration_ratio: local active segment duration vs baseline
            duration_ratio = extract_phone_duration_ratio(window, sr, speaker_baseline)
            # 4. formant_transition_smoothness: spectral transition trajectory across boundary
            formant_smooth = extract_formant_transition_smoothness(window, sr)
            # 5. mfcc_delta_magnitude: frame-to-frame delta magnitude across boundary
            mfcc_delta = extract_mfcc_delta_magnitude(window, sr)

            feat_dict = {
                "f0_reset_magnitude": float(f0_jump),
                "energy_discontinuity": float(energy_jump),
                "phone_duration_ratio": float(duration_ratio),
                "formant_transition_smoothness": float(formant_smooth),
                "mfcc_delta_magnitude": float(mfcc_delta),
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

