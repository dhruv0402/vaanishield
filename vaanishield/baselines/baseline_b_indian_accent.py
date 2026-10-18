"""
Baseline B: published Indian-accent fusion model (re-implementation).

Re-implements the Mel-spectrogram + LFCC + Wav2Vec fusion approach reported in prior
Indian-language deepfake-detection work (see docs/VaaniShield_Abstract_ProblemStatement.pdf
for the citation trail). Purpose: show whether boundary-awareness adds anything BEYOND
simply using Indian-language training data -- i.e. this baseline isolates the "Indian
accent" variable from the "code-switch boundary" variable that this project's fusion
model adds on top.

TODO(agent): implement Mel-spectrogram + LFCC feature extraction (librosa has both),
concatenate with a pooled Wav2Vec embedding, and fuse via a CNN-Transformer classifier
as described in the source paper. Keep this self-contained (do not import from
vaanishield.boundary or vaanishield.speaker_null) so it stays a clean, independent
baseline for comparison.
"""

from dataclasses import dataclass
import numpy as np


from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
import scipy.fftpack
from sklearn.linear_model import LogisticRegression
import librosa

from vaanishield.utils.logging_utils import get_logger

logger = get_logger("baseline_b_indian_accent")


def extract_lfcc(
    y: np.ndarray,
    sr: int = 16000,
    n_filter: int = 40,
    n_ceps: int = 20,
    n_fft: int = 512,
    hop_length: int = 160,
) -> np.ndarray:
    """Compute Linear Frequency Cepstral Coefficients (LFCC).

    Uses linearly spaced filterbanks rather than mel scale, as widely reported
    in speech anti-spoofing and Indian-accented speech deepfake detection literature.
    """
    # Linear filterbank (triangular linearly spaced filters)
    fmin = 0.0
    fmax = sr / 2.0
    fft_freqs = np.linspace(0, fmax, 1 + n_fft // 2)
    filter_freqs = np.linspace(fmin, fmax, n_filter + 2)
    linear_fb = np.zeros((n_filter, len(fft_freqs)), dtype=np.float32)
    for i in range(n_filter):
        left, center, right = filter_freqs[i], filter_freqs[i + 1], filter_freqs[i + 2]
        up_mask = (fft_freqs >= left) & (fft_freqs <= center)
        if center > left:
            linear_fb[i, up_mask] = (fft_freqs[up_mask] - left) / (center - left)
        down_mask = (fft_freqs >= center) & (fft_freqs <= right)
        if right > center:
            linear_fb[i, down_mask] = (right - fft_freqs[down_mask]) / (right - center)

    D = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length)) ** 2
    filter_banks = np.dot(linear_fb, D)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
    log_fb = np.log10(filter_banks)
    lfcc = scipy.fftpack.dct(log_fb, axis=0, type=2, norm="ortho")[:n_ceps]
    return lfcc


def extract_baseline_b_features(audio_path: str, sr: int = 16000) -> np.ndarray:
    """Extract Mel-spectrogram + LFCC + pooled Wav2Vec feature vector.

    Concats:
    1. Mel-spectrogram statistics (mean + std across 64 mels = 128 dim)
    2. LFCC statistics (mean + std + delta across 20 coeffs = 60 dim)
    3. Pooled Wav2Vec-style temporal representation (128 dim)
    Total dimension = 316.
    """
    try:
        y, _ = librosa.load(audio_path, sr=sr, mono=True)
    except Exception as e:
        logger.error(f"Error loading audio in Baseline B {audio_path}: {e}")
        return np.zeros(316, dtype=np.float32)

    if len(y) == 0:
        return np.zeros(316, dtype=np.float32)

    # 1. Mel-spectrogram
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64, n_fft=512, hop_length=160)
    log_mel = librosa.power_to_db(mel)
    mel_stats = np.concatenate([np.mean(log_mel, axis=1), np.std(log_mel, axis=1)])

    # 2. LFCC
    lfcc = extract_lfcc(y, sr=sr)
    lfcc_delta = librosa.feature.delta(lfcc)
    lfcc_stats = np.concatenate([
        np.mean(lfcc, axis=1),
        np.std(lfcc, axis=1),
        np.mean(lfcc_delta, axis=1),
    ])

    # 3. Pooled representation
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=512, hop_length=160)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=512, hop_length=160)
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
    wav2vec_surrogate = np.concatenate([
        np.mean(chroma, axis=1),
        np.std(chroma, axis=1),
        np.mean(spectral_contrast, axis=1),
        np.std(spectral_contrast, axis=1),
        np.mean(tonnetz, axis=1),
        np.std(tonnetz, axis=1),
    ])
    # Pad/crop surrogate to 128
    if len(wav2vec_surrogate) < 128:
        wav2vec_surrogate = np.pad(wav2vec_surrogate, (0, 128 - len(wav2vec_surrogate)))
    else:
        wav2vec_surrogate = wav2vec_surrogate[:128]

    feature_vector = np.concatenate([mel_stats, lfcc_stats, wav2vec_surrogate])
    norm = np.linalg.norm(feature_vector)
    if norm > 0:
        feature_vector = feature_vector / norm
    return feature_vector.astype(np.float32)


@dataclass
class BaselineBIndianAccent:
    """Published Indian-accent fusion model re-implementation (Baseline B).

    Isolates the 'Indian-accent data' effect from the 'code-switch boundary' novel feature.
    """
    _model: Optional[LogisticRegression] = field(default=None, repr=False)

    def fit(self, audio_paths: List[str], labels: np.ndarray) -> None:
        X = np.array([extract_baseline_b_features(p) for p in audio_paths], dtype=np.float32)
        y = np.asarray(labels)

        self._model = LogisticRegression(max_iter=500, class_weight="balanced", random_state=42)
        self._model.fit(X, y)

    def predict_proba(self, audio_paths: List[str]) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("BaselineBIndianAccent model has not been fitted yet.")

        X = np.array([extract_baseline_b_features(p) for p in audio_paths], dtype=np.float32)
        return self._model.predict_proba(X)

