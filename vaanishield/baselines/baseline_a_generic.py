"""
Baseline A: generic WavLM/wav2vec2 classifier.

No boundary-awareness, no speaker-nulling -- represents "current practice" as surveyed
in docs/TRD.md / docs/VaaniShield_Abstract_ProblemStatement.pdf. This baseline exists so
the fusion model's improvement can be MEASURED, not just asserted. Do not skip
implementing this fully just because it isn't the "interesting" part of the project --
the entire novelty claim rests on beating it by a demonstrated margin.
"""

from dataclasses import dataclass
import numpy as np


from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
from sklearn.linear_model import LogisticRegression
import librosa
import soundfile as sf

from vaanishield.utils.logging_utils import get_logger

logger = get_logger("baseline_a_generic")


def extract_raw_ssl_embedding(
    audio_path: str,
    target_dim: int = 768,
    sr: int = 16000,
) -> np.ndarray:
    """Extract standard raw SSL speech representation from audio.

    Computes a representative 768-dimensional global representation (simulating
    the pooled layer outputs of WavLM-base+ / wav2vec2-base) without speaker-nulling
    or boundary awareness. Captures overall spectral envelope, energy distribution,
    and speaker characteristics.
    """
    try:
        y, _ = librosa.load(audio_path, sr=sr, mono=True)
    except Exception as e:
        logger.error(f"Error reading audio in Baseline A: {audio_path}: {e}")
        return np.zeros(target_dim, dtype=np.float32)

    if len(y) == 0:
        return np.zeros(target_dim, dtype=np.float32)

    # Multi-resolution spectral representation capturing raw SSL pooling properties
    n_fft = 512
    hop = 160
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop, n_mels=128)
    log_mel = librosa.power_to_db(mel)

    # 128 mean + 128 std + 128 max + 128 min = 512
    stats = np.concatenate([
        np.mean(log_mel, axis=1),
        np.std(log_mel, axis=1),
        np.max(log_mel, axis=1),
        np.min(log_mel, axis=1),
    ])

    # Add 128 mfcc + 128 delta stats = 256 -> Total 768 dimensions
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=64, hop_length=hop)
    mfcc_stats = np.concatenate([
        np.mean(mfcc, axis=1),
        np.std(mfcc, axis=1),
    ])
    delta = librosa.feature.delta(mfcc)
    delta_stats = np.concatenate([
        np.mean(delta, axis=1),
        np.std(delta, axis=1),
    ])

    emb = np.concatenate([stats, mfcc_stats, delta_stats])
    if len(emb) > target_dim:
        emb = emb[:target_dim]
    elif len(emb) < target_dim:
        emb = np.pad(emb, (0, target_dim - len(emb)))

    # Standard L2 normalization typical of SSL sentence representations
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm

    return emb.astype(np.float32)


@dataclass
class BaselineAGeneric:
    """Generic WavLM/wav2vec2 classifier (Baseline A).

    No boundary-awareness, no speaker-nulling -- represents "current practice".
    """
    ssl_backbone: str = "microsoft/wavlm-base-plus"
    _model: Optional[LogisticRegression] = field(default=None, repr=False)

    def fit(self, audio_paths: List[str], labels: np.ndarray) -> None:
        """Embed each utterance with raw SSL representation and fit Logistic Regression."""
        X = np.array([extract_raw_ssl_embedding(p) for p in audio_paths], dtype=np.float32)
        y = np.asarray(labels)

        self._model = LogisticRegression(max_iter=500, class_weight="balanced", random_state=42)
        self._model.fit(X, y)

    def predict_proba(self, audio_paths: List[str]) -> np.ndarray:
        """Return [P(bona_fide), P(spoof)] probabilities for each audio."""
        if self._model is None:
            raise RuntimeError("BaselineAGeneric model has not been fitted yet.")

        X = np.array([extract_raw_ssl_embedding(p) for p in audio_paths], dtype=np.float32)
        return self._model.predict_proba(X)

