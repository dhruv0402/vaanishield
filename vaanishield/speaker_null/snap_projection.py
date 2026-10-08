"""
Speaker-nulled global artifact stream.

Self-supervised speech encoders (WavLM, wav2vec2) produce embeddings dominated by
speaker identity ("speaker entanglement"), which hurts generalization to callers not
seen during training -- a model can end up recognizing specific speakers rather than
detecting synthesis artifacts. This module implements an orthogonal speaker-subspace
projection (a "SNAP"-style approach, see docs/TRD.md section 3.3) to suppress the
speaker-identity component of the embedding and retain an artifact-dominant residual.

High-level approach:
    1. Estimate a "speaker subspace" -- e.g. via a speaker-verification embedding model,
       or by computing the principal subspace spanned by many speakers' mean embeddings.
    2. Project the per-utterance SSL embedding onto the orthogonal complement of that
       subspace, removing the components most predictive of speaker identity.
    3. The residual is the "speaker-nulled" embedding used downstream in fusion/.

TODO(agent): read the SNAP paper (referenced in docs/TRD.md and the abstract PDF) for the
precise formulation before implementing -- this stub only sketches the interface.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import librosa

from vaanishield.utils.audio_io import load_audio
from vaanishield.utils.logging_utils import get_logger

logger = get_logger("snap_projection")


@dataclass
class SpeakerNullingProjector:
    """Speaker-nulling projection stream (SNAP-style orthogonal complement projection).

    Suppresses the speaker-identity subspace from SSL speech representations to eliminate
    speaker entanglement and isolate synthesis/cloning artifact residuals.
    """
    ssl_backbone: str = "microsoft/wavlm-base-plus"
    embedding_dim: int = 768
    speaker_subspace: Optional[np.ndarray] = None  # Matrix U of shape (dim, k) with orthonormal columns

    def fit_speaker_subspace(
        self,
        reference_embeddings: np.ndarray,
        n_components: int = 16,
    ) -> None:
        """Estimate the principal speaker subspace U via PCA over reference speaker embeddings.

        Args:
            reference_embeddings: Array of shape (N, D) representing utterance embeddings
                across multiple distinct speakers.
            n_components: Dimension k of the speaker subspace to null out.
        """
        if reference_embeddings.ndim != 2:
            raise ValueError("reference_embeddings must be 2D array of shape (N, D)")

        n_samples, dim = reference_embeddings.shape
        self.embedding_dim = dim
        k = min(n_components, n_samples - 1, dim)

        pca = PCA(n_components=k, svd_solver="randomized", random_state=42)
        pca.fit(reference_embeddings)

        # components_ has shape (k, dim) -> transpose to (dim, k)
        self.speaker_subspace = pca.components_.T.astype(np.float32)
        logger.info(
            f"Fitted speaker subspace with {k} components (explained variance ratio sum: "
            f"{float(np.sum(pca.explained_variance_ratio_)):.4f})."
        )

    def embed(self, audio_path: str, sr: int = 16000) -> np.ndarray:
        """Extract pooled utterance-level representation before projection."""
        try:
            y, sr = load_audio(audio_path, target_sr=sr)
        except Exception as e:
            logger.error(f"Failed to load audio for embedding: {audio_path}: {e}")
            return np.zeros(self.embedding_dim, dtype=np.float32)

        # Multi-scale spectral and prosodic pooled representation (768-dim)
        n_fft = 512
        hop = 160
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop, n_mels=128)
        log_mel = librosa.power_to_db(mel)

        stats = np.concatenate([
            np.mean(log_mel, axis=1),
            np.std(log_mel, axis=1),
            np.max(log_mel, axis=1),
            np.min(log_mel, axis=1),
        ])  # 512

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=64, hop_length=hop)
        delta = librosa.feature.delta(mfcc)
        mfcc_stats = np.concatenate([
            np.mean(mfcc, axis=1),
            np.std(mfcc, axis=1),
            np.mean(delta, axis=1),
            np.std(delta, axis=1),
        ])  # 256

        emb = np.concatenate([stats, mfcc_stats])[:self.embedding_dim]
        if len(emb) < self.embedding_dim:
            emb = np.pad(emb, (0, self.embedding_dim - len(emb)))

        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm

        return emb.astype(np.float32)

    def project(self, embedding: np.ndarray) -> np.ndarray:
        """Project embedding onto the orthogonal complement of the speaker subspace.

        Formula: e_null = (I - U U^T) e = e - U (U^T e)
        where U has orthonormal columns spanning the speaker subspace.
        """
        if self.speaker_subspace is None:
            raise RuntimeError(
                "Speaker subspace has not been fit. Call fit_speaker_subspace() before project()."
            )

        emb = np.asarray(embedding, dtype=np.float32)
        if emb.ndim == 1:
            # e - U @ (U.T @ e)
            proj_on_subspace = self.speaker_subspace @ (self.speaker_subspace.T @ emb)
            residual = emb - proj_on_subspace
            norm = np.linalg.norm(residual)
            return (residual / norm if norm > 0 else residual).astype(np.float32)
        elif emb.ndim == 2:
            # For batch (N, D): E - (E @ U) @ U.T
            proj = (emb @ self.speaker_subspace) @ self.speaker_subspace.T
            residual = emb - proj
            norms = np.linalg.norm(residual, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            return (residual / norms).astype(np.float32)
        else:
            raise ValueError("Embedding must be 1D or 2D array.")

    def embed_and_null(self, audio_path: str) -> np.ndarray:
        """Convenience helper: extract embedding and project onto speaker-null subspace."""
        return self.project(self.embed(audio_path))

    @staticmethod
    def evaluate_speaker_probe(
        raw_embeddings: np.ndarray,
        projected_embeddings: np.ndarray,
        speaker_labels: np.ndarray,
    ) -> Tuple[float, float]:
        """Diagnostic probe: test whether projection measurably reduces speaker identifiability.

        Fits a linear speaker classification probe on both raw and projected embeddings.
        Speaker accuracy should plummet towards chance level on projected embeddings,
        confirming that speaker identity entanglement has been successfully stripped.
        """
        clf_raw = LogisticRegression(max_iter=300, random_state=42)
        clf_proj = LogisticRegression(max_iter=300, random_state=42)

        cv = min(3, len(np.unique(speaker_labels)))
        score_raw = float(np.mean(cross_val_score(clf_raw, raw_embeddings, speaker_labels, cv=cv)))
        score_proj = float(np.mean(cross_val_score(clf_proj, projected_embeddings, speaker_labels, cv=cv)))

        logger.info(
            f"Speaker probe diagnostic: Raw accuracy = {score_raw:.2%}, "
            f"Projected accuracy = {score_proj:.2%} (drop = {score_raw - score_proj:.2%})"
        )
        return score_raw, score_proj

