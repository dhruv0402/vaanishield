r"""
Speaker-nulled global artifact stream.

Self-supervised and acoustic speech encoders produce embeddings heavily influenced by
speaker identity ("speaker entanglement"), which hurts generalization to unseen callers:
a classifier risks memorizing training speaker traits rather than generalizing to synthesis
artifacts. This module implements orthogonal speaker-subspace projection (SNAP: Speaker-Nulling
Artifact Projection) to project speech representations onto the orthogonal complement of the
between-speaker subspace, stripping identity and isolating residual acoustic synthesis artifacts.

Mathematical formulation:
    1. Given bona-fide training speakers s in S, compute speaker mean centroids:
       \bar{e}_s = (1 / N_s) \sum_{i \in s} e_i
    2. Compute the top-k principal components U \in R^{D \times k} of between-speaker covariance:
       U = argmax \sum_s ||U^T (\bar{e}_s - \mu)||^2, s.t. U^T U = I
    3. Project any test utterance embedding e onto the orthogonal complement:
       e_null = (I - U U^T) e = e - U (U^T e)
    4. Normalize e_null to unit length for downstream fusion classification.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, List
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

    Suppresses the speaker-identity subspace from speech representations to eliminate
    speaker entanglement and isolate synthesis/cloning artifact residuals.
    """
    ssl_backbone: str = "microsoft/wavlm-base-plus"
    embedding_dim: int = 768
    speaker_subspace: Optional[np.ndarray] = None  # Matrix U of shape (dim, k) with orthonormal columns

    def fit_speaker_subspace(
        self,
        reference_embeddings: np.ndarray,
        speaker_ids: Optional[List[str]] = None,
        n_components: int = 16,
    ) -> None:
        """Estimate the principal speaker subspace U via PCA over speaker centroid representations.

        When speaker_ids are supplied, computes speaker-level centroid mean embeddings
        bar{e}_s = (1/N_s) sum_{i in s} e_i to prevent within-utterance noise, channel artifacts,
        or class variance from contaminating the speaker identity subspace.

        Args:
            reference_embeddings: Array of shape (N, D) representing utterance embeddings.
            speaker_ids: Optional list of speaker IDs corresponding to each row.
            n_components: Dimension k of the speaker subspace to null out.
        """
        if reference_embeddings.ndim != 2:
            raise ValueError("reference_embeddings must be 2D array of shape (N, D)")

        n_samples, dim = reference_embeddings.shape
        self.embedding_dim = dim

        if speaker_ids is not None and len(speaker_ids) == n_samples:
            unique_speakers = list(dict.fromkeys(speaker_ids))
            speaker_centroids = []
            for spk in unique_speakers:
                mask = [s == spk for s in speaker_ids]
                spk_embs = reference_embeddings[mask]
                speaker_centroids.append(np.mean(spk_embs, axis=0))
            fit_matrix = np.array(speaker_centroids, dtype=np.float32)
        else:
            fit_matrix = reference_embeddings

        n_pts, d = fit_matrix.shape
        k = max(1, min(n_components, n_pts - 1, d))

        pca = PCA(n_components=k, svd_solver="randomized", random_state=42)
        pca.fit(fit_matrix)

        # components_ has shape (k, dim) -> transpose to (dim, k)
        self.speaker_subspace = pca.components_.T.astype(np.float32)
        logger.info(
            f"Fitted speaker subspace with {k} components across {n_pts} speaker points "
            f"(explained variance ratio sum: {float(np.sum(pca.explained_variance_ratio_)):.4f})."
        )

    def embed(self, audio_path: str, sr: int = 16000) -> np.ndarray:
        """Extract pooled utterance-level representation before projection."""
        try:
            y, sr = load_audio(audio_path, target_sr=sr)
        except Exception as e:
            logger.error(f"Failed to load audio for embedding: {audio_path}: {e}")
            return np.zeros(self.embedding_dim, dtype=np.float32)

        # Multi-scale Cepstral Mean Normalized (CMN) temporal modulation representation (768-dim)
        # Subtracting the utterance-level mean spectrum (CMN) removes static channel EQ
        # and static speaker vocal-tract offsets before SNAP orthogonal subspace projection.
        n_fft = 512
        hop = 160
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop, n_mels=128, fmin=60.0, fmax=7600.0)
        log_mel = librosa.power_to_db(mel)
        cmn_mel = log_mel - np.mean(log_mel, axis=1, keepdims=True)
        d1_mel = librosa.feature.delta(cmn_mel)
        d2_mel = librosa.feature.delta(cmn_mel, order=2)

        stats = np.concatenate([
            np.std(cmn_mel, axis=1),
            np.percentile(cmn_mel, 90, axis=1) - np.percentile(cmn_mel, 10, axis=1),
            np.std(d1_mel, axis=1),
            np.std(d2_mel, axis=1),
        ])  # 512

        mfcc = librosa.feature.mfcc(S=log_mel, n_mfcc=64)
        cmn_mfcc = mfcc - np.mean(mfcc, axis=1, keepdims=True)
        delta = librosa.feature.delta(cmn_mfcc)
        delta2 = librosa.feature.delta(cmn_mfcc, order=2)
        mfcc_stats = np.concatenate([
            np.std(cmn_mfcc, axis=1),
            np.percentile(cmn_mfcc, 85, axis=1) - np.percentile(cmn_mfcc, 15, axis=1),
            np.std(delta, axis=1),
            np.std(delta2, axis=1),
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

