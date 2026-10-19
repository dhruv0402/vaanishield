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
from typing import Optional, Tuple, List, Any, Union
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
    Supports Adaptive Regularized Subspace Projection (ARS-SNAP) via Tikhonov soft-shrinkage
    to prevent over-nulling on out-of-domain commercial synthesizers and unseen speakers.
    """
    ssl_backbone: str = "microsoft/wavlm-base-plus"
    embedding_dim: int = 768
    speaker_subspace: Optional[np.ndarray] = None  # Matrix U of shape (dim, k) with orthonormal columns
    singular_values: Optional[np.ndarray] = None  # Singular values s of shape (k,)
    explained_variance_ratio: Optional[np.ndarray] = None
    default_regularization: float = 0.05

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
        self.singular_values = pca.singular_values_.astype(np.float32)
        self.explained_variance_ratio = pca.explained_variance_ratio_.astype(np.float32)
        logger.info(
            f"Fitted speaker subspace with {k} components across {n_pts} speaker points "
            f"(explained variance ratio sum: {float(np.sum(pca.explained_variance_ratio_)):.4f})."
        )

    def get_effective_subspace(self, regularization: Optional[float] = None) -> np.ndarray:
        """Compute the regularized subspace projection matrix U_eff with Tikhonov shrinkage.

        Mathematical formulation:
            w_i = \\sigma_i^2 / (\\sigma_i^2 + \\lambda \\cdot \\sigma_0^2)
            U_eff = U \\cdot \\text{diag}(\\sqrt{w_i})
        Such that (e @ U_eff) @ U_eff^T = \\sum_{i} w_i (e \\cdot u_i) u_i.
        When lambda == 0.0, w_i = 1 for all i (exact hard orthogonal nulling).
        When lambda > 0.0, weaker singular components that overfit to training vocoders
        are smoothly attenuated, protecting synthesis boundary cues on unseen speakers.
        """
        if self.speaker_subspace is None:
            raise RuntimeError(
                "Speaker subspace has not been fit. Call fit_speaker_subspace() before project()."
            )
        reg = self.default_regularization if regularization is None else float(regularization)
        if reg > 0.0 and self.singular_values is not None and len(self.singular_values) > 0:
            sigma_sq = self.singular_values ** 2
            scale_ref = float(sigma_sq[0]) if sigma_sq[0] > 0 else 1.0
            filter_weights = (sigma_sq / (sigma_sq + reg * scale_ref)).astype(np.float32)
            return (self.speaker_subspace * np.sqrt(filter_weights)).astype(np.float32)
        return self.speaker_subspace

    def embed(self, audio_input: Any, sr: int = 16000) -> np.ndarray:
        """Extract pooled utterance-level representation before projection."""
        try:
            if isinstance(audio_input, np.ndarray):
                y = audio_input
            else:
                y, sr = load_audio(str(audio_input), target_sr=sr)
        except Exception as e:
            logger.error(f"Failed to load audio for embedding: {audio_input}: {e}")
            return np.zeros(self.embedding_dim, dtype=np.float32)

        # Multi-resolution representation capturing SSL speech pooling properties
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

        emb = np.concatenate([stats, mfcc_stats, delta_stats])[:self.embedding_dim]
        if len(emb) < self.embedding_dim:
            emb = np.pad(emb, (0, self.embedding_dim - len(emb)))

        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm

        return emb.astype(np.float32)

    def project(self, embedding: np.ndarray, regularization: Optional[float] = None) -> np.ndarray:
        """Project embedding onto the orthogonal complement of the speaker subspace.

        Formula: e_null = (I - U_eff U_eff^T) e = e - U_eff (U_eff^T e)
        where U_eff is either the exact orthonormal basis (hard SNAP) or regularized
        shrinkage basis (ARS-SNAP).
        """
        U_eff = self.get_effective_subspace(regularization=regularization)
        emb = np.asarray(embedding, dtype=np.float32)
        if emb.ndim == 1:
            proj_on_subspace = U_eff @ (U_eff.T @ emb)
            residual = emb - proj_on_subspace
            norm = np.linalg.norm(residual)
            return (residual / norm if norm > 0 else residual).astype(np.float32)
        elif emb.ndim == 2:
            proj = (emb @ U_eff) @ U_eff.T
            residual = emb - proj
            norms = np.linalg.norm(residual, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            return (residual / norms).astype(np.float32)
        else:
            raise ValueError("Embedding must be 1D or 2D array.")

    def embed_and_null(self, audio_path: str, regularization: Optional[float] = None) -> np.ndarray:
        """Convenience helper: extract embedding and project onto speaker-null subspace."""
        return self.project(self.embed(audio_path), regularization=regularization)

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

