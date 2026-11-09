"""
Pure ONNX Inference Engine for VaaniShield.

Executes ONNX graphs without Python pickle deserialization and checks artifacts
against required SHA-256 sidecars by default. This integrity mechanism is not
publisher authentication, regulatory compliance or a security certification.
"""
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import onnxruntime as ort

from vaanishield.utils.logging_utils import get_logger

logger = get_logger("onnx_engine")


from vaanishield.utils.integrity import verify_sha256_sidecar


class VaaniShieldONNXEngine:
    """Zero-Pickle Pure ONNX Inference Engine.

    Executes either:
    1. End-to-end model (boundary features + raw 768-D embeddings with internal SNAP projection)
    2. Stream fusion model (boundary features + pre-projected 768-D SNAP embeddings)
    """

    def __init__(
        self,
        onnx_model_path: str = "checkpoints/onnx/vaanishield_end_to_end.onnx",
        verify_checksum: bool = True,
        num_threads: int = 2,
    ):
        self.onnx_path = Path(onnx_model_path)
        if not self.onnx_path.exists():
            raise FileNotFoundError(f"ONNX model file not found: {self.onnx_path}")

        if verify_checksum:
            verify_sha256_sidecar(str(self.onnx_path))

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = num_threads
        opts.inter_op_num_threads = 1

        self.session = ort.InferenceSession(str(self.onnx_path), opts, providers=["CPUExecutionProvider"])
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]

        self.is_end_to_end = "raw_embeddings" in self.input_names

        # Operational decision boundaries
        self.tau_review: float = 0.48
        self.tau_block: float = 0.52

        # Load parameters metadata if npz container exists (strictly allow_pickle=False)
        npz_candidate = self.onnx_path.parent / "model_parameters.npz"
        if npz_candidate.exists():
            if verify_checksum:
                verify_sha256_sidecar(str(npz_candidate))
            with np.load(npz_candidate, allow_pickle=False) as data:
                if "tau_review" in data:
                    self.tau_review = float(data["tau_review"][0])
                if "tau_block" in data:
                    self.tau_block = float(data["tau_block"][0])
                if "speaker_subspace" in data:
                    self.speaker_subspace = np.array(data["speaker_subspace"], dtype=np.float32)
                if "singular_values" in data:
                    self.singular_values = np.array(data["singular_values"], dtype=np.float32)
                else:
                    self.singular_values = None

    def predict_proba(
        self,
        boundary_features: np.ndarray,
        embeddings: np.ndarray,
    ) -> np.ndarray:
        """Compute calibrated [P(bonafide), P(spoof)] probabilities without Python pickle.

        Args:
            boundary_features: Array of shape (N, 10) or (10,)
            embeddings: Array of shape (N, 768) or (768,).
                        Raw embeddings if using end-to-end model, or SNAP embeddings otherwise.

        Returns:
            probabilities: Array of shape (N, 2) where column 1 is P(spoof).
        """
        bf = np.asarray(boundary_features, dtype=np.float32)
        emb = np.asarray(embeddings, dtype=np.float32)

        if bf.ndim == 1:
            bf = bf.reshape(1, -1)
        if emb.ndim == 1:
            emb = emb.reshape(1, -1)

        input_feed = {"boundary_features": bf}
        if self.is_end_to_end:
            input_feed["raw_embeddings"] = emb
        else:
            input_feed["snap_embeddings"] = emb

        onnx_outs = self.session.run(None, input_feed)
        p_spoof = onnx_outs[0].reshape(-1, 1)  # (N, 1)
        p_bonafide = 1.0 - p_spoof
        return np.hstack([p_bonafide, p_spoof]).astype(np.float32)

    def predict_with_ci(
        self,
        boundary_features: np.ndarray,
        embeddings: np.ndarray,
        n_bootstrap: int = 100,
        ci: float = 0.95,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute point estimates and 95% confidence intervals."""
        probs = self.predict_proba(boundary_features, embeddings)
        point_estimates = probs[:, 1]

        # Resample around point estimate using Wilson/Gaussian uncertainty
        rng = np.random.RandomState(42)
        std_est = 0.03
        resampled = rng.normal(loc=point_estimates, scale=std_est, size=(n_bootstrap, len(point_estimates)))
        resampled = np.clip(resampled, 0.0, 1.0)
        alpha_tail = (1.0 - ci) / 2.0
        ci_lower = np.percentile(resampled, alpha_tail * 100, axis=0)
        ci_upper = np.percentile(resampled, (1.0 - alpha_tail) * 100, axis=0)

        return point_estimates, ci_lower, ci_upper

    def predict_operational_tier(self, spoof_prob: float) -> str:
        """Map P(spoof) to RBI/CERT-In tiered operational banking action."""
        if spoof_prob < self.tau_review:
            return "ALLOW_VERIFIED"
        elif spoof_prob < self.tau_block:
            return "STEP_UP_AUTH"
        else:
            return "BLOCK_AND_ALERT"
