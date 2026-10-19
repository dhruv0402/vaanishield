#!/usr/bin/env python3
"""
Export VaaniShield hierarchical multi-stream fusion classifier and SNAP projector to ONNX format.
Eliminates Python pickle deserialization risks by exporting pure ONNX computational graphs
and zero-pickle parameter arrays with cryptographic SHA-256 integrity digests.
"""
import hashlib
import json
import os
import time
from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn

from vaanishield.fusion.classifier import FusionClassifier
from vaanishield.utils.logging_utils import get_logger

logger = get_logger("export_onnx")


class TorchHierarchicalFusionWrapper(nn.Module):
    """
    PyTorch replica of FusionClassifier's hierarchical late fusion logic.
    Inputs:
        boundary_features: (batch_size, 10)
        snap_embeddings: (batch_size, 768) - already speaker-nulled
    Outputs:
        spoof_probability: (batch_size, 1) calibrated P(spoof)
    """
    def __init__(
        self,
        b_mean: np.ndarray,
        b_scale: np.ndarray,
        b_coefs: np.ndarray,
        b_intercepts: np.ndarray,
        b_cal_a: np.ndarray,
        b_cal_b: np.ndarray,
        ssl_coefs: np.ndarray,
        ssl_intercepts: np.ndarray,
        ssl_cal_a: np.ndarray,
        ssl_cal_b: np.ndarray,
        weight_ssl: float = 0.75,
        weight_boundary: float = 0.25,
    ):
        super().__init__()
        # Boundary stream parameters
        self.b_mean = nn.Parameter(torch.from_numpy(b_mean).float(), requires_grad=False)
        self.b_scale = nn.Parameter(torch.from_numpy(b_scale).float(), requires_grad=False)
        self.b_coefs = nn.Parameter(torch.from_numpy(b_coefs).float(), requires_grad=False)  # (K, 10)
        self.b_intercepts = nn.Parameter(torch.from_numpy(b_intercepts).float(), requires_grad=False)  # (K,)
        self.b_cal_a = nn.Parameter(torch.from_numpy(b_cal_a).float(), requires_grad=False)  # (K,)
        self.b_cal_b = nn.Parameter(torch.from_numpy(b_cal_b).float(), requires_grad=False)  # (K,)

        # SSL stream parameters (no scaler - unit-norm geometry preserved)
        self.ssl_coefs = nn.Parameter(torch.from_numpy(ssl_coefs).float(), requires_grad=False)  # (K, 768)
        self.ssl_intercepts = nn.Parameter(torch.from_numpy(ssl_intercepts).float(), requires_grad=False)  # (K,)
        self.ssl_cal_a = nn.Parameter(torch.from_numpy(ssl_cal_a).float(), requires_grad=False)  # (K,)
        self.ssl_cal_b = nn.Parameter(torch.from_numpy(ssl_cal_b).float(), requires_grad=False)  # (K,)

        self.w_ssl = float(weight_ssl)
        self.w_b = float(weight_boundary)

    def forward(self, boundary_feats: torch.Tensor, snap_embeddings: torch.Tensor) -> torch.Tensor:
        # 1. Boundary Stream: scale, compute fold decision functions, sigmoid calibrate, average folds
        b_scaled = (boundary_feats - self.b_mean) / self.b_scale  # (N, 10)
        b_df = torch.matmul(b_scaled, self.b_coefs.t()) + self.b_intercepts  # (N, K)
        b_probs_folds = 1.0 / (1.0 + torch.exp(self.b_cal_a * b_df + self.b_cal_b))  # (N, K)
        b_prob = torch.mean(b_probs_folds, dim=1, keepdim=True)  # (N, 1)

        # 2. SSL Stream: compute fold decision functions, sigmoid calibrate, average folds
        ssl_df = torch.matmul(snap_embeddings, self.ssl_coefs.t()) + self.ssl_intercepts  # (N, K)
        ssl_probs_folds = 1.0 / (1.0 + torch.exp(self.ssl_cal_a * ssl_df + self.ssl_cal_b))  # (N, K)
        ssl_prob = torch.mean(ssl_probs_folds, dim=1, keepdim=True)  # (N, 1)

        # 3. Hierarchical late fusion
        p_spoof = self.w_ssl * ssl_prob + self.w_b * b_prob
        return p_spoof


class TorchEndToEndFusionWrapper(nn.Module):
    """
    Pure end-to-end PyTorch replica that combines SNAP Orthogonal Subspace Projection
    and Hierarchical Multi-Stream Classification into a single ONNX graph.
    Inputs:
        boundary_features: (batch_size, 10)
        raw_embeddings: (batch_size, 768) - raw pooled audio representations
    Outputs:
        spoof_probability: (batch_size, 1) calibrated P(spoof)
    """
    def __init__(
        self,
        speaker_subspace: np.ndarray,  # (768, K_sub)
        b_mean: np.ndarray,
        b_scale: np.ndarray,
        b_coefs: np.ndarray,
        b_intercepts: np.ndarray,
        b_cal_a: np.ndarray,
        b_cal_b: np.ndarray,
        ssl_coefs: np.ndarray,
        ssl_intercepts: np.ndarray,
        ssl_cal_a: np.ndarray,
        ssl_cal_b: np.ndarray,
        weight_ssl: float = 0.75,
        weight_boundary: float = 0.25,
    ):
        super().__init__()
        # SNAP projection matrix U: (768, K_sub)
        self.speaker_subspace = nn.Parameter(torch.from_numpy(speaker_subspace).float(), requires_grad=False)

        # Boundary stream parameters
        self.b_mean = nn.Parameter(torch.from_numpy(b_mean).float(), requires_grad=False)
        self.b_scale = nn.Parameter(torch.from_numpy(b_scale).float(), requires_grad=False)
        self.b_coefs = nn.Parameter(torch.from_numpy(b_coefs).float(), requires_grad=False)
        self.b_intercepts = nn.Parameter(torch.from_numpy(b_intercepts).float(), requires_grad=False)
        self.b_cal_a = nn.Parameter(torch.from_numpy(b_cal_a).float(), requires_grad=False)
        self.b_cal_b = nn.Parameter(torch.from_numpy(b_cal_b).float(), requires_grad=False)

        # SSL stream parameters
        self.ssl_coefs = nn.Parameter(torch.from_numpy(ssl_coefs).float(), requires_grad=False)
        self.ssl_intercepts = nn.Parameter(torch.from_numpy(ssl_intercepts).float(), requires_grad=False)
        self.ssl_cal_a = nn.Parameter(torch.from_numpy(ssl_cal_a).float(), requires_grad=False)
        self.ssl_cal_b = nn.Parameter(torch.from_numpy(ssl_cal_b).float(), requires_grad=False)

        self.w_ssl = float(weight_ssl)
        self.w_b = float(weight_boundary)

    def forward(self, boundary_feats: torch.Tensor, raw_embeddings: torch.Tensor) -> torch.Tensor:
        # 1. SNAP Subspace Projection: e_null = e - (e @ U) @ U^T
        proj = torch.matmul(torch.matmul(raw_embeddings, self.speaker_subspace), self.speaker_subspace.t())
        residual = raw_embeddings - proj
        norm = torch.norm(residual, p=2, dim=1, keepdim=True)
        snap_embeddings = residual / torch.clamp(norm, min=1e-8)

        # 2. Boundary Stream
        b_scaled = (boundary_feats - self.b_mean) / self.b_scale
        b_df = torch.matmul(b_scaled, self.b_coefs.t()) + self.b_intercepts
        b_probs_folds = 1.0 / (1.0 + torch.exp(self.b_cal_a * b_df + self.b_cal_b))
        b_prob = torch.mean(b_probs_folds, dim=1, keepdim=True)

        # 3. SSL Stream
        ssl_df = torch.matmul(snap_embeddings, self.ssl_coefs.t()) + self.ssl_intercepts
        ssl_probs_folds = 1.0 / (1.0 + torch.exp(self.ssl_cal_a * ssl_df + self.ssl_cal_b))
        ssl_prob = torch.mean(ssl_probs_folds, dim=1, keepdim=True)

        # 4. Hierarchical late fusion
        p_spoof = self.w_ssl * ssl_prob + self.w_b * b_prob
        return p_spoof


def compute_file_sha256(filepath: str) -> str:
    with open(filepath, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def write_sha256_sidecar(filepath: str) -> str:
    path = Path(filepath)
    digest = compute_file_sha256(str(path))
    sha_path = path.with_suffix(".sha256")
    with open(sha_path, "w", encoding="utf-8") as f:
        f.write(f"{digest}  {path.name}\n")
    return digest


def export_and_benchmark(
    checkpoint_path: str = "checkpoints/fusion_checkpoint.pkl",
    output_dir: str = "checkpoints/onnx",
    n_iterations: int = 200,
):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    onnx_fusion_file = str(out_path / "vaanishield_fusion.onnx")
    onnx_e2e_file = str(out_path / "vaanishield_end_to_end.onnx")
    npz_params_file = str(out_path / "model_parameters.npz")

    logger.info(f"Loading checkpoint from {checkpoint_path}...")
    data = FusionClassifier.load_checkpoint(checkpoint_path, verify_checksum=True)

    model: FusionClassifier = data["model"]
    projector = data["projector"]
    speaker_subspace = projector.speaker_subspace.astype(np.float32)

    # 1. Extract boundary parameters
    if model._scaler_b is not None:
        b_mean = model._scaler_b.mean_.astype(np.float32)
        b_scale = model._scaler_b.scale_.astype(np.float32)
    else:
        b_mean = np.zeros(10, dtype=np.float32)
        b_scale = np.ones(10, dtype=np.float32)

    b_clfs = getattr(model._calibrator_b, "calibrated_classifiers_", [])
    b_coefs = np.array([clf.estimator.coef_[0] for clf in b_clfs], dtype=np.float32)
    b_intercepts = np.array([clf.estimator.intercept_[0] for clf in b_clfs], dtype=np.float32)
    b_cal_a = np.array([clf.calibrators[0].a_ for clf in b_clfs], dtype=np.float32)
    b_cal_b = np.array([clf.calibrators[0].b_ for clf in b_clfs], dtype=np.float32)

    # 2. Extract SSL parameters
    ssl_clfs = getattr(model._calibrator_ssl, "calibrated_classifiers_", [])
    ssl_coefs = np.array([clf.estimator.coef_[0] for clf in ssl_clfs], dtype=np.float32)
    ssl_intercepts = np.array([clf.estimator.intercept_[0] for clf in ssl_clfs], dtype=np.float32)
    ssl_cal_a = np.array([clf.calibrators[0].a_ for clf in ssl_clfs], dtype=np.float32)
    ssl_cal_b = np.array([clf.calibrators[0].b_ for clf in ssl_clfs], dtype=np.float32)

    # 3. Export Pure Zero-Pickle NPZ Parameter Archive
    logger.info(f"Saving zero-pickle NumPy container to {npz_params_file}...")
    np.savez_compressed(
        npz_params_file,
        speaker_subspace=speaker_subspace,
        b_mean=b_mean,
        b_scale=b_scale,
        b_coefs=b_coefs,
        b_intercepts=b_intercepts,
        b_cal_a=b_cal_a,
        b_cal_b=b_cal_b,
        ssl_coefs=ssl_coefs,
        ssl_intercepts=ssl_intercepts,
        ssl_cal_a=ssl_cal_a,
        ssl_cal_b=ssl_cal_b,
        weight_ssl=np.array([model.weight_ssl], dtype=np.float32),
        weight_boundary=np.array([model.weight_boundary], dtype=np.float32),
        tau_review=np.array([getattr(model, "tau_review", 0.48)], dtype=np.float32),
        tau_block=np.array([getattr(model, "tau_block", 0.52)], dtype=np.float32),
        metadata=json.dumps({"format": "zero_pickle", "version": "2.0.0"}),
    )
    npz_sha = write_sha256_sidecar(npz_params_file)
    logger.info(f"Zero-pickle archive saved with SHA-256: {npz_sha[:12]}...")

    # 4. Instantiate PyTorch Wrappers
    fusion_wrapper = TorchHierarchicalFusionWrapper(
        b_mean=b_mean,
        b_scale=b_scale,
        b_coefs=b_coefs,
        b_intercepts=b_intercepts,
        b_cal_a=b_cal_a,
        b_cal_b=b_cal_b,
        ssl_coefs=ssl_coefs,
        ssl_intercepts=ssl_intercepts,
        ssl_cal_a=ssl_cal_a,
        ssl_cal_b=ssl_cal_b,
        weight_ssl=model.weight_ssl,
        weight_boundary=model.weight_boundary,
    )
    fusion_wrapper.eval()

    e2e_wrapper = TorchEndToEndFusionWrapper(
        speaker_subspace=speaker_subspace,
        b_mean=b_mean,
        b_scale=b_scale,
        b_coefs=b_coefs,
        b_intercepts=b_intercepts,
        b_cal_a=b_cal_a,
        b_cal_b=b_cal_b,
        ssl_coefs=ssl_coefs,
        ssl_intercepts=ssl_intercepts,
        ssl_cal_a=ssl_cal_a,
        ssl_cal_b=ssl_cal_b,
        weight_ssl=model.weight_ssl,
        weight_boundary=model.weight_boundary,
    )
    e2e_wrapper.eval()

    # Numerical Parity Verification with test batches
    np.random.seed(42)
    test_b = np.random.randn(10, 10).astype(np.float32)
    test_raw = np.random.randn(10, 768).astype(np.float32)
    test_raw /= np.linalg.norm(test_raw, axis=1, keepdims=True)
    expected_snap = projector.project(test_raw)
    expected_probs = model.predict_proba(test_b, expected_snap)[:, 1]

    with torch.no_grad():
        fusion_probs = fusion_wrapper(torch.from_numpy(test_b), torch.from_numpy(expected_snap)).squeeze(1).numpy()
        e2e_probs = e2e_wrapper(torch.from_numpy(test_b), torch.from_numpy(test_raw)).squeeze(1).numpy()

    assert np.max(np.abs(expected_probs - fusion_probs)) < 1e-4, "Torch fusion parity mismatch"
    assert np.max(np.abs(expected_probs - e2e_probs)) < 1e-4, "Torch end-to-end parity mismatch"
    logger.info("PyTorch parity verified against Scikit-Learn (< 1e-4 diff).")

    # 5. Export to ONNX
    dummy_b = torch.randn(1, 10, dtype=torch.float32)
    dummy_s = torch.randn(1, 768, dtype=torch.float32)

    logger.info(f"Exporting stream fusion ONNX to {onnx_fusion_file}...")
    torch.onnx.export(
        fusion_wrapper,
        (dummy_b, dummy_s),
        onnx_fusion_file,
        input_names=["boundary_features", "snap_embeddings"],
        output_names=["spoof_probability"],
        dynamic_axes={
            "boundary_features": {0: "batch_size"},
            "snap_embeddings": {0: "batch_size"},
            "spoof_probability": {0: "batch_size"},
        },
        opset_version=18,
    )
    write_sha256_sidecar(onnx_fusion_file)

    logger.info(f"Exporting end-to-end ONNX (SNAP projection + fusion) to {onnx_e2e_file}...")
    torch.onnx.export(
        e2e_wrapper,
        (dummy_b, dummy_s),
        onnx_e2e_file,
        input_names=["boundary_features", "raw_embeddings"],
        output_names=["spoof_probability"],
        dynamic_axes={
            "boundary_features": {0: "batch_size"},
            "raw_embeddings": {0: "batch_size"},
            "spoof_probability": {0: "batch_size"},
        },
        opset_version=18,
    )
    write_sha256_sidecar(onnx_e2e_file)

    # 6. Benchmark ONNX Runtime Session
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_options.intra_op_num_threads = 2
    session_e2e = ort.InferenceSession(onnx_e2e_file, session_options, providers=["CPUExecutionProvider"])

    onnx_out = session_e2e.run(None, {"boundary_features": test_b, "raw_embeddings": test_raw})[0].squeeze(1)
    max_onnx_diff = float(np.max(np.abs(expected_probs - onnx_out)))
    logger.info(f"ONNX Runtime End-to-End parity max diff: {max_onnx_diff:.2e}")
    assert max_onnx_diff < 1e-4, f"ONNX Runtime parity failed: {max_onnx_diff}"

    # Latency Warmup & Benchmark
    single_b = test_b[:1]
    single_raw = test_raw[:1]
    for _ in range(25):
        _ = session_e2e.run(None, {"boundary_features": single_b, "raw_embeddings": single_raw})

    latencies_ms = []
    for _ in range(n_iterations):
        t0 = time.perf_counter()
        _ = session_e2e.run(None, {"boundary_features": single_b, "raw_embeddings": single_raw})
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

    median_lat = float(np.median(latencies_ms))
    p95_lat = float(np.percentile(latencies_ms, 95))
    p99_lat = float(np.percentile(latencies_ms, 99))
    logger.info(f"ONNX Runtime End-to-End CPU Latency (N={n_iterations}):")
    logger.info(f"  Median: {median_lat:.3f} ms | P95: {p95_lat:.3f} ms | P99: {p99_lat:.3f} ms")

    benchmark_report = {
        "model": "VaaniShield End-to-End (SNAP + Late Fusion) ONNX",
        "size_kb": round(os.path.getsize(onnx_e2e_file) / 1024, 2),
        "onnx_runtime_version": ort.__version__,
        "median_latency_ms": round(median_lat, 3),
        "p95_latency_ms": round(p95_lat, 3),
        "p99_latency_ms": round(p99_lat, 3),
        "inferences_per_second": round(1000.0 / median_lat, 1),
        "parity_max_diff": max_onnx_diff,
        "zero_pickle_security": True,
        "sha256": compute_file_sha256(onnx_e2e_file),
    }

    with open(out_path / "latency_benchmark.json", "w") as f:
        json.dump(benchmark_report, f, indent=2)

    return benchmark_report


if __name__ == "__main__":
    export_and_benchmark()
