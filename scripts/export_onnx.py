#!/usr/bin/env python3
"""
Export VaaniShield hierarchical multi-stream fusion classifier to optimized ONNX format.
Performs numerical parity verification against scikit-learn and benchmarks CPU inference latency.
"""
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
        snap_embeddings: (batch_size, 768)
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
        # Platt sigmoid calibration: 1 / (1 + exp(a * df + b))
        b_probs_folds = 1.0 / (1.0 + torch.exp(self.b_cal_a * b_df + self.b_cal_b))  # (N, K)
        b_prob = torch.mean(b_probs_folds, dim=1, keepdim=True)  # (N, 1)

        # 2. SSL Stream: compute fold decision functions, sigmoid calibrate, average folds
        ssl_df = torch.matmul(snap_embeddings, self.ssl_coefs.t()) + self.ssl_intercepts  # (N, K)
        ssl_probs_folds = 1.0 / (1.0 + torch.exp(self.ssl_cal_a * ssl_df + self.ssl_cal_b))  # (N, K)
        ssl_prob = torch.mean(ssl_probs_folds, dim=1, keepdim=True)  # (N, 1)

        # 3. Hierarchical late fusion
        p_spoof = self.w_ssl * ssl_prob + self.w_b * b_prob
        return p_spoof


def export_and_benchmark(
    checkpoint_path: str = "checkpoints/fusion_checkpoint.pkl",
    output_dir: str = "checkpoints/onnx",
    n_iterations: int = 200,
):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    onnx_file = str(out_path / "vaanishield_fusion.onnx")

    logger.info(f"Loading checkpoint from {checkpoint_path}...")
    data = FusionClassifier.load_checkpoint(checkpoint_path, verify_checksum=True)

    model: FusionClassifier = data["model"]
    projector = data["projector"]

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

    torch_wrapper = TorchHierarchicalFusionWrapper(
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
    torch_wrapper.eval()

    # Numerical parity verification with dummy test batches
    np.random.seed(42)
    test_b = np.random.randn(5, 10).astype(np.float32)
    test_s = np.random.randn(5, 768).astype(np.float32)
    test_s /= np.linalg.norm(test_s, axis=1, keepdims=True)

    expected_probs = model.predict_proba(test_b, test_s)[:, 1]
    with torch.no_grad():
        torch_probs = torch_wrapper(torch.from_numpy(test_b), torch.from_numpy(test_s)).squeeze(1).numpy()

    max_torch_diff = np.max(np.abs(expected_probs - torch_probs))
    logger.info(f"PyTorch vs Scikit-Learn parity: max diff = {max_torch_diff:.2e}")
    assert max_torch_diff < 1e-4, f"Parity mismatch between Torch and SKLearn: {max_torch_diff}"

    # Export to ONNX
    logger.info("Exporting to ONNX...")
    dummy_b = torch.randn(1, 10, dtype=torch.float32)
    dummy_s = torch.randn(1, 768, dtype=torch.float32)

    torch.onnx.export(
        torch_wrapper,
        (dummy_b, dummy_s),
        onnx_file,
        input_names=["boundary_features", "snap_embeddings"],
        output_names=["spoof_probability"],
        dynamic_axes={
            "boundary_features": {0: "batch_size"},
            "snap_embeddings": {0: "batch_size"},
            "spoof_probability": {0: "batch_size"},
        },
        opset_version=18,
    )
    logger.info(f"ONNX model saved successfully to {onnx_file} ({os.path.getsize(onnx_file) / 1024:.1f} KB).")

    # Benchmark & verify ONNX Runtime
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_options.intra_op_num_threads = 2
    session = ort.InferenceSession(onnx_file, session_options, providers=["CPUExecutionProvider"])

    onnx_out = session.run(None, {"boundary_features": test_b, "snap_embeddings": test_s})[0].squeeze(1)
    max_onnx_diff = np.max(np.abs(expected_probs - onnx_out))
    logger.info(f"ONNX Runtime vs Scikit-Learn parity: max diff = {max_onnx_diff:.2e}")
    assert max_onnx_diff < 1e-4, f"Parity mismatch between ONNX and SKLearn: {max_onnx_diff}"

    # Latency Warmup
    single_b = test_b[:1]
    single_s = test_s[:1]
    for _ in range(20):
        _ = session.run(None, {"boundary_features": single_b, "snap_embeddings": single_s})

    latencies_ms = []
    for _ in range(n_iterations):
        t0 = time.perf_counter()
        _ = session.run(None, {"boundary_features": single_b, "snap_embeddings": single_s})
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

    median_lat = float(np.median(latencies_ms))
    p95_lat = float(np.percentile(latencies_ms, 95))
    p99_lat = float(np.percentile(latencies_ms, 99))
    logger.info(f"ONNX Runtime CPU Latency (N={n_iterations}):")
    logger.info(f"  Median: {median_lat:.3f} ms | P95: {p95_lat:.3f} ms | P99: {p99_lat:.3f} ms")

    benchmark_report = {
        "model": "VaaniShield Hierarchical Late Fusion ONNX",
        "size_kb": round(os.path.getsize(onnx_file) / 1024, 2),
        "onnx_runtime_version": ort.__version__,
        "median_latency_ms": round(median_lat, 3),
        "p95_latency_ms": round(p95_lat, 3),
        "p99_latency_ms": round(p99_lat, 3),
        "inferences_per_second": round(1000.0 / median_lat, 1),
        "parity_max_diff": float(max_onnx_diff),
    }

    with open(out_path / "latency_benchmark.json", "w") as f:
        json.dump(benchmark_report, f, indent=2)

    return benchmark_report


if __name__ == "__main__":
    export_and_benchmark()
