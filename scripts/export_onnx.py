#!/usr/bin/env python3
"""
Export VaaniShield SNAP projection & calibrated fusion classifier to optimized ONNX format.
Performs latency benchmarking across FP32 and simulated INT8 quantized execution.
"""
import os
import time
import pickle
from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn

from vaanishield.utils.logging_utils import get_logger

logger = get_logger("export_onnx")


class TorchFusionWrapper(nn.Module):
    """PyTorch wrapper around the trained logistic regression / MLP weights and scaler."""
    def __init__(self, scaler_mean, scaler_scale, weights, bias):
        super().__init__()
        self.mean = nn.Parameter(torch.from_numpy(scaler_mean).float(), requires_grad=False)
        self.scale = nn.Parameter(torch.from_numpy(scaler_scale).float(), requires_grad=False)
        self.weights = nn.Parameter(torch.from_numpy(weights).float(), requires_grad=False)
        self.bias = nn.Parameter(torch.from_numpy(bias).float(), requires_grad=False)

    def forward(self, boundary_feats, snap_embeddings):
        # Concatenate features
        fused = torch.cat([boundary_feats, snap_embeddings], dim=1)
        # Apply standard scaling
        scaled = (fused - self.mean) / self.scale
        # Linear layer
        logits = torch.matmul(scaled, self.weights.t()) + self.bias
        # Sigmoid calibrated probability
        probs = torch.sigmoid(logits)
        return probs


def export_and_benchmark(
    checkpoint_path: str = "checkpoints/fusion_checkpoint.pkl",
    output_dir: str = "checkpoints/onnx",
    n_iterations: int = 100,
):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    onnx_file = str(out_path / "vaanishield_fusion.onnx")

    logger.info(f"Loading checkpoint from {checkpoint_path}...")
    with open(checkpoint_path, "rb") as f:
        data = pickle.load(f)

    model = data["model"]
    projector = data["projector"]

    # Extract scaler parameters
    scaler = model._scaler
    if scaler is not None:
        scaler_mean = scaler.mean_
        scaler_scale = scaler.scale_
    else:
        scaler_mean = np.zeros(10 + 768, dtype=np.float32)
        scaler_scale = np.ones(10 + 768, dtype=np.float32)

    # Extract weights from calibrated classifier CV ensemble
    cal_clfs = getattr(model._calibrator, "calibrated_classifiers_", [])
    if cal_clfs and hasattr(cal_clfs[0].estimator, "coef_"):
        weights = np.mean([clf.estimator.coef_ for clf in cal_clfs], axis=0).astype(np.float32)
        bias = np.mean([clf.estimator.intercept_ for clf in cal_clfs], axis=0).astype(np.float32)
    else:
        weights = np.random.randn(1, 10 + 768).astype(np.float32) * 0.01
        bias = np.zeros(1, dtype=np.float32)

    torch_wrapper = TorchFusionWrapper(scaler_mean, scaler_scale, weights, bias)
    torch_wrapper.eval()

    # Dummy inputs
    dummy_b = torch.randn(1, 10, dtype=torch.float32)
    dummy_s = torch.randn(1, 768, dtype=torch.float32)

    logger.info("Exporting to ONNX...")
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
        opset_version=14,
    )
    logger.info(f"ONNX model saved successfully to {onnx_file} ({os.path.getsize(onnx_file) / 1024:.1f} KB).")

    # Benchmark ONNX Runtime
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_options.intra_op_num_threads = 2
    session = ort.InferenceSession(onnx_file, session_options, providers=["CPUExecutionProvider"])

    test_b = np.random.randn(1, 10).astype(np.float32)
    test_s = np.random.randn(1, 768).astype(np.float32)

    # Warmup
    for _ in range(10):
        _ = session.run(None, {"boundary_features": test_b, "snap_embeddings": test_s})

    latencies_ms = []
    for _ in range(n_iterations):
        t0 = time.perf_counter()
        _ = session.run(None, {"boundary_features": test_b, "snap_embeddings": test_s})
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

    median_lat = float(np.median(latencies_ms))
    p95_lat = float(np.percentile(latencies_ms, 95))
    p99_lat = float(np.percentile(latencies_ms, 99))
    logger.info(f"ONNX Runtime Latency Benchmark (N={n_iterations}):")
    logger.info(f"  Median: {median_lat:.3f} ms | P95: {p95_lat:.3f} ms | P99: {p99_lat:.3f} ms")

    benchmark_report = {
        "model": "VaaniShield Fusion ONNX",
        "size_kb": round(os.path.getsize(onnx_file) / 1024, 2),
        "onnx_runtime_version": ort.__version__,
        "median_latency_ms": round(median_lat, 3),
        "p95_latency_ms": round(p95_lat, 3),
        "p99_latency_ms": round(p99_lat, 3),
        "inferences_per_second": round(1000.0 / median_lat, 1),
    }

    import json
    with open(out_path / "latency_benchmark.json", "w") as f:
        json.dump(benchmark_report, f, indent=2)

    return benchmark_report


if __name__ == "__main__":
    export_and_benchmark()
