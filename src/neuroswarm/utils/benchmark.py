"""
ONNX Runtime Latency & Throughput Profiler with Automatic CUDA DLL Linking.

Benchmarks model inference latency (P50, P95, P99, Mean) and throughput (FPS)
across CUDA and CPU execution providers.
"""

import argparse
import os
from pathlib import Path
import sys
import time
from typing import Dict, Any, List

import numpy as np

# Dynamically link PyTorch's CUDA DLLs to Windows PATH before ONNX Runtime imports
try:
    import torch
    if os.name == "nt" and torch.cuda.is_available():
        torch_lib_path = os.path.join(os.path.dirname(torch.__file__), "lib")
        if os.path.exists(torch_lib_path):
            os.add_dll_directory(torch_lib_path)
            os.environ["PATH"] = torch_lib_path + os.path.pathsep + os.environ.get("PATH", "")
except ImportError:
    pass

try:
    import onnxruntime as ort
except ImportError:
    raise ImportError(
        "onnxruntime is required for benchmarking. Install via 'pip install onnxruntime-gpu' or 'pip install onnxruntime'."
    )


def benchmark_onnx_model(
    model_path: str,
    batch_size: int = 1,
    iterations: int = 1000,
    warmup: int = 100,
    device: str = "cuda"
) -> Dict[str, Any]:
    """
    Measures microsecond-level ONNX inference performance across CUDA and CPU providers.
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"ONNX model file not found at: {path.resolve()}")

    # Determine execution provider priorities
    available_providers = ort.get_available_providers()
    if device.lower() in ("cuda", "gpu") and "CUDAExecutionProvider" in available_providers:
        providers = [
            ("CUDAExecutionProvider", {
                "device_id": 0,
                "arena_extend_strategy": "kNextPowerOfTwo",
                "gpu_mem_limit": 2 * 1024 * 1024 * 1024,
                "cudnn_conv_algo_search": "EXHAUSTIVE",
            }),
            "CPUExecutionProvider"
        ]
    else:
        providers = ["CPUExecutionProvider"]

    # Initialize ONNX Session
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(path), session_options, providers=providers)

    # Detect actual active provider used by ORT engine after initialization
    active_providers = session.get_providers()
    primary_provider = active_providers[0] if active_providers else "UnknownProvider"

    # Resolve input node metadata
    input_node = session.get_inputs()[0]
    input_name = input_node.name
    input_shape = list(input_node.shape)

    # Handle dynamic batch dimensions
    if isinstance(input_shape[0], str) or input_shape[0] is None or input_shape[0] < 1:
        input_shape[0] = batch_size
    else:
        batch_size = input_shape[0]

    # Map tensor data types
    dtype_map = {
        "tensor(float)": np.float32,
        "tensor(float16)": np.float16,
        "tensor(double)": np.float64,
    }
    input_dtype = dtype_map.get(input_node.type, np.float32)

    # Instantiate dummy input array
    dummy_input = {input_name: np.random.randn(*input_shape).astype(input_dtype)}

    print("=" * 65)
    print(">>> ONNX Model Benchmark Initializing...")
    print(f"Model Path:       {path.name}")
    print(f"Active Provider:  {primary_provider}")
    print(f"Input Shape:      {input_shape} | Dtype: {input_dtype.__name__}")
    print(f"Warmup Runs:      {warmup} | Benchmark Runs: {iterations}")
    print("=" * 65)

    # Warmup Phase (allocates GPU memory arena and compiles execution kernels)
    for _ in range(warmup):
        session.run(None, dummy_input)

    # Timed Performance Loop
    latencies_ms: List[float] = []
    total_start = time.perf_counter()
    for _ in range(iterations):
        t0 = time.perf_counter()
        session.run(None, dummy_input)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)
    total_elapsed = time.perf_counter() - total_start

    # Performance Metrics
    latencies_np = np.array(latencies_ms)
    mean_latency = float(np.mean(latencies_np))
    std_latency = float(np.std(latencies_np))
    p50_latency = float(np.percentile(latencies_np, 50))
    p95_latency = float(np.percentile(latencies_np, 95))
    p99_latency = float(np.percentile(latencies_np, 99))

    total_samples = iterations * batch_size
    throughput_fps = total_samples / total_elapsed

    metrics = {
        "model": path.name,
        "provider": primary_provider,
        "batch_size": batch_size,
        "mean_latency_ms": mean_latency,
        "std_latency_ms": std_latency,
        "p50_latency_ms": p50_latency,
        "p95_latency_ms": p95_latency,
        "p99_latency_ms": p99_latency,
        "throughput_fps": throughput_fps,
    }

    print("\n--- Performance Results ---")
    print(f"  Batch Size:         {batch_size}")
    print(f"  Throughput (FPS):   {throughput_fps:,.2f} samples/sec")
    print(f"  Mean Latency:       {mean_latency:.4f} ms (± {std_latency:.4f})")
    print(f"  Median (P50):       {p50_latency:.4f} ms")
    print(f"  95th Percentile:    {p95_latency:.4f} ms")
    print(f"  99th Percentile:    {p99_latency:.4f} ms")
    print("=" * 65)

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ONNX Latency and Throughput Profiler")
    parser.add_argument("--model", type=str, required=True, help="Path to exported .onnx model file")
    parser.add_argument("--batch_size", type=int, default=1, help="Inference batch size")
    parser.add_argument("--iterations", type=int, default=1000, help="Number of benchmark iterations")
    parser.add_argument("--warmup", type=int, default=100, help="Warmup runs before timing")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Execution device target")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    benchmark_onnx_model(
        model_path=args.model,
        batch_size=args.batch_size,
        iterations=args.iterations,
        warmup=args.warmup,
        device=args.device,
    )
