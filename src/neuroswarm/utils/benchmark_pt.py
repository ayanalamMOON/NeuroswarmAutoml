"""
Native PyTorch CUDA Microsecond Latency & Throughput Profiler.
"""

import argparse
import time
from pathlib import Path
import numpy as np
import torch


def benchmark_pytorch_model(
    model_path: str,
    batch_size: int = 32,
    iterations: int = 1000,
    warmup: int = 100,
    device_str: str = "cuda",
) -> None:
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    path = Path(model_path)

    print("=" * 65)
    print(">>> PyTorch CUDA Hardware Benchmark Initializing...")
    print(f"Model Path:       {path.name}")
    print(f"Target Device:    {device_str.upper()} ({torch.cuda.get_device_name(0)})")
    print(f"Batch Size:       {batch_size}")
    print("=" * 65)

    # Load TorchScript model
    model = torch.jit.load(str(path)).to(device)
    model.eval()

    dummy_input = torch.randn(batch_size, 3, 32, 32, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)
    torch.cuda.synchronize()

    # Precise GPU Timing with CUDA Events
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]

    with torch.no_grad():
        for i in range(iterations):
            start_events[i].record()
            _ = model(dummy_input)
            end_events[i].record()

    torch.cuda.synchronize()

    latencies_ms = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    latencies_np = np.array(latencies_ms)

    mean_lat = float(np.mean(latencies_np))
    p50_lat = float(np.percentile(latencies_np, 50))
    p95_lat = float(np.percentile(latencies_np, 95))
    p99_lat = float(np.percentile(latencies_np, 99))
    fps = (iterations * batch_size) / (sum(latencies_ms) / 1000.0)

    print("\n--- Native CUDA Performance Results ---")
    print(f"  Throughput (FPS):   {fps:,.2f} samples/sec")
    print(f"  Mean Latency:       {mean_lat:.4f} ms")
    print(f"  Median (P50):       {p50_lat:.4f} ms")
    print(f"  95th Percentile:    {p95_lat:.4f} ms")
    print(f"  99th Percentile:    {p99_lat:.4f} ms")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to exported .pt TorchScript model",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=100)
    args = parser.parse_args()

    benchmark_pytorch_model(args.model, args.batch_size, args.iterations, args.warmup)
