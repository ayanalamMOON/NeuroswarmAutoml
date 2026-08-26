"""
TensorRT Hardware Execution Profiler.

Profiles microsecond inference latency and sample throughput for serialized
TensorRT (.engine) binaries using PyTorch CUDA stream bindings and CUDA events.
"""

import argparse
import logging
import os
from pathlib import Path
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import torch

# Link CUDA/cuDNN/TensorRT DLLs on Windows
if os.name == "nt":
    venv_base = sys.prefix
    dll_paths = [
        os.path.join(venv_base, "Lib", "site-packages", "nvidia", "tensorrt"),
        os.path.join(venv_base, "Lib", "site-packages", "nvidia", "tensorrt_cu12_libs"),
        os.path.join(venv_base, "Lib", "site-packages", "nvidia", "tensorrt_cu13_libs"),
        os.path.join(venv_base, "Lib", "site-packages", "nvidia", "cudnn", "lib"),
        os.path.join(venv_base, "Lib", "site-packages", "nvidia", "cublas", "lib"),
        os.path.join(os.path.dirname(torch.__file__), "lib"),
    ]
    for p in dll_paths:
        if os.path.exists(p):
            try:
                os.add_dll_directory(p)
            except AttributeError:
                pass
            os.environ["PATH"] = p + os.path.pathsep + os.environ.get("PATH", "")

try:
    import tensorrt as trt
    HAS_TRT = True
except ImportError:
    HAS_TRT = False

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("neuroswarm.benchmark_trt")


class TensorRTProfiler:
    """Handles TensorRT engine execution profiling via PyTorch CUDA pointers."""

    def __init__(self, engine_path: str):
        if not HAS_TRT:
            raise RuntimeError("TensorRT Python bindings not installed. Run 'pip install tensorrt tensorrt-cu12'.")

        self.engine_path = Path(engine_path)
        if not self.engine_path.exists():
            raise FileNotFoundError(f"TensorRT engine not found: {self.engine_path.resolve()}")

        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)

        with open(self.engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())

        if self.engine is None:
            raise RuntimeError("Failed to deserialize TensorRT engine binary.")

        self.context = self.engine.create_execution_context()
        self._inspect_tensors()

    def _inspect_tensors(self):
        """Identifies input/output tensor names from the engine metadata."""
        self.input_names = []
        self.output_names = []

        if hasattr(self.engine, "num_io_tensors"):
            for i in range(self.engine.num_io_tensors):
                name = self.engine.get_tensor_name(i)
                mode = self.engine.get_tensor_mode(name)
                if mode == trt.TensorIOMode.INPUT:
                    self.input_names.append(name)
                else:
                    self.output_names.append(name)
        else:
            for i in range(self.engine.num_bindings):
                name = self.engine.get_binding_name(i)
                if self.engine.binding_is_input(i):
                    self.input_names.append(name)
                else:
                    self.output_names.append(name)

    def benchmark(
        self,
        batch_size: int = 32,
        input_shape: Tuple[int, ...] = (3, 32, 32),
        num_classes: int = 100,
        warmup: int = 100,
        iterations: int = 1000
    ) -> Dict[str, float]:
        """Runs CUDA event-based benchmark profiling on the TensorRT engine."""
        device = torch.device("cuda")
        full_input_shape = (batch_size, *input_shape)

        # Allocate PyTorch CUDA memory for inputs and outputs
        dummy_input = torch.randn(*full_input_shape, device=device, dtype=torch.float32)
        dummy_output = torch.empty((batch_size, num_classes), device=device, dtype=torch.float32)

        # Configure dynamic shapes & addresses in TRT context
        for inp_name in self.input_names:
            if hasattr(self.context, "set_input_shape"):
                self.context.set_input_shape(inp_name, full_input_shape)
            if hasattr(self.context, "set_tensor_address"):
                self.context.set_tensor_address(inp_name, dummy_input.data_ptr())

        for out_name in self.output_names:
            if hasattr(self.context, "set_tensor_address"):
                self.context.set_tensor_address(out_name, dummy_output.data_ptr())

        stream = torch.cuda.Stream()

        def run_forward():
            if hasattr(self.context, "execute_async_v3"):
                self.context.execute_async_v3(stream.cuda_stream)
            else:
                bindings = [dummy_input.data_ptr(), dummy_output.data_ptr()]
                self.context.execute_async_v2(bindings=bindings, stream_handle=stream.cuda_stream)

        # Warmup Phase
        for _ in range(warmup):
            run_forward()
        stream.synchronize()

        # Profiling Phase with CUDA Events
        latencies_ms = []
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        total_start_time = time.perf_counter()
        for _ in range(iterations):
            start_event.record(stream)
            run_forward()
            end_event.record(stream)
            stream.synchronize()

            latencies_ms.append(start_event.elapsed_time(end_event))

        total_wall_time = time.perf_counter() - total_start_time

        latencies_ms = np.array(latencies_ms)
        mean_latency = float(np.mean(latencies_ms))
        median_latency = float(np.median(latencies_ms))
        p95_latency = float(np.percentile(latencies_ms, 95))
        p99_latency = float(np.percentile(latencies_ms, 99))
        throughput_fps = (batch_size * iterations) / total_wall_time

        return {
            "throughput_fps": throughput_fps,
            "mean_latency_ms": mean_latency,
            "median_latency_ms": median_latency,
            "p95_latency_ms": p95_latency,
            "p99_latency_ms": p99_latency,
            "total_samples": batch_size * iterations,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NeuroSwarm-AutoML TensorRT Hardware Profiler")
    parser.add_argument("--engine", type=str, default="./outputs_trt/winner_4f5974bd.engine", help="Path to .engine binary")
    parser.add_argument("--batch_size", type=int, default=32, help="Inference batch size")
    parser.add_argument("--num_classes", type=int, default=100, help="Output classification classes")
    parser.add_argument("--warmup", type=int, default=100, help="Warmup iterations")
    parser.add_argument("--iterations", type=int, default=1000, help="Benchmark iterations")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    profiler = TensorRTProfiler(engine_path=args.engine)

    print("=" * 65)
    print(">>> TensorRT CUDA Engine Hardware Profiler Initializing...")
    print(f"Engine Path:   {args.engine}")
    print(f"GPU Hardware:  {torch.cuda.get_device_name(0)}")
    print(f"Batch Size:    {args.batch_size} | Iterations: {args.iterations}")
    print("=" * 65)

    metrics = profiler.benchmark(
        batch_size=args.batch_size,
        num_classes=args.num_classes,
        warmup=args.warmup,
        iterations=args.iterations
    )

    print("\n--- TensorRT Engine Execution Results ---")
    print(f"  Throughput (FPS):   {metrics['throughput_fps']:,.2f} samples/sec")
    print(f"  Mean Latency:       {metrics['mean_latency_ms']:.4f} ms")
    print(f"  Median (P50):       {metrics['median_latency_ms']:.4f} ms")
    print(f"  95th Percentile:    {metrics['p95_latency_ms']:.4f} ms")
    print(f"  99th Percentile:    {metrics['p99_latency_ms']:.4f} ms")
    print("=" * 65)
