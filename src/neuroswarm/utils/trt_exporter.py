"""
Native NVIDIA TensorRT Engine Serialization & Export Module.

Compiles ONNX computational graphs into optimized, hardware-bound TensorRT (.engine)
binaries for maximum Tensor Core execution throughput and low-latency inference.
Supports TensorRT 8.x through 11.x with dynamic shape optimization profiles.
"""

import argparse
import logging
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Dict, Any, Tuple, Optional, Union

import numpy as np

# Link CUDA/cuDNN/TensorRT DLLs on Windows environments
if os.name == "nt":
    venv_base = sys.prefix
    dll_paths = [
        os.path.join(venv_base, "Lib", "site-packages", "nvidia", "tensorrt"),
        os.path.join(venv_base, "Lib", "site-packages", "nvidia", "tensorrt_cu12_libs"),
        os.path.join(venv_base, "Lib", "site-packages", "nvidia", "tensorrt_cu13_libs"),
        os.path.join(venv_base, "Lib", "site-packages", "nvidia", "cudnn", "lib"),
        os.path.join(venv_base, "Lib", "site-packages", "nvidia", "cublas", "lib"),
        os.path.join(venv_base, "Lib", "site-packages", "torch", "lib"),
    ]
    for p in dll_paths:
        if os.path.exists(p):
            try:
                os.add_dll_directory(p)
            except AttributeError:
                pass
            os.environ["PATH"] = p + os.path.pathsep + os.environ.get("PATH", "")

# Conditional TensorRT & ONNX Runtime Imports
try:
    import tensorrt as trt

    HAS_TRT = True
except ImportError:
    HAS_TRT = False

try:
    import onnxruntime as ort

    HAS_ORT = True
except ImportError:
    HAS_ORT = False

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("neuroswarm.trt_exporter")


def get_file_size_mb(path: Union[str, Path]) -> float:
    """Returns file size in Megabytes (MB)."""
    return Path(path).stat().st_size / (1024.0 * 1024.0)


class TensorRTExporter:
    """
    Serializes ONNX models into hardware-optimized TensorRT execution engines.
    """

    def __init__(self, output_dir: str = "./outputs_trt"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_via_python_api(
        self,
        onnx_path: str,
        engine_path: str,
        use_fp16: bool = True,
        max_workspace_gb: float = 2.0,
    ) -> bool:
        """
        Builds TensorRT engine using the native TensorRT Python API.
        Includes IOptimizationProfile handling for ONNX dynamic input dimensions.
        """
        if not HAS_TRT:
            return False

        logger.info(f"Building TensorRT engine via Python API (v{trt.__version__})...")
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(TRT_LOGGER)

        # Handle explicit batch flag removal across TRT versions
        network_flags = 0
        if hasattr(trt, "NetworkDefinitionCreationFlag") and hasattr(
            trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH"
        ):
            network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

        network = builder.create_network(network_flags)
        config = builder.create_builder_config()
        parser = trt.OnnxParser(network, TRT_LOGGER)

        # Set workspace memory limit safely
        workspace_bytes = int(max_workspace_gb * (1024**3))
        if hasattr(config, "set_memory_pool_limit") and hasattr(trt, "MemoryPoolType"):
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
        elif hasattr(config, "max_workspace_size"):
            config.max_workspace_size = workspace_bytes

        # Parse ONNX computational graph
        with open(onnx_path, "rb") as model_file:
            if not parser.parse(model_file.read()):
                for error in range(parser.num_errors):
                    logger.error(f"ONNX Parsing Error: {parser.get_error(error)}")
                return False

        # Define Optimization Profile for ONNX Dynamic Dimensions
        profile = builder.create_optimization_profile()
        for i in range(network.num_inputs):
            input_tensor = network.get_input(i)
            shape = list(input_tensor.shape)

            # Map dynamic batch (-1) or dynamic dims to concrete ranges
            min_shape = tuple(1 if dim <= 0 else dim for dim in shape)
            opt_shape = tuple(
                32 if idx == 0 and dim <= 0 else (1 if dim <= 0 else dim)
                for idx, dim in enumerate(shape)
            )
            max_shape = tuple(
                128 if idx == 0 and dim <= 0 else (1 if dim <= 0 else dim)
                for idx, dim in enumerate(shape)
            )

            profile.set_shape(input_tensor.name, min_shape, opt_shape, max_shape)
            logger.info(
                f"Configured TRT Input Profile '{input_tensor.name}': min={min_shape}, opt={opt_shape}, max={max_shape}"
            )

        config.add_optimization_profile(profile)

        # Set FP16 mode flag
        if use_fp16:
            if hasattr(trt, "BuilderFlag") and hasattr(trt.BuilderFlag, "FP16"):
                config.set_flag(trt.BuilderFlag.FP16)
                logger.info("TensorRT FP16 mode enabled.")

        # Build serialized engine binary
        logger.info("Compiling CUDA kernels and optimizing execution graph...")
        start_time = time.time()
        serialized_engine = builder.build_serialized_network(network, config)

        if serialized_engine is None:
            logger.error("Failed to build TensorRT serialized network.")
            return False

        with open(engine_path, "wb") as f:
            f.write(serialized_engine)

        elapsed = time.time() - start_time
        logger.info(f"TensorRT Engine build successful in {elapsed:.2f}s!")
        return True

    def export_via_onnxruntime_ep(
        self, onnx_path: str, engine_path: str, use_fp16: bool = True
    ) -> bool:
        """Fallback engine compilation via ONNX Runtime TensorRT Execution Provider."""
        if not HAS_ORT:
            return False

        logger.info(
            "Attempting TensorRT engine compilation via ONNX Runtime TRT Provider..."
        )
        try:
            available_providers = ort.get_available_providers()
            trt_ep_name = None
            for p in available_providers:
                if p.lower() == "tensorrtexecutionprovider":
                    trt_ep_name = p
                    break

            if not trt_ep_name:
                logger.warning(
                    "TensorRTExecutionProvider not registered in available ORT providers."
                )
                return False

            trt_options = {
                "device_id": 0,
                "trt_fp16_enable": use_fp16,
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": str(self.output_dir),
            }
            providers = [
                (trt_ep_name, trt_options),
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ]
            session = ort.InferenceSession(onnx_path, providers=providers)

            # Trigger dummy forward pass to force engine compilation and disk caching
            inp_meta = session.get_inputs()[0]
            dummy_input = {inp_meta.name: np.zeros((1, 3, 32, 32), dtype=np.float32)}
            session.run(None, dummy_input)

            logger.info(
                f"ONNX Runtime TensorRT engine compiled and cached in {self.output_dir}"
            )
            return True
        except Exception as e:
            logger.warning(f"ONNX Runtime TensorRT compilation failed: {e}")
            return False

    def export_via_trtexec_cli(
        self, onnx_path: str, engine_path: str, use_fp16: bool = True
    ) -> bool:
        """Fallback build method using NVIDIA trtexec CLI binary."""
        cmd = [
            "trtexec",
            f"--onnx={onnx_path}",
            f"--saveEngine={engine_path}",
            "--workspace=2048",
        ]
        if use_fp16:
            cmd.append("--fp16")

        try:
            logger.info(f"Executing CLI fallback: {' '.join(cmd)}")
            subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
            )
            logger.info("trtexec engine compilation complete.")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.warning(f"trtexec execution failed: {e}")
            return False

    def convert_onnx_to_engine(
        self, onnx_path: str, use_fp16: bool = True, filename: Optional[str] = None
    ) -> Optional[str]:
        """Main conversion workflow trying Python API -> ORT TRT Provider -> trtexec CLI."""
        inp_path = Path(onnx_path)
        if not inp_path.exists():
            raise FileNotFoundError(f"Source ONNX file not found: {inp_path.resolve()}")

        out_name = filename or f"{inp_path.stem}.engine"
        out_path = self.output_dir / out_name

        logger.info(
            f"Starting TensorRT Compilation: {inp_path.name} -> {out_name} (FP16: {use_fp16})..."
        )

        success = False
        if HAS_TRT:
            try:
                success = self.export_via_python_api(
                    str(inp_path), str(out_path), use_fp16=use_fp16
                )
            except Exception as e:
                logger.warning(
                    f"Python TensorRT API build failed ({e}). Attempting fallbacks..."
                )

        if not success:
            success = self.export_via_onnxruntime_ep(
                str(inp_path), str(out_path), use_fp16=use_fp16
            )

        if not success:
            success = self.export_via_trtexec_cli(
                str(inp_path), str(out_path), use_fp16=use_fp16
            )

        if success and (out_path.exists() or list(self.output_dir.glob("*.engine"))):
            created_engine = (
                out_path
                if out_path.exists()
                else list(self.output_dir.glob("*.engine"))[0]
            )
            logger.info(
                f"TensorRT Engine Export Complete: {created_engine.name} ({get_file_size_mb(created_engine):.2f} MB)"
            )
            return str(created_engine)

        logger.error("TensorRT export failed across all conversion backends.")
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NeuroSwarm-AutoML TensorRT Engine Exporter"
    )
    parser.add_argument(
        "--onnx", type=str, required=True, help="Path to input .onnx model binary"
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=True,
        help="Enable FP16 Tensor Core acceleration",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs_trt",
        help="Directory to save serialized .engine file",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    exporter = TensorRTExporter(output_dir=args.output_dir)
    engine_file = exporter.convert_onnx_to_engine(args.onnx, use_fp16=args.fp16)

    if engine_file:
        print("=" * 65)
        print(">>> TensorRT Export Successful")
        print(f"Serialized Engine Saved: {engine_file}")
        print("=" * 65)
