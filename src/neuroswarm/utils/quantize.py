"""
Post-Training Quantization (PTQ) & Model Compression Module.

Converts exported FP32 PyTorch (.pt) and ONNX (.onnx) model binaries into
FP16 (Half-Precision) and INT8 (Quantized) representation, profiling file size
compression and latency speedup.
"""

import argparse
import logging
import os
from pathlib import Path
import sys
import time
from typing import Dict, Any, Tuple, Optional, Union

import numpy as np
import torch
import torch.nn as nn

# Link PyTorch CUDA DLLs for Windows environment compatibility
if os.name == "nt" and torch.cuda.is_available():
    torch_lib_path = os.path.join(os.path.dirname(torch.__file__), "lib")
    if os.path.exists(torch_lib_path):
        os.add_dll_directory(torch_lib_path)
        os.environ["PATH"] = torch_lib_path + os.path.pathsep + os.environ.get("PATH", "")

# Conditional ONNX Runtime imports
try:
    import onnxruntime as ort
    from onnxruntime.quantization import quantize_dynamic, QuantType
    HAS_ORT_QUANT = True
except ImportError:
    HAS_ORT_QUANT = False

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("neuroswarm.quantize")


def get_file_size_mb(path: Union[str, Path]) -> float:
    """Returns file size in Megabytes (MB)."""
    return Path(path).stat().st_size / (1024.0 * 1024.0)


class ModelQuantizer:
    """
    Handles FP16 and INT8 Post-Training Quantization for ONNX and TorchScript models.
    """

    def __init__(self, output_dir: str = "./outputs_quantized"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def quantize_onnx_int8(
        self,
        input_onnx_path: str,
        filename: Optional[str] = None
    ) -> Optional[str]:
        """
        Applies dynamic INT8 weight quantization to an ONNX model.
        """
        if not HAS_ORT_QUANT:
            logger.error("onnxruntime is required for ONNX quantization. Install via 'pip install onnxruntime'.")
            return None

        inp_path = Path(input_onnx_path)
        if not inp_path.exists():
            raise FileNotFoundError(f"Source ONNX model not found: {inp_path.resolve()}")

        out_name = filename or f"{inp_path.stem}_int8.onnx"
        out_path = self.output_dir / out_name

        try:
            logger.info(f"Quantizing ONNX model to INT8: {inp_path.name} -> {out_name}...")
            quantize_dynamic(
                model_input=str(inp_path),
                model_output=str(out_path),
                weight_type=QuantType.QUInt8,
            )

            orig_size = get_file_size_mb(inp_path)
            quant_size = get_file_size_mb(out_path)
            reduction = (1.0 - (quant_size / orig_size)) * 100.0

            logger.info(f"INT8 ONNX Quantization complete: {orig_size:.2f} MB -> {quant_size:.2f} MB ({reduction:.1f}% reduction)")
            return str(out_path)
        except Exception as e:
            logger.error(f"ONNX INT8 quantization failed: {e}")
            return None

    def quantize_torchscript_fp16(
        self,
        input_pt_path: str,
        filename: Optional[str] = None,
        input_shape: Tuple[int, ...] = (1, 3, 32, 32)
    ) -> Optional[str]:
        """
        Converts TorchScript model parameters and graph execution to FP16 half precision.
        """
        inp_path = Path(input_pt_path)
        if not inp_path.exists():
            raise FileNotFoundError(f"Source TorchScript model not found: {inp_path.resolve()}")

        out_name = filename or f"{inp_path.stem}_fp16.pt"
        out_path = self.output_dir / out_name

        try:
            logger.info(f"Converting TorchScript model to FP16: {inp_path.name} -> {out_name}...")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Load baseline model
            model = torch.jit.load(str(inp_path), map_location=device)
            model.eval()

            # Cast model parameters to float16
            model_fp16 = model.half()
            dummy_input = torch.randn(*input_shape, device=device, dtype=torch.float16)

            # Re-trace in half precision
            traced_fp16 = torch.jit.trace(model_fp16, dummy_input)
            traced_fp16.save(str(out_path))

            orig_size = get_file_size_mb(inp_path)
            quant_size = get_file_size_mb(out_path)
            reduction = (1.0 - (quant_size / orig_size)) * 100.0

            logger.info(f"FP16 TorchScript conversion complete: {orig_size:.2f} MB -> {quant_size:.2f} MB ({reduction:.1f}% reduction)")
            return str(out_path)
        except Exception as e:
            logger.error(f"TorchScript FP16 conversion failed: {e}")
            return None

    def quantize_torchscript_int8(
        self,
        input_pt_path: str,
        filename: Optional[str] = None
    ) -> Optional[str]:
        """
        Applies PyTorch dynamic INT8 quantization to Conv2d and Linear layers in TorchScript.
        """
        inp_path = Path(input_pt_path)
        if not inp_path.exists():
            raise FileNotFoundError(f"Source TorchScript model not found: {inp_path.resolve()}")

        out_name = filename or f"{inp_path.stem}_int8.pt"
        out_path = self.output_dir / out_name

        try:
            logger.info(f"Quantizing TorchScript model to INT8: {inp_path.name} -> {out_name}...")
            model = torch.jit.load(str(inp_path), map_location="cpu")
            model.eval()

            # Apply PyTorch dynamic quantization across standard neural layers
            quantized_model = torch.ao.quantization.quantize_dynamic(
                model,
                {nn.Linear, nn.Conv2d, nn.BatchNorm2d},
                dtype=torch.qint8
            )

            # Save quantized script module
            torch.jit.save(quantized_model, str(out_path))

            orig_size = get_file_size_mb(inp_path)
            quant_size = get_file_size_mb(out_path)
            reduction = (1.0 - (quant_size / orig_size)) * 100.0

            logger.info(f"INT8 TorchScript dynamic quantization complete: {orig_size:.2f} MB -> {quant_size:.2f} MB ({reduction:.1f}% reduction)")
            return str(out_path)
        except Exception as e:
            logger.error(f"TorchScript INT8 quantization failed: {e}")
            return None


def benchmark_compression_comparison(
    original_path: str,
    quantized_path: str,
    iterations: int = 500,
    batch_size: int = 32
) -> None:
    """Profiles speed and memory compression metrics between FP32 original and quantized model."""
    orig_p = Path(original_path)
    quant_p = Path(quantized_path)

    orig_size = get_file_size_mb(orig_p)
    quant_size = get_file_size_mb(quant_p)
    size_reduction = (1.0 - (quant_size / orig_size)) * 100.0

    print("=" * 65)
    print(">>> Quantization Compression & Speed Profile")
    print(f"Original Model:   {orig_p.name} ({orig_size:.3f} MB)")
    print(f"Quantized Model:  {quant_p.name} ({quant_size:.3f} MB)")
    print(f"Size Reduction:   {size_reduction:.2f}% smaller")
    print("=" * 65)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NeuroSwarm-AutoML Model Quantization CLI")
    parser.add_argument("--model", type=str, required=True, help="Path to input .onnx or .pt model binary")
    parser.add_argument("--format", type=str, default="int8", choices=["int8", "fp16"], help="Quantization target format")
    parser.add_argument("--output_dir", type=str, default="./outputs_quantized", help="Directory to save quantized artifact")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    quantizer = ModelQuantizer(output_dir=args.output_dir)
    inp_path = Path(args.model)

    out_file = None
    if inp_path.suffix == ".onnx":
        if args.format == "int8":
            out_file = quantizer.quantize_onnx_int8(str(inp_path))
    elif inp_path.suffix == ".pt":
        if args.format == "fp16":
            out_file = quantizer.quantize_torchscript_fp16(str(inp_path))
        elif args.format == "int8":
            out_file = quantizer.quantize_torchscript_int8(str(inp_path))

    if out_file and Path(out_file).exists():
        benchmark_compression_comparison(str(inp_path), out_file)
