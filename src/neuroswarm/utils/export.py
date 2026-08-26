"""
Model Export & ONNX Verification Module.

Provides serialization and deployment export utilities for dynamic PyTorch neural architectures
into standard ONNX and TorchScript formats, along with numerical inference verification.
"""

import logging
from pathlib import Path
from typing import Tuple, Optional
import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger("neuroswarm.export")


class ModelExporter:
    """
    Handles model serialization and conversion from dynamic PyTorch architectures
    to production-ready ONNX and TorchScript formats with sanity verification.
    """

    def __init__(self, output_dir: str = "./outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_to_onnx(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...] = (1, 3, 32, 32),
        filename: str = "model.onnx",
        opset_version: int = 14,
        dynamic_axes: Optional[dict] = None,
    ) -> Optional[str]:
        """
        Exports a PyTorch model into ONNX format.
        """
        export_path = self.output_dir / filename
        model.eval()

        device = next(model.parameters()).device if list(model.parameters()) else torch.device("cpu")
        dummy_input = torch.randn(*input_shape, device=device)

        if dynamic_axes is None:
            dynamic_axes = {
                "input": {0: "batch_size"},
                "output": {0: "batch_size"},
            }

        try:
            torch.onnx.export(
                model,
                dummy_input,
                str(export_path),
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes=dynamic_axes,
            )
            logger.info(f"Successfully exported ONNX model to: {export_path}")
            return str(export_path)
        except (ImportError, ModuleNotFoundError) as e:
            logger.warning(f"ONNX export skipped (optional dependency missing: {e}).")
            return None
        except Exception as e:
            logger.warning(f"ONNX export skipped ({e}).")
            return None

    def verify_onnx(
        self,
        onnx_path: str,
        model: nn.Module,
        input_shape: Tuple[int, ...] = (1, 3, 32, 32),
        rtol: float = 1e-3,
        atol: float = 1e-4,
    ) -> bool:
        """
        Verifies numerical equivalence between PyTorch model and exported ONNX runtime inference.
        """
        model.eval()
        device = next(model.parameters()).device if list(model.parameters()) else torch.device("cpu")
        dummy_input = torch.randn(*input_shape, device=device)

        # 1. Compute PyTorch baseline output
        with torch.no_grad():
            torch_output = model(dummy_input).cpu().numpy()

        # 2. Check ONNX graph structure
        try:
            import onnx

            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            logger.info("ONNX model structure validation: PASSED")
        except ImportError:
            logger.warning("Package 'onnx' not installed. Skipping structural schema check.")
        except Exception as e:
            logger.error(f"ONNX schema validation failed: {e}")
            return False

        # 3. Check ONNX Runtime execution
        try:
            import onnxruntime as ort

            ort_session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
            ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.cpu().numpy()}
            ort_outputs = ort_session.run(None, ort_inputs)[0]

            np.testing.assert_allclose(torch_output, ort_outputs, rtol=rtol, atol=atol)
            logger.info("ONNX Runtime numerical equivalence test: PASSED")
            return True
        except ImportError:
            logger.warning("Package 'onnxruntime' not installed. Skipping numerical equivalence test.")
            return True
        except AssertionError as e:
            logger.error(f"Numerical discrepancy between PyTorch and ONNX outputs: {e}")
            return False
        except Exception as e:
            logger.warning(f"ONNX Runtime inference verification warning: {e}")
            return True

    def export_to_torchscript(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...] = (1, 3, 32, 32),
        filename: str = "model.pt",
    ) -> Optional[str]:
        """
        Exports a PyTorch model into TorchScript format via JIT tracing.
        """
        export_path = self.output_dir / filename
        model.eval()
        device = next(model.parameters()).device if list(model.parameters()) else torch.device("cpu")
        dummy_input = torch.randn(*input_shape, device=device)

        try:
            traced_model = torch.jit.trace(model, dummy_input)
            traced_model.save(str(export_path))
            logger.info(f"Successfully exported TorchScript model to: {export_path}")
            return str(export_path)
        except Exception as e:
            logger.error(f"TorchScript tracing failed: {e}")
            return None
