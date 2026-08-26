"""
NeuroSwarm-AutoML: Automated Neural Architecture Search & Hyperparameter Optimization.

Combining Graph Evolutionary Algorithms, Hybrid PSO-DE, and Gaussian Process Surrogates.
Automatically configures global PyTorch CUDA runtime settings for NVIDIA RTX GPUs
(Tensor Float 32, cuDNN benchmarking, and Windows DLL linking) upon package import.
"""

__version__ = "0.1.0"

import os
import sys
import logging
import torch

logger = logging.getLogger("neuroswarm")


def _initialize_rtx_hardware_acceleration() -> None:
    """Configures global RTX GPU Tensor Core and CUDA performance flags."""
    if not torch.cuda.is_available():
        return

    # 1. Dynamically link cuDNN 9 & cuBLAS 12 DLLs for Windows virtual environments
    if os.name == "nt":
        venv_base = sys.prefix
        dll_paths = [
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
        # 2. Enable TF32 (TensorFloat-32) on RTX Tensor Cores for FP32 matrix GEMM operations
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # 3. Enable cuDNN autotuning kernel selection for fixed input dimensions
        torch.backends.cudnn.benchmark = True
    except Exception as e:
        logger.warning(f"Failed to apply hardware acceleration flags: {e}")


# Run hardware acceleration initialization prior to package symbol imports
_initialize_rtx_hardware_acceleration()

# Package API Exports
from neuroswarm.core.candidate import Candidate
from neuroswarm.core.runner import ParallelRunner, train_and_evaluate_candidate
from neuroswarm.search_space.dag_space import DAGSearchSpace
from neuroswarm.search_space.dynamic_builder import DynamicNeuralNetwork
from neuroswarm.optimizers.base_optimizer import BaseOptimizer
from neuroswarm.optimizers.ga_topology import TopologyGAOptimizer
from neuroswarm.optimizers.pso_de_continuous import ContinuousPSODE
from neuroswarm.optimizers.bilevel_engine import BiLevelCoEvolutionEngine
from neuroswarm.surrogates.base_surrogate import BaseSurrogateModel
from neuroswarm.surrogates.gp_estimator import GaussianProcessSurrogate
from neuroswarm.surrogates.graph_embedder import GraphEmbedder
from neuroswarm.utils.metrics import count_parameters, estimate_flops, compute_accuracy
from neuroswarm.utils.visualization import (
    plot_convergence_curve,
    plot_pareto_front,
    plot_dag_architecture,
)
from neuroswarm.utils.pareto import (
    dominates,
    fast_non_dominated_sort,
    calculate_crowding_distance,
    get_pareto_front,
)
from neuroswarm.utils.export import ModelExporter

__all__ = [
    "__version__",
    "Candidate",
    "ParallelRunner",
    "train_and_evaluate_candidate",
    "DAGSearchSpace",
    "DynamicNeuralNetwork",
    "BaseOptimizer",
    "TopologyGAOptimizer",
    "ContinuousPSODE",
    "BiLevelCoEvolutionEngine",
    "BaseSurrogateModel",
    "GaussianProcessSurrogate",
    "GraphEmbedder",
    "count_parameters",
    "estimate_flops",
    "compute_accuracy",
    "plot_convergence_curve",
    "plot_pareto_front",
    "plot_dag_architecture",
    "dominates",
    "fast_non_dominated_sort",
    "calculate_crowding_distance",
    "get_pareto_front",
    "ModelExporter",
]
