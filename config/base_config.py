"""
Global Configuration Module with CUDA Acceleration Settings.

Dataclass-based configuration objects for all tunable search, optimizer,
surrogate, and GPU/CPU training parameters in the NeuroSwarm-AutoML pipeline.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import yaml
from pathlib import Path
import torch


@dataclass
class GAConfig:
    """Genetic Algorithm topology optimizer configuration."""
    population_size: int = 20
    crossover_prob: float = 0.7
    mutation_prob: float = 0.3
    max_nodes: int = 10
    available_ops: List[str] = field(default_factory=lambda: [
        "conv3x3", "conv5x5", "depthwise_conv", "resnet_block", "identity"
    ])


@dataclass
class PSOConfig:
    """PSO-DE continuous hyperparameter optimizer configuration."""
    w_max: float = 0.9
    w_min: float = 0.4
    c1: float = 1.496
    c2: float = 1.496
    de_scaling_factor: float = 0.6
    velocity_threshold: float = 1e-4
    bounds: Optional[List[List[float]]] = None


@dataclass
class SurrogateConfig:
    """Gaussian Process surrogate model configuration."""
    alpha: float = 1e-6
    n_restarts_optimizer: int = 5
    ucb_kappa: float = 1.96
    min_samples_to_fit: int = 5
    uncertainty_threshold: float = 0.5


@dataclass
class TrainingConfig:
    """PyTorch CUDA training and evaluation configuration."""
    short_epochs: int = 5
    full_epochs: int = 100
    in_channels: int = 3
    base_channels: int = 32
    num_classes: int = 10
    num_workers: int = 2
    use_ray: bool = False
    gpus_per_worker: float = 1.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp: bool = True
    pin_memory: bool = True
    cudnn_benchmark: bool = True
    dataset: str = "cifar10"
    data_root: str = "./data"


@dataclass
class SearchConfig:
    """Top-level search configuration aggregating all sub-configs."""
    max_generations: int = 50
    population_size: int = 20
    min_nodes: int = 4
    max_nodes: int = 10
    edge_prob: float = 0.4
    warm_start_count: int = 5
    eval_top_fraction: float = 0.20
    seed: int = 42

    ga: GAConfig = field(default_factory=GAConfig)
    pso: PSOConfig = field(default_factory=PSOConfig)
    surrogate: SurrogateConfig = field(default_factory=SurrogateConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "SearchConfig":
        """Load configuration from a YAML file, merging with defaults."""
        path = Path(yaml_path)
        if not path.exists():
            return cls()

        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}

        ga_data = data.pop("ga", {})
        pso_data = data.pop("pso", {})
        surrogate_data = data.pop("surrogate", {})
        training_data = data.pop("training", {})

        return cls(
            **{k: v for k, v in data.items() if k in cls.__dataclass_fields__},
            ga=GAConfig(**ga_data),
            pso=PSOConfig(**pso_data),
            surrogate=SurrogateConfig(**surrogate_data),
            training=TrainingConfig(**training_data),
        )
