"""
Candidate Representation Module with Hardware-Aware Fitness Scoring.

Encapsulates the bi-level individual: discrete DAG topology combined with continuous
hyperparameter particle vectors, execution metrics, and latency/FLOP-penalized fitness.
"""

from dataclasses import dataclass, field
from typing import Dict, Any
import uuid
import numpy as np
import networkx as nx


@dataclass
class Candidate:
    """
    Composite agent representing a neural architecture and its hyperparameter state,
    with hardware-aware constraint evaluation.
    """

    # Unique identifier
    candidate_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    # Discrete Topology (GA Space)
    graph: nx.DiGraph = field(default_factory=nx.DiGraph)

    # Continuous Hyperparameters (PSO-DE Space)
    # Hyperparams vector convention: [log10(lr), momentum/beta1, weight_decay, batch_size_exp]
    hyperparams: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.float64))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.float64))

    # Personal Best Memory for PSO
    pbest_hyperparams: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.float64))
    pbest_score: float = float("-inf")

    # Fitness & Performance Metadata
    fitness: float = float("-inf")  # Raw validation accuracy or surrogate score
    constrained_fitness: float = float("-inf")  # Latency/FLOP-penalized score
    uncertainty: float = 1.0  # Surrogate estimation variance (sigma)
    evaluated_epochs: int = 0
    is_ground_truth: bool = False  # True if evaluated via actual PyTorch training

    # Hardware & Resource Metrics
    param_count: int = 0
    flops: int = 0
    latency_ms: float = 0.0  # Measured hardware inference latency in ms

    def __post_init__(self):
        """Ensure initial personal best matches starting position if uninitialized."""
        if np.all(self.pbest_hyperparams == 0) and not np.all(self.hyperparams == 0):
            self.pbest_hyperparams = np.copy(self.hyperparams)

    def effective_fitness(self, use_constrained: bool = True) -> float:
        """Returns either penalized constrained fitness or raw accuracy depending on search mode."""
        if use_constrained and self.constrained_fitness != float("-inf"):
            return self.constrained_fitness
        return self.fitness

    def compute_hardware_aware_fitness(
        self,
        target_latency_ms: float = 0.0,
        alpha: float = 0.05,
        target_flops: int = 0,
        beta: float = 0.0,
    ) -> float:
        """
        Calculates latency and FLOP penalized fitness score:
        F_constrained = Fitness - alpha * max(0, Latency - Target_Latency) - beta * max(0, FLOPs - Target_FLOPs)
        """
        if self.fitness == float("-inf"):
            self.constrained_fitness = float("-inf")
            return float("-inf")

        penalty = 0.0

        # Latency Penalty Threshold: alpha * max(0, Latency - tau)
        if target_latency_ms > 0.0 and self.latency_ms > 0.0:
            latency_excess = max(0.0, self.latency_ms - target_latency_ms)
            penalty += alpha * latency_excess

        # FLOPs Excess Penalty (measured in Millions of FLOPs)
        if target_flops > 0 and self.flops > 0:
            flops_excess = max(0.0, (self.flops - target_flops) / 1e6)
            penalty += beta * flops_excess

        self.constrained_fitness = max(0.0, self.fitness - penalty)
        return self.constrained_fitness

    def update_pbest(self, use_constrained: bool = False) -> bool:
        """Updates the particle's historical best position if current fitness improves."""
        current_score = self.effective_fitness(use_constrained=use_constrained)
        if current_score > self.pbest_score:
            self.pbest_score = current_score
            self.pbest_hyperparams = np.copy(self.hyperparams)
            return True
        return False

    def calculate_flops(self, in_channels: int = 3, base_channels: int = 32) -> int:
        """Estimates model FLOPs based on node count and convolution operations in DAG."""
        if self.flops > 0:
            return self.flops

        nodes = list(self.graph.nodes)
        node_count = len(nodes)

        base_ops_per_node = 2 * base_channels * base_channels * 32 * 32 * 9
        estimated_flops = node_count * base_ops_per_node
        self.flops = max(estimated_flops, 1000)
        return self.flops

    def get_decoded_hyperparams(self) -> Dict[str, Any]:
        """
        Decodes continuous optimization vector into usable PyTorch parameters.
        """
        return {
            "learning_rate": float(10 ** self.hyperparams[0]),
            "beta1": float(np.clip(self.hyperparams[1], 0.8, 0.999)),
            "weight_decay": float(10 ** self.hyperparams[2]),
            "batch_size": int(2 ** int(round(self.hyperparams[3]))),
        }

    def clone(self) -> "Candidate":
        """Deep copies the candidate structure including velocity and hardware state."""
        return Candidate(
            candidate_id=self.candidate_id,
            graph=self.graph.copy(),
            hyperparams=np.copy(self.hyperparams),
            velocity=np.copy(self.velocity),
            pbest_hyperparams=np.copy(self.pbest_hyperparams),
            pbest_score=self.pbest_score,
            fitness=self.fitness,
            constrained_fitness=self.constrained_fitness,
            uncertainty=self.uncertainty,
            evaluated_epochs=self.evaluated_epochs,
            is_ground_truth=self.is_ground_truth,
            param_count=self.param_count,
            flops=self.flops,
            latency_ms=self.latency_ms,
        )

    def __repr__(self) -> str:
        return (
            f"<Candidate ID={self.candidate_id} | Acc={self.fitness:.4f} | "
            f"Constrained={self.constrained_fitness:.4f} | Latency={self.latency_ms:.2f}ms | "
            f"Params={self.param_count:,}>"
        )
