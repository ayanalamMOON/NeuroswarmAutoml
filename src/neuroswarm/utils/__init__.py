"""Utility functions for model evaluation, metrics, Pareto optimization, export, and visualization."""

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
