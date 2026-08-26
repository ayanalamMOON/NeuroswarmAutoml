"""
Visualization and Plotting Module for NeuroSwarm-AutoML.

Provides convergence curve plotting, multi-objective Pareto front visualization,
and NetworkX DAG architecture rendering utilities for NAS search analysis.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import numpy as np
import networkx as nx

from neuroswarm.core.candidate import Candidate
from neuroswarm.utils.pareto import get_pareto_front

try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend for server/CI environments
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False

if HAS_PLOTTING:
    sns.set_theme(style="darkgrid")


def plot_convergence_curve(
    history: List[Dict[str, Any]],
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
) -> None:
    """
    Plots the best and mean fitness convergence over co-evolution generations.

    Args:
        history: List of dicts with keys 'generation', 'best_fitness', 'mean_fitness'.
        save_path: Optional file path to save the figure.
        show: Whether to display the plot interactively.
    """
    if not HAS_PLOTTING:
        print("[Visualization] matplotlib/seaborn not available. Skipping convergence plot.")
        return

    if not history:
        print("[Visualization] History log empty. Skipping convergence plot.")
        return

    generations = [h["generation"] for h in history]
    best_fitness = [h["best_fitness"] for h in history]
    mean_fitness = [h["mean_fitness"] for h in history]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(generations, best_fitness, "r-o", label="Best Fitness", linewidth=2.5, markersize=5)
    ax.plot(generations, mean_fitness, "b--s", label="Mean Population Fitness", linewidth=1.5, markersize=4)

    ax.set_xlabel("Generation", fontsize=12, fontweight="bold")
    ax.set_ylabel("Fitness (Validation Accuracy)", fontsize=12, fontweight="bold")
    ax.set_title("NeuroSwarm-AutoML Convergence Curve", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, loc="lower right")
    ax.grid(True, alpha=0.3)
    sns.despine()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_pareto_front(
    candidates_or_params: Union[List[Candidate], List[int]],
    accuracies: Optional[List[float]] = None,
    labels: Optional[List[str]] = None,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
) -> None:
    """
    Plots a Pareto front of model accuracy vs. parameter count. Accepts either a list
    of Candidate objects or separate parameter and accuracy arrays.

    Args:
        candidates_or_params: List of Candidate objects OR raw parameter count integers.
        accuracies: List of corresponding validation accuracies (required if passing raw lists).
        labels: Optional candidate labels for point annotation.
        save_path: Optional file path to save the figure.
        show: Whether to display the plot interactively.
    """
    if not HAS_PLOTTING:
        print("[Visualization] matplotlib/seaborn not available. Skipping Pareto plot.")
        return

    # Extract coordinates from Candidate list if supplied directly
    if candidates_or_params and isinstance(candidates_or_params[0], Candidate):
        candidates = [c for c in candidates_or_params if c.is_ground_truth]
        if not candidates:
            candidates = candidates_or_params

        param_counts = [c.param_count for c in candidates]
        accuracies = [c.fitness for c in candidates]
        labels = [c.candidate_id for c in candidates]

        # Calculate Pareto frontier for highlighted line plotting
        pareto_candidates = get_pareto_front(candidates)
        pareto_params = [c.param_count for c in pareto_candidates]
        pareto_accs = [c.fitness for c in pareto_candidates]
    else:
        param_counts = candidates_or_params
        if accuracies is None:
            raise ValueError("accuracies must be provided when passing parameter count lists.")
        pareto_params, pareto_accs = [], []

    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(
        param_counts, accuracies,
        c=accuracies, cmap="viridis", s=70, edgecolors="black", linewidths=0.5, alpha=0.85, label="Explored Candidates"
    )
    plt.colorbar(scatter, ax=ax, label="Validation Accuracy")

    # Connect Pareto frontier points with a red line if available
    if pareto_params:
        sorted_pareto = sorted(zip(pareto_params, pareto_accs), key=lambda x: x[0])
        p_x, p_y = zip(*sorted_pareto)
        ax.plot(p_x, p_y, color="crimson", linestyle="--", linewidth=2, label="Pareto Frontier")
        ax.scatter(pareto_params, pareto_accs, c="crimson", s=100, edgecolors="black", zorder=5, label="Pareto Optimal")

    # Annotate points
    if labels:
        for i, label in enumerate(labels):
            ax.annotate(label, (param_counts[i], accuracies[i]), fontsize=7, alpha=0.7, xytext=(4, 4), textcoords="offset points")

    ax.set_xlabel("Parameter Count", fontsize=12, fontweight="bold")
    ax.set_ylabel("Validation Accuracy", fontsize=12, fontweight="bold")
    ax.set_title("NeuroSwarm-AutoML Pareto Trade-Off", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.3)
    sns.despine()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_dag_architecture(
    dag: nx.DiGraph,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
) -> None:
    """
    Renders a NetworkX DiGraph neural architecture as a directed layer visualization.

    Args:
        dag: NetworkX DiGraph representing the neural architecture.
        save_path: Optional file path to save the figure.
        show: Whether to display the plot interactively.
    """
    if not HAS_PLOTTING:
        print("[Visualization] matplotlib/seaborn not available. Skipping DAG plot.")
        return

    fig, ax = plt.subplots(figsize=(12, 8))

    # Compute topological layer positions
    try:
        topo_order = list(nx.topological_sort(dag))
        pos = {}
        for layer_idx, node in enumerate(topo_order):
            pos[node] = (layer_idx, -layer_idx * 0.4)
    except nx.NetworkXUnfeasible:
        pos = nx.spring_layout(dag, seed=42)

    # Construct node labels with operation types
    node_labels = {}
    for node in dag.nodes():
        op = dag.nodes[node].get("op", "unknown")
        node_labels[node] = f"Node {node}\n({op})"

    # Color palette for operational layers
    op_colors = {
        "conv3x3": "#3498db",
        "conv5x5": "#2ecc71",
        "depthwise_conv": "#e74c3c",
        "resnet_block": "#f39c12",
        "identity": "#95a5a6",
    }
    colors = [op_colors.get(dag.nodes[n].get("op", ""), "#bdc3c7") for n in dag.nodes()]

    nx.draw_networkx(
        dag, pos, ax=ax,
        labels=node_labels,
        node_color=colors,
        node_size=1600,
        font_size=8,
        font_weight="bold",
        arrows=True,
        arrowsize=18,
        edge_color="#444444",
        width=1.8,
    )

    ax.set_title("Neural Architecture Dynamic DAG Structure", fontsize=14, fontweight="bold")
    ax.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
