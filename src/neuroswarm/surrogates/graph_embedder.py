"""
Graph & Hyperparameter Feature Embedding Module.

Transforms DAG neural architectures and continuous hyperparameter vectors into fixed-dimensional
numerical feature embeddings suitable for tabular machine learning surrogate models.
"""

from typing import List, Optional, Dict
import numpy as np
import networkx as nx
from scipy.linalg import svdvals

from neuroswarm.core.candidate import Candidate


class GraphEmbedder:
    """
    Computes comprehensive vector embeddings combining topological graph invariants,
    spectral properties, operation distributions, and continuous hyperparameters.
    """

    DEFAULT_OPS = [
        "conv3x3",
        "conv5x5",
        "depthwise_conv",
        "resnet_block",
        "identity",
    ]

    def __init__(
        self,
        available_ops: Optional[List[str]] = None,
        max_nodes: int = 10,
        spectral_k: int = 4,
    ):
        self.available_ops = available_ops or self.DEFAULT_OPS
        self.max_nodes = max_nodes
        self.spectral_k = spectral_k
        self.op_to_idx: Dict[str, int] = {op: i for i, op in enumerate(self.available_ops)}

    @property
    def embedding_dim(self) -> int:
        """Calculates the total dimension of the extracted feature vector."""
        topological_dim = 8
        op_hist_dim = len(self.available_ops)
        spectral_dim = self.spectral_k
        hyperparam_dim = 4  # Standard continuous hyperparameter vector length
        return topological_dim + op_hist_dim + spectral_dim + hyperparam_dim

    def extract_features(self, candidate: Candidate) -> np.ndarray:
        """
        Extracts a unified 1D numerical feature vector for a candidate.

        Feature Vector Structure:
        [
            - Structural Topological Invariants (8 dimensions)
            - Operation Normalized Histogram (len(available_ops) dimensions)
            - Singular Value Spectral Properties (spectral_k dimensions)
            - Continuous Hyperparameter Vector (4 dimensions)
        ]
        """
        g = candidate.graph

        # 1. Structural Graph Invariants
        topological_feats = self._extract_topological_features(g)

        # 2. Node Operation Frequency Histogram
        op_histogram = self._extract_operation_histogram(g)

        # 3. Spectral Properties (Adjacency Matrix SVD Singular Values)
        spectral_feats = self._extract_spectral_features(g)

        # 4. Continuous Hyperparameter Vector
        hyperparam_feats = np.copy(candidate.hyperparams)

        # Concatenate all feature sub-vectors into a single 1D vector
        return np.concatenate(
            [
                topological_feats,
                op_histogram,
                spectral_feats,
                hyperparam_feats,
            ],
            dtype=np.float64,
        )

    def batch_extract_features(self, candidates: List[Candidate]) -> np.ndarray:
        """
        Extracts feature vectors for a batch of candidates into a 2D matrix of shape (N, D).
        """
        if not candidates:
            return np.empty((0, self.embedding_dim), dtype=np.float64)

        return np.array([self.extract_features(c) for c in candidates], dtype=np.float64)

    def _extract_topological_features(self, g: nx.DiGraph) -> np.ndarray:
        """Extracts structural invariants from the graph."""
        num_nodes = g.number_of_nodes()
        num_edges = g.number_of_edges()

        if num_nodes == 0:
            return np.zeros(8, dtype=np.float64)

        density = nx.density(g)
        in_degrees = [d for _, d in g.in_degree()]
        out_degrees = [d for _, d in g.out_degree()]

        mean_in = float(np.mean(in_degrees)) if in_degrees else 0.0
        max_in = float(np.max(in_degrees)) if in_degrees else 0.0
        mean_out = float(np.mean(out_degrees)) if out_degrees else 0.0
        max_out = float(np.max(out_degrees)) if out_degrees else 0.0

        try:
            longest_path_len = float(len(nx.dag_longest_path(g)))
        except Exception:
            longest_path_len = 0.0

        return np.array(
            [
                float(num_nodes),
                float(num_edges),
                float(density),
                mean_in,
                max_in,
                mean_out,
                max_out,
                longest_path_len,
            ],
            dtype=np.float64,
        )

    def _extract_operation_histogram(self, g: nx.DiGraph) -> np.ndarray:
        """Computes normalized operation frequency distribution across DAG nodes."""
        counts = np.zeros(len(self.available_ops), dtype=np.float64)
        for _, data in g.nodes(data=True):
            op = data.get("op", None)
            if op in self.op_to_idx:
                counts[self.op_to_idx[op]] += 1.0

        # Normalize by total node count
        total_nodes = g.number_of_nodes()
        if total_nodes > 0:
            counts /= float(total_nodes)

        return counts

    def _extract_spectral_features(self, g: nx.DiGraph) -> np.ndarray:
        """Computes top singular values of the graph adjacency matrix."""
        if g.number_of_nodes() == 0:
            return np.zeros(self.spectral_k, dtype=np.float64)

        adj_matrix = nx.to_numpy_array(g, dtype=np.float64)

        try:
            s = svdvals(adj_matrix)
        except Exception:
            s = np.zeros(g.number_of_nodes(), dtype=np.float64)

        # Pad or truncate to fixed spectral_k dimension
        padded_s = np.zeros(self.spectral_k, dtype=np.float64)
        k = min(len(s), self.spectral_k)
        padded_s[:k] = s[:k]

        return padded_s
