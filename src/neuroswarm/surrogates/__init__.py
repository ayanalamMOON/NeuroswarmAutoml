"""
Surrogate Performance Estimation Package for NeuroSwarm-AutoML.

Provides abstract surrogate interfaces, graph feature embedders, and
Gaussian Process regression models for uncertainty-aware fitness approximation.
"""

from neuroswarm.surrogates.base_surrogate import BaseSurrogateModel
from neuroswarm.surrogates.gp_estimator import GaussianProcessSurrogate
from neuroswarm.surrogates.graph_embedder import GraphEmbedder

__all__ = [
    "BaseSurrogateModel",
    "GaussianProcessSurrogate",
    "GraphEmbedder",
]
