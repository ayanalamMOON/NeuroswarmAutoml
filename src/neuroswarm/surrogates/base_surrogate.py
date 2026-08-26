"""
Abstract Base Surrogate Model Module.

Defines the unified interface for surrogate performance estimators (e.g., Gaussian Processes,
LightGBM, Random Forests) used to accelerate Neural Architecture Search (NAS).
"""

from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np

from neuroswarm.core.candidate import Candidate


class BaseSurrogateModel(ABC):
    """Abstract Base Class for surrogate performance estimation models."""

    @property
    @abstractmethod
    def is_fitted(self) -> bool:
        """Returns True if the surrogate has been trained on ground-truth evaluation data."""

    @abstractmethod
    def fit(self, candidates: List[Candidate]) -> None:
        """
        Trains the surrogate model on evaluated ground-truth candidates.

        Args:
            candidates: List of Candidate objects containing evaluated fitness scores.
        """

    @abstractmethod
    def predict(self, candidate: Candidate) -> Tuple[float, float]:
        """
        Predicts performance mean and epistemic variance for a single candidate.

        Args:
            candidate: Target Candidate object.

        Returns:
            Tuple[float, float]: (predicted_mean, uncertainty_std)
        """

    def predict_batch(self, candidates: List[Candidate]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predicts performance mean and variance across a batch of candidates.

        Args:
            candidates: List of target Candidate objects.

        Returns:
            Tuple[np.ndarray, np.ndarray]: (array_of_means, array_of_stds)
        """
        means, stds = [], []
        for cand in candidates:
            mu, sigma = self.predict(cand)
            means.append(mu)
            stds.append(sigma)
        return np.array(means, dtype=np.float64), np.array(stds, dtype=np.float64)

    def acquisition_ucb(self, candidate: Candidate, kappa: float = 1.96) -> float:
        """
        Computes Upper Confidence Bound (UCB) acquisition score.

        Score = mu + kappa * sigma

        Args:
            candidate: Candidate instance to evaluate.
            kappa: Exploration trade-off coefficient.

        Returns:
            float: Calculated UCB acquisition score.
        """
        mu, sigma = self.predict(candidate)
        return float(mu + kappa * sigma)
