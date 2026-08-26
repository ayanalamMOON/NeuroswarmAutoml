"""
Gaussian Process Surrogate Estimator Module.

Fits continuous Gaussian Process models on graph embeddings and hyperparameter vectors
to estimate performance mean and epistemic variance for fast UCB filtering.
"""

from typing import List, Tuple
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.preprocessing import StandardScaler

from neuroswarm.core.candidate import Candidate
from neuroswarm.surrogates.base_surrogate import BaseSurrogateModel
from neuroswarm.surrogates.graph_embedder import GraphEmbedder


class GaussianProcessSurrogate(BaseSurrogateModel):
    """
    Gaussian Process Regressor wrapped with standard scaling and variance estimation.
    """

    def __init__(self, alpha: float = 1e-6, n_restarts_optimizer: int = 5):
        # Radial Basis Function (RBF) kernel with noise term for observation variances
        kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(
            noise_level=1e-3
        )

        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=alpha,
            n_restarts_optimizer=n_restarts_optimizer,
            normalize_y=True,
            random_state=42,
        )
        self.scaler = StandardScaler()
        self.embedder = GraphEmbedder()
        self._is_fitted: bool = False

    @property
    def is_fitted(self) -> bool:
        """Returns True if the surrogate has been trained on ground-truth evaluation data."""
        return self._is_fitted

    def fit(self, candidates: List[Candidate]) -> None:
        """
        Trains the Gaussian Process on evaluated ground-truth candidates.
        """
        ground_truth_cands = [c for c in candidates if c.is_ground_truth and c.fitness != float("-inf")]

        if len(ground_truth_cands) < 3:
            # Not enough samples to fit a stable GP kernel
            return

        X_raw = np.array([self.embedder.extract_features(c) for c in ground_truth_cands])
        y = np.array([c.fitness for c in ground_truth_cands])

        # Standardize input vector dimensions
        X_scaled = self.scaler.fit_transform(X_raw)

        # Fit Gaussian Process
        self.gp.fit(X_scaled, y)
        self._is_fitted = True

    def predict(self, candidate: Candidate) -> Tuple[float, float]:
        """
        Predicts mean performance (mu) and standard deviation variance (sigma) for a candidate.

        Returns:
            Tuple[float, float]: (mean_prediction, uncertainty_sigma)
        """
        if not self._is_fitted:
            # Return prior baseline if model is uninitialized
            return 0.0, 1.0

        feat = self.embedder.extract_features(candidate).reshape(1, -1)
        feat_scaled = self.scaler.transform(feat)

        mean, std = self.gp.predict(feat_scaled, return_std=True)
        return float(mean[0]), float(std[0])
