"""
Abstract Base Optimizer Module.

Defines the contract for discrete, continuous, and hybrid population-based optimizers.
"""

from abc import ABC, abstractmethod
from typing import List
from neuroswarm.core.candidate import Candidate


class BaseOptimizer(ABC):
    """Abstract Base Class governing population evolution."""

    def __init__(self, population_size: int):
        self.population_size = population_size

    @abstractmethod
    def step(self, population: List[Candidate], current_gen: int, max_gens: int) -> List[Candidate]:
        """
        Advances the population by one evolutionary generation / velocity iteration.

        Args:
            population: List of candidate agents.
            current_gen: Current generation index.
            max_gens: Total number of planned generations.

        Returns:
            Evolved list of candidate agents.
        """
