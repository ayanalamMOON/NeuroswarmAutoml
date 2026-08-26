"""
Hybrid Particle Swarm Optimization with Differential Evolution (PSO-DE).

Optimizes continuous hyperparameter spaces using standard velocity vectors with dynamic
inertia decay, coupled with DE mutation vectors to escape velocity stagnation.
"""

from typing import List
import numpy as np

from neuroswarm.optimizers.base_optimizer import BaseOptimizer
from neuroswarm.core.candidate import Candidate


class ContinuousPSODE(BaseOptimizer):
    """
    Continuous hyperparameter solver combining Swarm acceleration with DE/rand/1 mutation operators.
    """

    def __init__(
        self,
        population_size: int,
        bounds: np.ndarray = None,
        w_max: float = 0.9,
        w_min: float = 0.4,
        c1: float = 1.496,
        c2: float = 1.496,
        de_scaling_factor: float = 0.6,
        velocity_threshold: float = 1e-4,
    ):
        super().__init__(population_size)
        # Bounds: [ [log10_lr_min, log10_lr_max], [beta1_min, beta1_max], [log10_wd_min, log10_wd_max], [batch_exp_min, batch_exp_max] ]
        self.bounds = bounds if bounds is not None else np.array([
            [-4.0, -1.0],   # LR: 1e-4 to 1e-1
            [0.8, 0.999],   # Beta1 / Momentum
            [-6.0, -2.0],   # Weight Decay: 1e-6 to 1e-2
            [4.0, 8.0],     # Batch Size exponent: 2^4 (16) to 2^8 (256)
        ])
        self.w_max = w_max
        self.w_min = w_min
        self.c1 = c1
        self.c2 = c2
        self.F = de_scaling_factor
        self.velocity_threshold = velocity_threshold

    def step(self, population: List[Candidate], current_gen: int, max_gens: int) -> List[Candidate]:
        """Executes one iteration of PSO-DE continuous parameter velocity updating."""
        # Identify global best agent
        gbest_cand = max(population, key=lambda c: c.pbest_score if c.pbest_score != float("-inf") else c.fitness)
        gbest_pos = np.copy(gbest_cand.pbest_hyperparams)

        # Compute dynamic inertia weight
        w = self.w_max - (current_gen / max(1, max_gens)) * (self.w_max - self.w_min)
        num_candidates = len(population)

        for i, cand in enumerate(population):
            r1, r2 = np.random.rand(self.bounds.shape[0]), np.random.rand(self.bounds.shape[0])

            # Standard Swarm Acceleration
            cognitive = self.c1 * r1 * (cand.pbest_hyperparams - cand.hyperparams)
            social = self.c2 * r2 * (gbest_pos - cand.hyperparams)
            new_velocity = w * cand.velocity + cognitive + social

            # Apply DE Mutation if velocity magnitude drops below threshold (Stagnation Breakout)
            if np.linalg.norm(new_velocity) < self.velocity_threshold and num_candidates >= 4:
                idx_pool = [j for j in range(num_candidates) if j != i]
                r_a, r_b = np.random.choice(idx_pool, size=2, replace=False)
                de_mutation_vec = self.F * (population[r_a].hyperparams - population[r_b].hyperparams)
                new_velocity = new_velocity + de_mutation_vec

            # Clamp Velocity
            v_max = 0.2 * (self.bounds[:, 1] - self.bounds[:, 0])
            new_velocity = np.clip(new_velocity, -v_max, v_max)

            # Update position and clamp to search domain bounds
            new_position = cand.hyperparams + new_velocity
            new_position = np.clip(new_position, self.bounds[:, 0], self.bounds[:, 1])

            # Assign updated vectors back to candidate
            cand.velocity = new_velocity
            cand.hyperparams = new_position

        return population
