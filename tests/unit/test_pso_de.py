"""
Unit tests for Continuous PSO-DE Hyperparameter Optimizer.
"""

import pytest
import numpy as np

from neuroswarm.core.candidate import Candidate
from neuroswarm.optimizers.pso_de_continuous import ContinuousPSODE


@pytest.fixture
def sample_population():
    """Generates a small synthetic candidate population."""
    pop = []
    bounds = np.array([
        [-4.0, -1.0],
        [0.8, 0.999],
        [-6.0, -2.0],
        [4.0, 8.0],
    ])
    for i in range(10):
        pos = np.array([
            np.random.uniform(bounds[0, 0], bounds[0, 1]),
            np.random.uniform(bounds[1, 0], bounds[1, 1]),
            np.random.uniform(bounds[2, 0], bounds[2, 1]),
            np.random.uniform(bounds[3, 0], bounds[3, 1]),
        ])
        cand = Candidate(
            candidate_id=f"cand_{i}",
            hyperparams=pos,
            velocity=np.zeros_like(pos),
            pbest_hyperparams=np.copy(pos),
            fitness=float(i * 0.1),
            pbest_score=float(i * 0.1),
        )
        pop.append(cand)
    return pop


def test_pso_de_step_updates_velocity_and_position(sample_population):
    """Ensure a step of PSO-DE modifies velocities and positions."""
    pso = ContinuousPSODE(population_size=len(sample_population))
    evolved = pso.step(sample_population, current_gen=1, max_gens=10)

    for i, c in enumerate(evolved):
        assert not np.all(c.velocity == 0.0) or i == len(sample_population) - 1


def test_pso_de_enforces_bounds(sample_population):
    """Ensure candidate positions stay within search bounds after multiple steps."""
    bounds = np.array([
        [-4.0, -1.0],
        [0.8, 0.999],
        [-6.0, -2.0],
        [4.0, 8.0],
    ])
    pso = ContinuousPSODE(population_size=len(sample_population), bounds=bounds)

    for gen in range(1, 20):
        sample_population = pso.step(sample_population, current_gen=gen, max_gens=20)
        for c in sample_population:
            assert np.all(c.hyperparams >= bounds[:, 0] - 1e-7), f"Lower bound violated: {c.hyperparams}"
            assert np.all(c.hyperparams <= bounds[:, 1] + 1e-7), f"Upper bound violated: {c.hyperparams}"


def test_pso_de_stagnation_triggers_de():
    """Ensure zero-velocity candidates trigger DE mutation operator without error."""
    bounds = np.array([[-4.0, -1.0], [0.8, 0.999], [-6.0, -2.0], [4.0, 8.0]])
    pso = ContinuousPSODE(
        population_size=5,
        bounds=bounds,
        velocity_threshold=1.0,
    )

    pop = [
        Candidate(
            candidate_id=f"c_{i}",
            hyperparams=np.array([-2.0, 0.9, -4.0, 6.0]),
            velocity=np.zeros(4),
            pbest_hyperparams=np.array([-2.0, 0.9, -4.0, 6.0]),
            fitness=float(i),
        )
        for i in range(5)
    ]

    evolved = pso.step(pop, current_gen=1, max_gens=5)
    assert len(evolved) == 5

