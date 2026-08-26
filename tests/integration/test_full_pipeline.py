"""
Integration tests for the complete NeuroSwarm-AutoML surrogate-assisted search loop.
"""

import os
import tempfile
import numpy as np

from neuroswarm.core.candidate import Candidate
from neuroswarm.search_space.dag_space import DAGSearchSpace
from neuroswarm.optimizers.ga_topology import TopologyGAOptimizer
from neuroswarm.optimizers.pso_de_continuous import ContinuousPSODE
from neuroswarm.optimizers.bilevel_engine import BiLevelCoEvolutionEngine
from neuroswarm.surrogates.gp_estimator import GaussianProcessSurrogate
from neuroswarm.surrogates.graph_embedder import GraphEmbedder
from neuroswarm.utils.export import ModelExporter


def test_graph_embedder_dimension():
    """Ensure GraphEmbedder produces the correct feature dimensionality."""
    embedder = GraphEmbedder()
    dag_space = DAGSearchSpace(min_nodes=4, max_nodes=6)
    dag = dag_space.sample_random_dag()

    cand = Candidate(
        graph=dag,
        hyperparams=np.array([-2.0, 0.9, -4.0, 6.0]),
    )

    feat = embedder.extract_features(cand)
    assert feat.shape == (embedder.embedding_dim,), f"Expected dim {embedder.embedding_dim}, got {feat.shape}"
    assert not np.any(np.isnan(feat)), "Feature vector contains NaNs"


def test_surrogate_fit_and_predict():
    """Ensure GaussianProcessSurrogate fits on candidates and outputs valid mean and variance."""
    surrogate = GaussianProcessSurrogate()
    dag_space = DAGSearchSpace(min_nodes=4, max_nodes=6)

    candidates = []
    for i in range(6):
        c = Candidate(
            graph=dag_space.sample_random_dag(),
            hyperparams=np.array([-2.0, 0.9, -4.0, 6.0]),
            fitness=0.5 + i * 0.05,
            is_ground_truth=True,
        )
        candidates.append(c)

    surrogate.fit(candidates)
    assert surrogate.is_fitted

    test_cand = Candidate(
        graph=dag_space.sample_random_dag(),
        hyperparams=np.array([-3.0, 0.95, -5.0, 5.0]),
    )
    mu, sigma = surrogate.predict(test_cand)
    assert isinstance(mu, float)
    assert isinstance(sigma, float)
    assert sigma >= 0.0

    ucb = surrogate.acquisition_ucb(test_cand, kappa=1.96)
    assert ucb == mu + 1.96 * sigma


def test_bilevel_coevolution_single_generation():
    """Ensure BiLevelCoEvolutionEngine runs a generation without errors."""
    pop_size = 6
    dag_space = DAGSearchSpace(min_nodes=4, max_nodes=6)
    pop = [
        Candidate(
            candidate_id=f"c_{i}",
            graph=dag_space.sample_random_dag(),
            hyperparams=np.array([-2.0, 0.9, -4.0, 6.0]),
        )
        for i in range(pop_size)
    ]

    engine = BiLevelCoEvolutionEngine(
        population_size=pop_size,
        ga_optimizer=TopologyGAOptimizer(population_size=pop_size),
        pso_optimizer=ContinuousPSODE(population_size=pop_size),
        surrogate=GaussianProcessSurrogate(),
    )

    def mock_eval_fn(cand, epochs, config):
        return 0.75, 50000, 100000

    next_pop = engine.run_generation(
        population=pop,
        current_gen=1,
        max_gens=5,
        eval_fn=mock_eval_fn,
        dataset_config={},
        short_epochs=1,
    )

    assert len(next_pop) == pop_size
    assert engine.global_best_candidate is not None
    assert engine.global_best_candidate.fitness == 0.75
    assert len(engine.history) == 1


def test_model_exporter():
    """Ensure ModelExporter exports TorchScript models successfully."""
    from neuroswarm.search_space.dynamic_builder import DynamicNeuralNetwork

    dag_space = DAGSearchSpace(min_nodes=4, max_nodes=6)
    dag = dag_space.sample_random_dag()

    model = DynamicNeuralNetwork(
        dag=dag,
        in_channels=3,
        base_channels=16,
        num_classes=5,
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        exporter = ModelExporter(output_dir=tmp_dir)
        ts_path = exporter.export_to_torchscript(
            model=model,
            input_shape=(1, 3, 32, 32),
            filename="test_model.pt",
        )
        assert os.path.exists(ts_path)
