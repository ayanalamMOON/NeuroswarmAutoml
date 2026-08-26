"""
Unit tests for DAG Search Space and graph validation/repair mechanics.
"""

import pytest
import networkx as nx
from neuroswarm.search_space.dag_space import DAGSearchSpace


@pytest.fixture
def dag_space():
    return DAGSearchSpace(min_nodes=4, max_nodes=8, edge_prob=0.5)


def test_sample_random_dag_is_valid(dag_space):
    """Ensure sampled DAGs are directed acyclic graphs and have valid source-to-sink paths."""
    for _ in range(20):
        g = dag_space.sample_random_dag()
        assert nx.is_directed_acyclic_graph(g), "Sampled graph contains cycles"
        assert dag_space.validate_dag(g), "Sampled graph failed validation"
        assert dag_space.min_nodes <= g.number_of_nodes() <= dag_space.max_nodes


def test_validate_dag_detects_cycles(dag_space):
    """Ensure validation fails for graphs containing cycles."""
    cyclic_graph = nx.DiGraph()
    cyclic_graph.add_edges_from([(0, 1), (1, 2), (2, 0)])
    assert not dag_space.validate_dag(cyclic_graph)


def test_validate_dag_detects_disconnected_sink(dag_space):
    """Ensure validation fails when sink is unreachable from source."""
    disconnected = nx.DiGraph()
    disconnected.add_node(0, op="conv3x3")
    disconnected.add_node(1, op="conv3x3")
    disconnected.add_node(2, op="conv3x3")
    disconnected.add_edge(0, 1)
    assert not dag_space.validate_dag(disconnected)


def test_repair_dag_removes_cycles(dag_space):
    """Ensure repair_dag removes cycles and ensures connectivity."""
    cyclic_graph = nx.DiGraph()
    cyclic_graph.add_node(0, op="conv3x3")
    cyclic_graph.add_node(1, op="conv3x3")
    cyclic_graph.add_node(2, op="conv3x3")
    cyclic_graph.add_edges_from([(0, 1), (1, 2), (2, 1)])

    repaired = dag_space.repair_dag(cyclic_graph)
    assert nx.is_directed_acyclic_graph(repaired)
    assert dag_space.validate_dag(repaired)


def test_repair_dag_connects_orphans(dag_space):
    """Ensure orphaned nodes are connected during repair."""
    orphan_graph = nx.DiGraph()
    orphan_graph.add_node(0, op="conv3x3")
    orphan_graph.add_node(1, op="conv5x5")
    orphan_graph.add_node(2, op="identity")
    orphan_graph.add_node(3, op="resnet_block")
    orphan_graph.add_edge(0, 3)

    repaired = dag_space.repair_dag(orphan_graph)
    assert nx.is_directed_acyclic_graph(repaired)
    assert dag_space.validate_dag(repaired)
    for n in repaired.nodes():
        assert repaired.in_degree(n) + repaired.out_degree(n) > 0

