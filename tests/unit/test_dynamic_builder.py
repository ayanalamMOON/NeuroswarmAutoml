"""
Unit tests for Dynamic PyTorch Neural Network compilation and execution.
"""

import pytest
import torch
import networkx as nx

from neuroswarm.search_space.dynamic_builder import (
    DynamicNeuralNetwork,
    DynamicOpNode,
    ResNetBlock,
)
from neuroswarm.search_space.dag_space import DAGSearchSpace


def test_resnet_block_forward():
    """Ensure ResNetBlock executes forward pass preserving tensor dimensions."""
    block = ResNetBlock(channels=16)
    x = torch.randn(2, 16, 8, 8)
    out = block(x)
    assert out.shape == (2, 16, 8, 8)


@pytest.mark.parametrize("op_type", [
    "conv3x3",
    "conv5x5",
    "depthwise_conv",
    "resnet_block",
    "identity",
])
def test_dynamic_op_nodes(op_type):
    """Ensure all dynamic operation types maintain channel and spatial dimensions."""
    node = DynamicOpNode(op_type=op_type, channels=16)
    x = torch.randn(2, 16, 8, 8)
    out = node(x)
    assert out.shape == (2, 16, 8, 8)


def test_dynamic_op_node_invalid():
    """Ensure unsupported op types raise ValueError."""
    with pytest.raises(ValueError, match="Unsupported operation type"):
        DynamicOpNode(op_type="unknown_op", channels=16)


def test_dynamic_neural_network_forward():
    """Ensure compiled DAG network produces valid output logits."""
    dag_space = DAGSearchSpace(min_nodes=4, max_nodes=6)
    dag = dag_space.sample_random_dag()

    model = DynamicNeuralNetwork(
        dag=dag,
        in_channels=3,
        base_channels=16,
        num_classes=10,
    )

    x = torch.randn(4, 3, 32, 32)
    out = model(x)
    assert out.shape == (4, 10), f"Expected shape (4, 10), got {out.shape}"


def test_dynamic_neural_network_backward():
    """Ensure backward gradient propagation works through the compiled DAG."""
    dag = nx.DiGraph()
    dag.add_node(0, op="conv3x3")
    dag.add_node(1, op="resnet_block")
    dag.add_node(2, op="identity")
    dag.add_node(3, op="depthwise_conv")
    dag.add_edges_from([(0, 1), (0, 2), (1, 3), (2, 3)])

    model = DynamicNeuralNetwork(
        dag=dag,
        in_channels=3,
        base_channels=16,
        num_classes=5,
    )

    x = torch.randn(2, 3, 16, 16)
    target = torch.tensor([1, 3], dtype=torch.long)
    criterion = torch.nn.CrossEntropyLoss()

    output = model(x)
    loss = criterion(output, target)
    loss.backward()

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradient missing for parameter {name}"

