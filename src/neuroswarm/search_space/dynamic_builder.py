"""
Dynamic PyTorch Graph Compiler.

Compiles arbitrary NetworkX DAG topologies into executable PyTorch nn.Modules, handling
channel dimension alignments, skip connection additions, and topological execution order.
"""

from typing import Dict, List, Tuple, Any
import networkx as nx
import torch
import torch.nn as nn


class ResNetBlock(nn.Module):
    """Standard Residual Conv Block for DAG nodes."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + residual)


class DynamicOpNode(nn.Module):
    """
    Wraps individual layer operations within the execution graph.
    """
    def __init__(self, op_type: str, channels: int):
        super().__init__()
        self.op_type = op_type

        if op_type == "conv3x3":
            self.op = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)
            )
        elif op_type == "conv5x5":
            self.op = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=5, padding=2, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)
            )
        elif op_type == "depthwise_conv":
            self.op = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
                nn.Conv2d(channels, channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)
            )
        elif op_type == "resnet_block":
            self.op = ResNetBlock(channels)
        elif op_type == "identity":
            self.op = nn.Identity()
        else:
            raise ValueError(f"Unsupported operation type: {op_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class DynamicNeuralNetwork(nn.Module):
    """
    Compiles a NetworkX DAG into a fully functional, end-to-end PyTorch neural network.
    """

    def __init__(
        self,
        dag: nx.DiGraph,
        in_channels: int = 3,
        base_channels: int = 32,
        num_classes: int = 10,
    ):
        super().__init__()
        self.dag = dag.copy()
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.num_classes = num_classes

        # Validate DAG topological order
        self.topological_order: List[int] = list(nx.topological_sort(self.dag))

        # Initial STEM layer to project raw input image channels
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )

        # Dynamic Node operations
        self.node_ops = nn.ModuleDict()
        for node in self.topological_order:
            op_type = self.dag.nodes[node].get("op", "conv3x3")
            self.node_ops[str(node)] = DynamicOpNode(op_type, base_channels)

        # Final Classification Head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(base_channels, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Executes DAG forward propagation via tensor aggregation over edge dependencies.
        """
        node_outputs: Dict[int, torch.Tensor] = {}
        stem_out = self.stem(x)

        for node in self.topological_order:
            in_edges = list(self.dag.in_edges(node))

            if not in_edges:
                # Source node receives stem features
                node_input = stem_out
            else:
                # Intermediate nodes aggregate input tensors from parent nodes
                parent_tensors = [node_outputs[u] for u, _ in in_edges]
                if len(parent_tensors) == 1:
                    node_input = parent_tensors[0]
                else:
                    # Sum element-wise across multiple incoming edges (skip connections)
                    node_input = torch.stack(parent_tensors, dim=0).sum(dim=0)

            # Apply dynamic layer operation
            node_outputs[node] = self.node_ops[str(node)](node_input)

        # Aggregate sink node outputs (nodes without outgoing edges)
        sink_nodes = [n for n in self.topological_order if self.dag.out_degree(n) == 0]
        if len(sink_nodes) == 1:
            final_features = node_outputs[sink_nodes[0]]
        else:
            sink_tensors = [node_outputs[n] for n in sink_nodes]
            final_features = torch.stack(sink_tensors, dim=0).sum(dim=0)

        return self.head(final_features)
