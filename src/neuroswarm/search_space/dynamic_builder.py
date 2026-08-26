"""
Dynamic PyTorch Graph Compiler & Search Space Module.

Compiles arbitrary NetworkX DAG topologies into executable PyTorch nn.Modules.
Handles topological execution order, channel dimension alignments via 1x1 convolutions,
multi-parent feature aggregation, and advanced neural primitives (Depthwise-Separable Convs,
Squeeze-and-Excitation, and Inverted Residual MBConv blocks).
"""

from typing import Dict, List, Tuple, Any, Optional
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================================
# Advanced Neural Primitives
# =====================================================================

class DepthwiseSeparableConv(nn.Module):
    """3x3 or 5x5 Depthwise Separable Convolution with BatchNorm and ReLU."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        padding = kernel_size // 2
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.pointwise(self.depthwise(x))))


class SqueezeAndExcitation(nn.Module):
    """Squeeze-and-Excitation (SE) Channel Attention Block."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        reduced_dim = max(4, channels // reduction)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, reduced_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_dim, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        scale = self.fc(x).view(b, c, 1, 1)
        return x * scale


class InvertedResidualBlock(nn.Module):
    """MobileNetV2-style Inverted Residual Bottleneck (MBConv) with SE attention."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expand_ratio: int = 4,
        use_se: bool = True
    ):
        super().__init__()
        self.stride = stride
        self.use_residual = (stride == 1 and in_channels == out_channels)
        hidden_dim = int(in_channels * expand_ratio)

        layers: List[nn.Module] = []
        # Pointwise Expansion
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True)
            ])

        # Depthwise Convolution
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True)
        ])

        # Squeeze-and-Excitation Attention
        if use_se:
            layers.append(SqueezeAndExcitation(hidden_dim, reduction=8))

        # Pointwise Linear Projection
        layers.extend([
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_residual:
            return x + self.conv(x)
        return self.conv(x)


class ResNetBlock(nn.Module):
    """Standard Residual Convolutional Block with optional shortcut alignment."""

    def __init__(self, in_channels: int, out_channels: Optional[int] = None, stride: int = 1):
        super().__init__()
        out_channels = out_channels or in_channels
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + residual)


# =====================================================================
# Factory Dispatcher & Dynamic Op Wrapper
# =====================================================================

def build_primitive_op(op_name: str, in_ch: int, out_ch: int) -> nn.Module:
    """Instantiates neural primitives based on operation string identifier."""
    op_clean = op_name.lower().strip()
    if op_clean in ["conv3x3", "conv"]:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    elif op_clean == "conv5x5":
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    elif op_clean in ["depthwise_conv", "dw_conv3x3"]:
        return DepthwiseSeparableConv(in_ch, out_ch, kernel_size=3)
    elif op_clean in ["resnet_block", "resnet"]:
        return ResNetBlock(in_ch, out_ch)
    elif op_clean in ["mbconv", "inverted_residual"]:
        return InvertedResidualBlock(in_ch, out_ch, expand_ratio=4, use_se=True)
    elif op_clean in ["se_block", "squeeze_excitation"]:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            SqueezeAndExcitation(out_ch)
        )
    elif op_clean == "identity":
        if in_ch == out_ch:
            return nn.Identity()
        return nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
    else:
        # Default fallback to standard 3x3 conv
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )


class DynamicOpNode(nn.Module):
    """Wraps dynamic operations within individual execution graph nodes."""

    def __init__(self, op_type: str, channels: int):
        super().__init__()
        self.op_type = op_type
        self.op = build_primitive_op(op_type, channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


# =====================================================================
# Main Dynamic Graph Compiler
# =====================================================================

class DynamicNeuralNetwork(nn.Module):
    """
    Compiles a NetworkX DAG into a fully functional, end-to-end PyTorch neural network.
    Supports both element-wise summation and 1x1 projection merging for multi-parent nodes.
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

        # Validate and store topological execution order
        self.topological_order: List[int] = list(nx.topological_sort(self.dag))

        # Initial STEM layer to project raw input image channels
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )

        # Dynamic Node operations & multi-input aggregation projection layers
        self.node_ops = nn.ModuleDict()
        self.merge_convs = nn.ModuleDict()

        for node in self.topological_order:
            in_edges = list(self.dag.in_edges(node))
            op_type = self.dag.nodes[node].get("op", "conv3x3")

            # Channel alignment projection for multi-parent concatenation
            if len(in_edges) > 1:
                concat_ch = base_channels * len(in_edges)
                self.merge_convs[str(node)] = nn.Conv2d(concat_ch, base_channels, kernel_size=1, bias=False)

            self.node_ops[str(node)] = DynamicOpNode(op_type, base_channels)

        # Final Classification Head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(base_channels, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Executes DAG forward propagation via feature aggregation over edge dependencies.
        """
        node_outputs: Dict[int, torch.Tensor] = {}
        stem_out = self.stem(x)

        for node in self.topological_order:
            in_edges = list(self.dag.in_edges(node))

            if not in_edges:
                # Source nodes receive stem features
                node_input = stem_out
            elif len(in_edges) == 1:
                # Single parent dependency
                node_input = node_outputs[in_edges[0][0]]
            else:
                # Multi-parent feature aggregation via concatenation & 1x1 Conv projection
                parent_tensors = [node_outputs[u] for u, _ in in_edges]
                concat_feats = torch.cat(parent_tensors, dim=1)
                node_input = self.merge_convs[str(node)](concat_feats)

            # Apply dynamic node operation
            node_outputs[node] = self.node_ops[str(node)](node_input)

        # Aggregate sink node outputs (nodes without outgoing edges)
        sink_nodes = [n for n in self.topological_order if self.dag.out_degree(n) == 0]
        if len(sink_nodes) == 1:
            final_features = node_outputs[sink_nodes[0]]
        else:
            sink_tensors = [node_outputs[n] for n in sink_nodes]
            final_features = torch.stack(sink_tensors, dim=0).mean(dim=0)

        return self.head(final_features)


# Class alias for backwards compatibility
DynamicDAGNetwork = DynamicNeuralNetwork
