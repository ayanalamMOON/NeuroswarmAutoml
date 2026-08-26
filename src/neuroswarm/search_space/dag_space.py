"""
Directed Acyclic Graph Search Space Manager.

Generates, validates, and manipulates search space DAG structures for neural architectures.
"""

from typing import List, Optional, Dict, Any
import random
import networkx as nx


class DAGSearchSpace:
    """
    Manages structural constraints, node operation assignments, and validity checks for DAGs.
    """

    DEFAULT_OPERATIONS = [
        "conv3x3",
        "conv5x5",
        "depthwise_conv",
        "resnet_block",
        "identity",
    ]

    def __init__(
        self,
        min_nodes: int = 4,
        max_nodes: int = 10,
        available_ops: Optional[List[str]] = None,
        edge_prob: float = 0.4,
    ):
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.available_ops = available_ops or self.DEFAULT_OPERATIONS
        self.edge_prob = edge_prob

    def sample_random_dag(self) -> nx.DiGraph:
        """
        Generates a valid, topological DAG with random op nodes and connectivity.
        """
        num_nodes = random.randint(self.min_nodes, self.max_nodes)
        g = nx.DiGraph()

        # Add topologically ordered nodes
        for i in range(num_nodes):
            g.add_node(i, op=random.choice(self.available_ops))

        # Add forward edges ensuring u < v to prevent cycles
        for u in range(num_nodes):
            for v in range(u + 1, num_nodes):
                if random.random() < self.edge_prob:
                    g.add_edge(u, v)

        # Force connectivity from source (0) to sink (num_nodes - 1)
        self.repair_dag(g)
        return g

    def validate_dag(self, g: nx.DiGraph) -> bool:
        """
        Verifies that a graph is a valid DAG with reachable input/output paths.
        """
        if not nx.is_directed_acyclic_graph(g):
            return False

        nodes = sorted(list(g.nodes()))
        if len(nodes) < 2:
            return False

        source, sink = nodes[0], nodes[-1]

        # Check path existence from source to sink
        if not nx.has_path(g, source, sink):
            return False

        return True

    def repair_dag(self, g: nx.DiGraph) -> nx.DiGraph:
        """
        Fixes cycles, orphaned nodes, and isolated subgraphs to guarantee executable execution graphs.
        """
        # 1. Eliminate self loops and cycles
        g.remove_edges_from(list(nx.selfloop_edges(g)))
        while not nx.is_directed_acyclic_graph(g):
            try:
                cycle = nx.find_cycle(g)
                g.remove_edge(*cycle[0][:2])
            except nx.NetworkXNoCycle:
                break

        nodes = sorted(list(g.nodes()))
        if len(nodes) < 2:
            return g

        source, sink = nodes[0], nodes[-1]

        # 2. Connect orphaned internal nodes safely
        for n in nodes:
            if n != source and g.in_degree(n) == 0:
                candidates = [
                    prev for prev in nodes if prev < n and not nx.has_path(g, n, prev)
                ]
                if candidates:
                    g.add_edge(random.choice(candidates), n)
                elif not nx.has_path(g, n, source):
                    g.add_edge(source, n)

            if n != sink and g.out_degree(n) == 0:
                candidates = [
                    nxt for nxt in nodes if nxt > n and not nx.has_path(g, nxt, n)
                ]
                if candidates:
                    g.add_edge(n, random.choice(candidates))
                elif not nx.has_path(g, sink, n):
                    g.add_edge(n, sink)

        # 3. Ensure direct or indirect path from source to sink
        if not nx.has_path(g, source, sink):
            if not nx.has_path(g, sink, source):
                g.add_edge(source, sink)
            else:
                while nx.has_path(g, sink, source):
                    path = nx.shortest_path(g, sink, source)
                    g.remove_edge(path[0], path[1])
                g.add_edge(source, sink)

        # 4. Final cycle elimination safeguard
        while not nx.is_directed_acyclic_graph(g):
            try:
                cycle = nx.find_cycle(g)
                g.remove_edge(*cycle[0][:2])
            except nx.NetworkXNoCycle:
                break

        return g
