"""
Genetic Algorithm Topology Optimizer.

Handles structural evolution of the Directed Acyclic Graph (DAG) search space via
single-point matrix/subgraph crossovers, edge toggling, node insertion/deletion,
node-type mutations, and hardware-aware fitness evaluation.
"""

import copy
import logging
import random
from typing import List, Tuple, Optional
import networkx as nx
import numpy as np

from neuroswarm.optimizers.base_optimizer import BaseOptimizer
from neuroswarm.core.candidate import Candidate

logger = logging.getLogger("neuroswarm.ga_topology")


class TopologyGAOptimizer(BaseOptimizer):
    """
    GA Engine executing graph structural crossover and mutation on candidate DAGs.
    """

    def __init__(
        self,
        population_size: int = 8,
        crossover_prob: float = 0.7,
        mutation_prob: float = 0.3,
        tournament_size: int = 3,
        available_ops: Optional[List[str]] = None,
        min_nodes: int = 4,
        max_nodes: int = 10,
        **kwargs
    ):
        super().__init__(population_size)

        # Parameter aliases for compatibility across runner configurations
        self.crossover_prob = kwargs.get("crossover_rate", crossover_prob)
        self.crossover_rate = self.crossover_prob
        self.mutation_prob = kwargs.get("mutation_rate", mutation_prob)
        self.mutation_rate = self.mutation_prob

        self.tournament_size = tournament_size
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.available_ops = available_ops or [
            "conv3x3",
            "conv5x5",
            "depthwise_conv",
            "resnet_block",
            "identity",
        ]

    def step(
        self,
        population: List[Candidate],
        current_gen: int = 1,
        max_gens: int = 10,
        use_constrained: bool = False,
        **kwargs
    ) -> List[Candidate]:
        """Performs tournament selection, crossover, and graph mutations."""
        if not population:
            return []

        if len(population) < 2:
            logger.warning("Population too small for crossover. Returning cloned population.")
            return [c.clone() for c in population]

        # Sort population by effective fitness (raw or hardware-constrained)
        sorted_pop = sorted(
            population,
            key=lambda c: c.effective_fitness(use_constrained=use_constrained),
            reverse=True
        )
        next_population: List[Candidate] = []

        # Elitism: Retain top performers un-mutated
        elite_count = max(1, int(len(population) * 0.10))
        for elite in sorted_pop[:elite_count]:
            next_population.append(elite.clone())

        # Generate offspring for remaining slots
        while len(next_population) < len(population):
            p1 = self._tournament_select(population, k=self.tournament_size, use_constrained=use_constrained)
            p2 = self._tournament_select(population, k=self.tournament_size, use_constrained=use_constrained)

            if random.random() < self.crossover_prob:
                child_graph1, child_graph2 = self._subgraph_crossover(p1.graph, p2.graph)
            else:
                child_graph1, child_graph2 = p1.graph.copy(), p2.graph.copy()

            for child_g in (child_graph1, child_graph2):
                if len(next_population) >= len(population):
                    break

                if random.random() < self.mutation_prob:
                    child_g = self._mutate_graph(child_g)

                # Inherit hyperparameter space state from parent p1
                child_cand = p1.clone()
                child_cand.graph = child_g
                child_cand.is_ground_truth = False
                child_cand.fitness = float("-inf")
                child_cand.constrained_fitness = float("-inf")
                next_population.append(child_cand)

        return next_population[:len(population)]

    def _tournament_select(
        self,
        population: List[Candidate],
        k: int = 3,
        use_constrained: bool = False
    ) -> Candidate:
        """Tournament selection operator safely handling small population sizes."""
        if not population:
            raise ValueError("Population is empty during tournament selection.")

        # Dynamically clamp tournament size k to available population size
        actual_k = max(1, min(k, len(population)))
        selected = random.sample(population, actual_k)

        return max(selected, key=lambda c: c.effective_fitness(use_constrained=use_constrained))

    def _subgraph_crossover(self, g1: nx.DiGraph, g2: nx.DiGraph) -> Tuple[nx.DiGraph, nx.DiGraph]:
        """Swaps topological subgraph layers between parents while ensuring DAG validity."""
        c1, c2 = g1.copy(), g2.copy()

        nodes1 = list(c1.nodes())
        nodes2 = list(c2.nodes())

        # Mid-point crossover swap for internal node attributes
        min_len = min(len(nodes1), len(nodes2))
        if min_len > 2:
            cut_pt = random.randint(1, min_len - 1)
            for i in range(cut_pt):
                n1, n2 = nodes1[i], nodes2[i]
                op1 = c1.nodes[n1].get("op", random.choice(self.available_ops))
                op2 = c2.nodes[n2].get("op", random.choice(self.available_ops))
                c1.nodes[n1]["op"] = op2
                c2.nodes[n2]["op"] = op1

        self._ensure_dag_sanity(c1)
        self._ensure_dag_sanity(c2)
        return c1, c2

    def _mutate_graph(self, g: nx.DiGraph) -> nx.DiGraph:
        """Applies edge toggling, operation swapping, or node additions/deletions."""
        g_mut = g.copy()
        mutation_type = random.choice(["toggle_edge", "mutate_op", "add_node"])

        nodes = sorted(list(g_mut.nodes()))

        if mutation_type == "toggle_edge" and len(nodes) > 1:
            u, v = random.sample(nodes, 2)
            if u > v:  # Maintain DAG directionality order (u -> v)
                u, v = v, u

            if g_mut.has_edge(u, v):
                g_mut.remove_edge(u, v)
            else:
                g_mut.add_edge(u, v)

        elif mutation_type == "mutate_op" and len(nodes) > 0:
            target_node = random.choice(nodes)
            g_mut.nodes[target_node]["op"] = random.choice(self.available_ops)

        elif mutation_type == "add_node" and len(nodes) < self.max_nodes:
            new_id = max(nodes) + 1 if nodes else 0
            g_mut.add_node(new_id, op=random.choice(self.available_ops))

            # Connect to existing nodes ensuring DAG flow
            target = random.choice(nodes)
            if target < new_id:
                g_mut.add_edge(target, new_id)
            else:
                g_mut.add_edge(new_id, target)

        self._ensure_dag_sanity(g_mut)
        return g_mut

    def _ensure_dag_sanity(self, g: nx.DiGraph):
        """Fixes cycles or disconnected components in the NetworkX structure."""
        g.remove_edges_from(list(nx.selfloop_edges(g)))
        while not nx.is_directed_acyclic_graph(g):
            try:
                cycle = nx.find_cycle(g)
                g.remove_edge(*cycle[0][:2])
            except nx.NetworkXNoCycle:
                break

        nodes = sorted(list(g.nodes()))
        if len(nodes) < 2:
            return

        in_0, out_max = nodes[0], nodes[-1]
        for n in nodes[1:]:
            if g.in_degree(n) == 0:
                if not nx.has_path(g, n, in_0):
                    g.add_edge(in_0, n)
            if g.out_degree(n) == 0 and n != out_max:
                if not nx.has_path(g, out_max, n):
                    g.add_edge(n, out_max)

        while not nx.is_directed_acyclic_graph(g):
            try:
                cycle = nx.find_cycle(g)
                g.remove_edge(*cycle[0][:2])
            except nx.NetworkXNoCycle:
                break
