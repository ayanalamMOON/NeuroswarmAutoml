"""
Multi-Objective Pareto Optimization Utilities.

Provides non-dominated sorting, crowding distance calculations, and Pareto frontier
extraction for balancing validation accuracy against hardware efficiency metrics (FLOPs / Parameters).
"""

from typing import List
from neuroswarm.core.candidate import Candidate


def dominates(cand_a: Candidate, cand_b: Candidate) -> bool:
    """
    Checks if Candidate A Pareto-dominates Candidate B.

    Objectives:
    1. Maximize Fitness (Accuracy) -> Higher is better
    2. Minimize FLOPs -> Lower is better
    3. Minimize Parameter Count -> Lower is better
    """
    # A is at least as good as B across all objectives
    acc_better_or_equal = cand_a.fitness >= cand_b.fitness
    flops_better_or_equal = cand_a.flops <= cand_b.flops
    params_better_or_equal = cand_a.param_count <= cand_b.param_count

    # A is strictly better than B in at least one objective
    acc_strictly_better = cand_a.fitness > cand_b.fitness
    flops_strictly_better = cand_a.flops < cand_b.flops
    params_strictly_better = cand_a.param_count < cand_b.param_count

    at_least_as_good = acc_better_or_equal and flops_better_or_equal and params_better_or_equal
    strictly_better = acc_strictly_better or flops_strictly_better or params_strictly_better

    return at_least_as_good and strictly_better


def fast_non_dominated_sort(candidates: List[Candidate]) -> List[List[Candidate]]:
    """
    Executes NSGA-II fast non-dominated sorting algorithm on candidate population.

    Returns:
        List[List[Candidate]]: List of Pareto fronts, where front[0] is the non-dominated Pareto set.
    """
    fronts: List[List[Candidate]] = [[]]
    domination_counts = {c.candidate_id: 0 for c in candidates}
    dominated_solutions = {c.candidate_id: [] for c in candidates}

    for i, p in enumerate(candidates):
        for q in candidates[i + 1 :]:
            if dominates(p, q):
                dominated_solutions[p.candidate_id].append(q)
                domination_counts[q.candidate_id] += 1
            elif dominates(q, p):
                dominated_solutions[q.candidate_id].append(p)
                domination_counts[p.candidate_id] += 1

        if domination_counts[p.candidate_id] == 0:
            fronts[0].append(p)

    i = 0
    while len(fronts[i]) > 0:
        next_front = []
        for p in fronts[i]:
            for q in dominated_solutions[p.candidate_id]:
                domination_counts[q.candidate_id] -= 1
                if domination_counts[q.candidate_id] == 0:
                    next_front.append(q)
        i += 1
        fronts.append(next_front)

    if not fronts[-1]:
        fronts.pop()

    return fronts


def calculate_crowding_distance(front: List[Candidate]) -> List[Candidate]:
    """
    Calculates crowding distance metrics for candidates within a single Pareto front
    to maintain population diversity during multi-objective selection.
    """
    length = len(front)
    if length == 0:
        return []
    if length <= 2:
        for cand in front:
            cand.uncertainty = float("inf")
        return front

    distances = {c.candidate_id: 0.0 for c in front}

    # Evaluate across 3 objectives: Fitness (Max), FLOPs (Min), Params (Min)
    objectives = [("fitness", True), ("flops", False), ("param_count", False)]

    for attr, maximize in objectives:
        sorted_front = sorted(front, key=lambda c: getattr(c, attr), reverse=maximize)

        # Boundary elements get infinite distance priority
        distances[sorted_front[0].candidate_id] = float("inf")
        distances[sorted_front[-1].candidate_id] = float("inf")

        val_range = getattr(sorted_front[0], attr) - getattr(sorted_front[-1], attr)
        if val_range == 0:
            continue

        for i in range(1, length - 1):
            if distances[sorted_front[i].candidate_id] != float("inf"):
                prev_val = getattr(sorted_front[i - 1], attr)
                next_val = getattr(sorted_front[i + 1], attr)
                distances[sorted_front[i].candidate_id] += abs(next_val - prev_val) / abs(val_range)

    return front


def get_pareto_front(candidates: List[Candidate]) -> List[Candidate]:
    """Extracts the first (non-dominated) Pareto front from a population."""
    fronts = fast_non_dominated_sort(candidates)
    return fronts[0] if fronts else []
