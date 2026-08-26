"""
Bi-Level Co-Evolutionary Optimization Engine with Telemetry and Hardware-Aware Constraints.

Synchronizes Upper-Level Topology Genetic Algorithm (GA) and Lower-Level Continuous
Particle Swarm Optimization / Differential Evolution (PSO-DE) with Gaussian Process surrogates,
UCB acquisition selection, hardware latency constraints, and real-time telemetry streaming.
"""

import logging
from typing import List, Dict, Any, Callable, Optional
import numpy as np

from neuroswarm.core.candidate import Candidate
from neuroswarm.optimizers.ga_topology import TopologyGAOptimizer
from neuroswarm.optimizers.pso_de_continuous import ContinuousPSODE
from neuroswarm.surrogates.base_surrogate import BaseSurrogateModel
from neuroswarm.surrogates.gp_estimator import GaussianProcessSurrogate

try:
    pass

    HAS_TELEMETRY = True
except ImportError:
    HAS_TELEMETRY = False

logger = logging.getLogger("neuroswarm.bilevel")


class BiLevelCoEvolutionEngine:
    """
    Coordinating runtime orchestrating bi-level structural (GA) and hyperparameter (PSO-DE) evolution
    with surrogate UCB acquisition, hardware-aware penalty constraints, and real-time telemetry logging.
    """

    def __init__(
        self,
        population_size: int = 20,
        ga_optimizer: Optional[TopologyGAOptimizer] = None,
        pso_optimizer: Optional[ContinuousPSODE] = None,
        surrogate: Optional[BaseSurrogateModel] = None,
        telemetry: Optional[Any] = None,
        target_latency_ms: float = 0.0,
        latency_alpha: float = 0.05,
        target_flops: int = 0,
        flops_beta: float = 0.0,
    ):
        self.population_size = population_size
        self.ga_opt = ga_optimizer or TopologyGAOptimizer(population_size=population_size)
        self.pso_opt = pso_optimizer or ContinuousPSODE(population_size=population_size)
        self.surrogate = surrogate or GaussianProcessSurrogate()
        self.telemetry = telemetry

        # Hardware-Aware Penalty Constraints
        self.target_latency_ms = target_latency_ms
        self.latency_alpha = latency_alpha
        self.target_flops = target_flops
        self.flops_beta = flops_beta
        self.use_hardware_aware = target_latency_ms > 0.0 or target_flops > 0

        self.global_best_candidate: Optional[Candidate] = None
        self.history: List[Dict[str, Any]] = []

    def _eval_ground_truth(
        self,
        candidate: Candidate,
        eval_fn: Callable,
        epochs: int,
        dataset_config: Dict[str, Any],
    ) -> Candidate:
        """Executes ground-truth training and calculates hardware-aware fitness score."""
        res = eval_fn(candidate, epochs=epochs, config=dataset_config)

        # Flexible unpacking for 3-tuple (acc, params, flops) or 4-tuple (acc, params, flops, latency_ms)
        if len(res) == 4:
            val_acc, params, flops, latency_ms = res
        else:
            val_acc, params, flops = res
            latency_ms = candidate.latency_ms

        candidate.fitness = val_acc
        candidate.param_count = params
        candidate.flops = flops
        candidate.latency_ms = latency_ms
        candidate.is_ground_truth = True
        candidate.uncertainty = 0.0
        candidate.evaluated_epochs += epochs

        # Calculate penalized hardware fitness score and update pbest
        candidate.compute_hardware_aware_fitness(
            target_latency_ms=self.target_latency_ms,
            alpha=self.latency_alpha,
            target_flops=self.target_flops,
            beta=self.flops_beta,
        )
        candidate.update_pbest(use_constrained=self.use_hardware_aware)
        return candidate

    def run_generation(
        self,
        population: List[Candidate],
        current_gen: int,
        max_gens: int,
        eval_fn: Callable,
        dataset_config: Dict[str, Any],
        short_epochs: int = 5,
        ucb_kappa: float = 1.96,
    ) -> List[Candidate]:
        """
        Executes a bi-level evolutionary step:
        1. Evolve continuous hyperparameters via PSO-DE.
        2. Evolve discrete DAG topologies via GA.
        3. Filter candidates through Gaussian Process surrogate UCB selection.
        4. Train selected candidates for short epochs with hardware-aware penalties.
        5. Stream VRAM and generation progress to telemetry manager.
        """
        logger.info(
            f"\n--- Bi-Level Generation [{current_gen}/{max_gens}] (Hardware-Aware: {self.use_hardware_aware}) ---"
        )

        # Step 1: Continuous Parameter Velocity Update (PSO-DE)
        try:
            population = self.pso_opt.step(
                population,
                current_gen,
                max_gens,
                global_best=self.global_best_candidate,
                use_constrained=self.use_hardware_aware,
            )
        except TypeError:
            try:
                population = self.pso_opt.step(
                    population=population,
                    global_best=self.global_best_candidate,
                    use_constrained=self.use_hardware_aware,
                )
            except TypeError:
                population = self.pso_opt.step(population, current_gen, max_gens)

        # Step 2: Discrete Topology Evolution (GA)
        try:
            population = self.ga_opt.step(
                population,
                current_gen,
                max_gens,
                use_constrained=self.use_hardware_aware,
            )
        except TypeError:
            try:
                population = self.ga_opt.step(population=population, use_constrained=self.use_hardware_aware)
            except TypeError:
                population = self.ga_opt.step(population, current_gen, max_gens)

        # Step 3: Surrogate Filtering & Acquisition Scoring
        if getattr(self.surrogate, "is_fitted", False):
            predictions = [self.surrogate.predict(cand) for cand in population]
            for cand, (mu, sigma) in zip(population, predictions):
                cand.uncertainty = sigma
                # Upper Confidence Bound (UCB) acquisition
                if not cand.is_ground_truth:
                    cand.fitness = mu + ucb_kappa * sigma
                    cand.compute_hardware_aware_fitness(
                        target_latency_ms=self.target_latency_ms,
                        alpha=self.latency_alpha,
                        target_flops=self.target_flops,
                        beta=self.flops_beta,
                    )

            # Sort by effective fitness to select top candidates for ground-truth evaluation
            population.sort(
                key=lambda c: c.effective_fitness(use_constrained=self.use_hardware_aware),
                reverse=True,
            )
            eval_cutoff = max(2, int(self.population_size * 0.20))
            candidates_to_eval = [c for c in population[:eval_cutoff] if not c.is_ground_truth]
        else:
            # Warm-start phase: evaluate uninitialized candidates
            candidates_to_eval = [c for c in population if not c.is_ground_truth]

        # Step 4: Ground-truth Training Phase
        if candidates_to_eval:
            for cand in candidates_to_eval:
                self._eval_ground_truth(cand, eval_fn, short_epochs, dataset_config)

            # Refit surrogate with newly evaluated samples
            evaluated_samples = [c for c in population if c.is_ground_truth]
            if len(evaluated_samples) >= 5:
                try:
                    self.surrogate.fit(evaluated_samples)
                    logger.info(f"Surrogate GP refitted on {len(evaluated_samples)} ground truth samples.")
                except Exception as e:
                    logger.warning(f"Surrogate GP fitting skipped ({e}).")

        # Track global best candidate based on effective fitness
        current_best = max(
            population,
            key=lambda c: c.effective_fitness(use_constrained=self.use_hardware_aware),
        )
        is_new_discovery = False

        if self.global_best_candidate is None or current_best.effective_fitness(
            self.use_hardware_aware
        ) > self.global_best_candidate.effective_fitness(self.use_hardware_aware):
            self.global_best_candidate = current_best.clone()
            is_new_discovery = True

        # Log Metrics
        best_cand = self.global_best_candidate
        gen_metrics = {
            "generation": current_gen,
            "best_fitness": best_cand.fitness if best_cand else 0.0,
            "best_constrained_fitness": (best_cand.constrained_fitness if best_cand else 0.0),
            "best_latency_ms": best_cand.latency_ms if best_cand else 0.0,
            "best_params": best_cand.param_count if best_cand else 0,
            "mean_fitness": float(np.mean([c.effective_fitness(self.use_hardware_aware) for c in population])),
            "surrogate_fitted": getattr(self.surrogate, "is_fitted", False),
            "best_candidate_id": best_cand.candidate_id if best_cand else "N/A",
        }
        self.history.append(gen_metrics)

        # Stream Generation Metrics & Webhook Alerts to Telemetry Manager
        if self.telemetry:
            try:
                self.telemetry.log_generation(gen_metrics)
                if is_new_discovery and best_cand:
                    self.telemetry.notify_pareto_discovery(
                        candidate_id=best_cand.candidate_id,
                        accuracy=best_cand.fitness,
                        latency_ms=best_cand.latency_ms,
                        params=best_cand.param_count,
                    )
            except Exception as e:
                logger.warning(f"Telemetry logging encountered an error: {e}")

        logger.info(
            f"Gen [{current_gen:02d}/{max_gens:02d}] Global Best ID: {gen_metrics['best_candidate_id']} | "
            f"Acc: {gen_metrics['best_fitness']:.4f} | Constrained: {gen_metrics['best_constrained_fitness']:.4f} | "
            f"Latency: {gen_metrics['best_latency_ms']:.2f}ms"
        )

        return population
