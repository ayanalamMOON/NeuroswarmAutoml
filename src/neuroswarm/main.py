"""
NeuroSwarm-AutoML Main Orchestrator with Scale-Up Search Space & CIFAR-100 Support.

CLI entrypoint executing bi-level co-evolutionary Neural Architecture Search (NAS)
with Gaussian Process surrogates, Pareto front extraction, and ONNX/TorchScript model exporting.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
from pathlib import Path
import random
import sys
import time
from typing import Dict, Any, List, Optional

# Add project root and src directory to sys.path for direct CLI execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_ROOT = Path(__file__).resolve().parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from neuroswarm.core.candidate import Candidate
from neuroswarm.core.runner import ParallelRunner, train_and_evaluate_candidate
from neuroswarm.optimizers.bilevel_engine import BiLevelCoEvolutionEngine
from neuroswarm.optimizers.ga_topology import TopologyGAOptimizer
from neuroswarm.optimizers.pso_de_continuous import ContinuousPSODE
from neuroswarm.search_space.dag_space import DAGSearchSpace
from neuroswarm.search_space.dynamic_builder import DynamicNeuralNetwork
from neuroswarm.surrogates.gp_estimator import GaussianProcessSurrogate
from neuroswarm.utils.export import ModelExporter
from neuroswarm.utils.pareto import get_pareto_front, fast_non_dominated_sort
from neuroswarm.utils.visualization import (
    plot_convergence_curve,
    plot_pareto_front,
    plot_dag_architecture,
)

# Configure Logging Format
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("neuroswarm.main")

Tuple_Dataset = Dict[str, Any]


def set_seed(seed: int = 42) -> None:
    """Sets deterministic random seeds across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_dataset_loaders(
    dataset_name: str,
    batch_size: int = 64,
    base_channels: int = 32,
    pin_memory: bool = True
) -> Tuple_Dataset:
    """Loads target dataset DataLoaders or returns synthetic fallbacks with CUDA pin_memory."""
    dataset_lower = dataset_name.lower()
    data_path = "./data/raw"

    if dataset_lower in ("cifar10", "cifar100"):
        try:
            if dataset_lower == "cifar100":
                mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
                num_classes = 100
                dataset_cls = datasets.CIFAR100
            else:
                mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                num_classes = 10
                dataset_cls = datasets.CIFAR10

            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])

            train_set = dataset_cls(root=data_path, train=True, download=True, transform=transform_train)
            val_set = dataset_cls(root=data_path, train=False, download=True, transform=transform_test)

            train_loader = DataLoader(
                train_set,
                batch_size=batch_size,
                shuffle=True,
                num_workers=2 if os.name != "nt" else 0,
                pin_memory=pin_memory
            )
            val_loader = DataLoader(
                val_set,
                batch_size=batch_size,
                shuffle=False,
                num_workers=2 if os.name != "nt" else 0,
                pin_memory=pin_memory
            )

            return {
                "train_loader": train_loader,
                "val_loader": val_loader,
                "in_channels": 3,
                "num_classes": num_classes,
                "base_channels": base_channels,
            }
        except Exception as e:
            logger.warning(f"Failed to load {dataset_name} ({e}). Falling back to synthetic.")

    logger.info("Using Synthetic Data Config for fast benchmarking.")
    return {
        "train_loader": None,
        "val_loader": None,
        "in_channels": 3,
        "num_classes": 10 if dataset_lower != "cifar100" else 100,
        "base_channels": base_channels,
    }


def parse_args() -> argparse.Namespace:
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description="NeuroSwarm-AutoML CLI Search Orchestrator")
    parser.add_argument("--generations", type=int, default=5, help="Total co-evolution generations")
    parser.add_argument("--population", type=int, default=8, help="Population size per generation")
    parser.add_argument("--short_epochs", type=int, default=2, help="Short training epochs for search phase")
    parser.add_argument("--final_epochs", type=int, default=10, help="Epochs for training final winning candidate")
    parser.add_argument("--dataset", type=str, default="synthetic", choices=["synthetic", "cifar10", "cifar100"], help="Dataset target")
    parser.add_argument("--min_nodes", type=int, default=4, help="Minimum DAG nodes")
    parser.add_argument("--max_nodes", type=int, default=8, help="Maximum DAG nodes")
    parser.add_argument("--base_channels", type=int, default=32, help="Base network width channels")
    parser.add_argument("--device", type=str, default=None, help="Target device (e.g. cuda, cuda:0, cpu)")
    parser.add_argument("--no_amp", action="store_true", help="Disable Automatic Mixed Precision (AMP)")
    parser.add_argument("--num_workers", type=int, default=2, help="Parallel process worker count")
    parser.add_argument("--use_ray", action="store_true", help="Enable distributed Ray cluster execution")
    parser.add_argument("--gpus_per_worker", type=float, default=1.0, help="GPUs allocated per Ray worker")
    parser.add_argument("--output_dir", type=str, default="./outputs_run", help="Output directory for logs and exports")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # CUDA Hardware Detection & Optimization
    has_cuda = torch.cuda.is_available()
    if args.device is not None:
        target_device = args.device
    else:
        target_device = "cuda" if has_cuda else "cpu"

    use_amp = (not args.no_amp) and ("cuda" in target_device and has_cuda)

    print("=" * 70)
    print(">>> NeuroSwarm-AutoML Search Engine Initializing...")
    print(f"Population: {args.population} | Generations: {args.generations}")
    print(f"Dataset: {args.dataset.upper()} | Target Device: {target_device.upper()} | AMP: {use_amp}")
    print(f"DAG Search Depth: Nodes [{args.min_nodes} -> {args.max_nodes}] | Base Channels: {args.base_channels}")
    if has_cuda:
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"CUDA Hardware: {gpu_name} ({vram_gb:.1f} GB VRAM) | Devices: {torch.cuda.device_count()}")
        torch.backends.cudnn.benchmark = True
    print(f"Ray Distributed: {args.use_ray} | Seed: {args.seed}")
    print("=" * 70)

    # Load Dataset Configuration
    dataset_config = get_dataset_loaders(args.dataset, base_channels=args.base_channels, pin_memory=has_cuda)
    dataset_config["use_amp"] = use_amp
    dataset_config["device"] = target_device

    # Initialize Execution Engine & Search Space
    runner = ParallelRunner(
        num_workers=args.num_workers,
        use_ray=args.use_ray,
        gpus_per_worker=args.gpus_per_worker
    )
    search_space = DAGSearchSpace(min_nodes=args.min_nodes, max_nodes=args.max_nodes)

    # Create Initial Population Pool
    population: List[Candidate] = []
    for _ in range(args.population):
        dag = search_space.sample_random_dag()
        hparams = np.array([
            random.uniform(-4.0, -1.0),
            random.uniform(0.8, 0.999),
            random.uniform(-6.0, -2.0),
            random.uniform(4.0, 8.0),
        ], dtype=np.float64)
        cand = Candidate(graph=dag, hyperparams=hparams)
        population.append(cand)

    # Initialize Bi-Level Co-Evolutionary Engine
    ga_opt = TopologyGAOptimizer(population_size=args.population)
    pso_opt = ContinuousPSODE(population_size=args.population)
    surrogate = GaussianProcessSurrogate()
    bilevel_engine = BiLevelCoEvolutionEngine(
        population_size=args.population,
        ga_optimizer=ga_opt,
        pso_optimizer=pso_opt,
        surrogate=surrogate
    )

    # Phase 1: Warm-Start Surrogate Evaluation
    warmstart_count = min(5, args.population)
    logger.info(f"\n[Phase 1] Warm-Starting Surrogate with {warmstart_count} candidates on {target_device.upper()}...")
    warmstart_batch = population[:warmstart_count]
    runner.evaluate_candidates(warmstart_batch, epochs=args.short_epochs, config=dataset_config)

    surrogate.fit(warmstart_batch)
    logger.info(f"Surrogate fitted: {surrogate.is_fitted}")

    # Phase 2: Bi-Level Co-Evolution Loop
    logger.info("\n[Phase 2] Starting Bi-Level Co-Evolution Loop...")
    start_time = time.time()

    for gen in range(1, args.generations + 1):
        gen_start = time.time()

        def eval_wrapper(cand: Candidate, epochs: int, config: Dict[str, Any]):
            val_acc, params, flops = train_and_evaluate_candidate(cand, epochs, config, device_str=target_device)
            return val_acc, params, flops

        population = bilevel_engine.run_generation(
            population=population,
            current_gen=gen,
            max_gens=args.generations,
            eval_fn=eval_wrapper,
            dataset_config=dataset_config,
            short_epochs=args.short_epochs
        )

        elapsed = time.time() - gen_start
        best_acc = bilevel_engine.global_best_candidate.fitness if bilevel_engine.global_best_candidate else 0.0
        print(f"  Gen [{gen:02d}/{args.generations:02d}] Best Acc: {best_acc:.4f} | Surrogate Fitted: {surrogate.is_fitted} | Elapsed: {elapsed:.2f}s")

        # Periodically clean CUDA memory cache
        if has_cuda:
            torch.cuda.empty_cache()
            gc.collect()

    total_search_time = time.time() - start_time
    logger.info(f"Co-Evolution completed in {total_search_time:.2f} seconds.")

    # Phase 3: Pareto Front Analysis & Reporting
    logger.info("\n[Phase 3] Generating Visualizations & Pareto Reports...")
    all_evaluated = [c for c in population if c.is_ground_truth]
    pareto_front = get_pareto_front(all_evaluated if all_evaluated else population)

    pareto_report = []
    print("\n--- Non-Dominated Pareto Optimal Candidates ---")
    for idx, cand in enumerate(pareto_front):
        info = {
            "rank": idx + 1,
            "candidate_id": cand.candidate_id,
            "accuracy": cand.fitness,
            "params": cand.param_count,
            "flops": cand.flops,
            "hyperparams": cand.get_decoded_hyperparams()
        }
        pareto_report.append(info)
        print(f"  Rank {idx+1}: ID={cand.candidate_id} | Acc={cand.fitness:.4f} | Params={cand.param_count:,} | FLOPs={cand.flops:,}")

    with open(output_dir / "pareto_report.json", "w") as f:
        json.dump(pareto_report, f, indent=4)

    # Generate Visualizations
    plot_convergence_curve(
        bilevel_engine.history,
        save_path=str(output_dir / "convergence.png"),
    )
    if all_evaluated:
        plot_pareto_front(
            candidates_or_params=all_evaluated,
            save_path=str(output_dir / "pareto_front.png"),
        )
    best_dag_cand = bilevel_engine.global_best_candidate or population[0]
    plot_dag_architecture(
        best_dag_cand.graph,
        save_path=str(output_dir / "best_dag.png"),
    )

    # Phase 4: Final Full-Epoch Training of Best Candidate
    winning_cand = bilevel_engine.global_best_candidate or population[0]
    logger.info(f"\n[Phase 4] Training Final Winning Architecture ({winning_cand.candidate_id}) on {target_device.upper()}...")

    final_acc, final_params, final_flops = train_and_evaluate_candidate(
        winning_cand,
        epochs=args.final_epochs,
        config=dataset_config,
        device_str=target_device
    )
    winning_cand.fitness = final_acc
    winning_cand.param_count = final_params
    winning_cand.flops = final_flops

    # Phase 5: Model Deployment Serialization & Export
    logger.info("\n[Phase 5] Exporting Winning Candidate Model...")
    winning_model = DynamicNeuralNetwork(
        dag=winning_cand.graph,
        in_channels=dataset_config["in_channels"],
        base_channels=dataset_config["base_channels"],
        num_classes=dataset_config["num_classes"]
    )

    exporter = ModelExporter(output_dir=str(output_dir))

    # 1. Export ONNX
    onnx_path = exporter.export_to_onnx(
        model=winning_model,
        input_shape=(1, dataset_config["in_channels"], 32, 32),
        filename=f"winner_{winning_cand.candidate_id}.onnx"
    )
    if onnx_path:
        exporter.verify_onnx(
            onnx_path=onnx_path,
            model=winning_model,
            input_shape=(1, dataset_config["in_channels"], 32, 32)
        )

    # 2. Export TorchScript (.pt)
    ts_path = exporter.export_to_torchscript(
        model=winning_model,
        input_shape=(1, dataset_config["in_channels"], 32, 32),
        filename=f"winner_{winning_cand.candidate_id}.pt"
    )

    print("=" * 70)
    print("[SUCCESS] AutoML Search & Export Completed Successfully!")
    print(f"Winning Candidate ID: {winning_cand.candidate_id}")
    print(f"Final Validation Accuracy: {winning_cand.fitness:.4f}")
    print(f"Parameters: {winning_cand.param_count:,} | FLOPs: {winning_cand.flops:,}")
    print(f"Decoded Hyperparameters: {winning_cand.get_decoded_hyperparams()}")
    if onnx_path:
        print(f"ONNX Model Exported: {onnx_path}")
    if ts_path:
        print(f"TorchScript Model Exported: {ts_path}")
    print(f"Artifacts saved to: {output_dir.resolve()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
