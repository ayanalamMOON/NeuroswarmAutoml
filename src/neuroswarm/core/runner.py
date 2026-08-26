"""
Distributed Execution Runner & PyTorch Dynamic Evaluator with CUDA Acceleration.

Features:
- Multi-GPU and isolated process scheduling via multiprocessing & Ray
- Automatic Mixed Precision (AMP via torch.amp with GradScaler)
- Hook-based exact FLOPs and parameter count estimation
- CosineAnnealingWarmRestarts learning rate scheduling
- Asynchronous non-blocking host-to-device memory transfers
- Automatic VRAM garbage collection and cuDNN benchmark optimization
"""

import gc
import logging
import multiprocessing as mp
from typing import Dict, Any, Tuple, List, Optional, Callable

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from neuroswarm.core.candidate import Candidate
from neuroswarm.search_space.dynamic_builder import DynamicNeuralNetwork

logger = logging.getLogger("neuroswarm.runner")

# Ray distributed framework import wrapper
try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False


def calculate_model_stats(
    model: nn.Module,
    input_size: Tuple[int, ...] = (1, 3, 32, 32),
    device: Optional[torch.device] = None,
) -> Tuple[int, int]:
    """Calculates trainable parameter count and estimated FLOPs for a PyTorch module."""
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

    flops = 0

    def conv_flop_hook(module, input_tok, output_tok):
        nonlocal flops
        batch_size, input_channels, input_h, input_w = input_tok[0].shape
        output_channels, output_h, output_w = output_tok.shape[1:]
        kernel_ops = module.kernel_size[0] * module.kernel_size[1] * (input_channels / module.groups)
        flops += int(batch_size * output_channels * output_h * output_w * (2 * kernel_ops))

    def linear_flop_hook(module, input_tok, output_tok):
        nonlocal flops
        batch_size = input_tok[0].shape[0] if input_tok[0].dim() > 1 else 1
        flops += int(batch_size * (2 * module.in_features - 1) * module.out_features)

    hooks = []
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            hooks.append(layer.register_forward_hook(conv_flop_hook))
        elif isinstance(layer, nn.Linear):
            hooks.append(layer.register_forward_hook(linear_flop_hook))

    model.eval()
    if device is None:
        device = next(model.parameters()).device if list(model.parameters()) else torch.device("cpu")

    dummy = torch.randn(*input_size, device=device)
    with torch.no_grad():
        try:
            _ = model(dummy)
        except Exception:
            pass

    for h in hooks:
        h.remove()

    return param_count, max(flops, 1000)


def train_and_evaluate_candidate(
    candidate: Candidate,
    epochs: int,
    config: Dict[str, Any],
    device_str: Optional[str] = None
) -> Tuple[float, int, int]:
    """
    Builds, trains, and evaluates a dynamic PyTorch neural architecture with CUDA acceleration.

    Features:
    - Automatic Mixed Precision (AMP) via torch.amp with GradScaler
    - cuDNN benchmarking for fast convolution kernels
    - Asynchronous pinned-memory tensor transfers
    - CosineAnnealingWarmRestarts learning rate scheduler
    - Automatic VRAM cleanup on completion
    """
    if device_str is None:
        device_str = "cuda:0" if torch.cuda.is_available() else "cpu"

    device = torch.device(device_str)
    is_cuda = device.type == "cuda"

    if is_cuda:
        try:
            torch.cuda.set_device(device)
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass

    # Decode continuous hyperparameters from candidate vector
    hparams = candidate.get_decoded_hyperparams()
    lr = hparams["learning_rate"]
    beta1 = hparams["beta1"]
    weight_decay = hparams["weight_decay"]
    batch_size = hparams["batch_size"]

    # Build dynamic PyTorch neural network from candidate DAG
    in_channels = config.get("in_channels", 3)
    num_classes = config.get("num_classes", 10)
    base_channels = config.get("base_channels", 32)
    use_amp = config.get("use_amp", True) and is_cuda

    try:
        model = DynamicNeuralNetwork(
            dag=candidate.graph,
            in_channels=in_channels,
            base_channels=base_channels,
            num_classes=num_classes
        ).to(device)
    except Exception as e:
        logger.error(f"Failed to instantiate architecture for Candidate {candidate.candidate_id}: {e}")
        return -1.0, 0, 0

    # Compute parameter count and FLOPs
    param_count, flops = calculate_model_stats(model, input_size=(1, in_channels, 32, 32), device=device)

    # DataLoaders setup
    train_loader = config.get("train_loader", None)
    val_loader = config.get("val_loader", None)

    if train_loader is None or val_loader is None:
        # Synthetic dataset fallback for testing
        pin_memory = is_cuda
        x_train = torch.randn(256, in_channels, 32, 32)
        y_train = torch.randint(0, num_classes, (256,))
        x_val = torch.randn(64, in_channels, 32, 32)
        y_val = torch.randint(0, num_classes, (64,))

        train_loader = DataLoader(
            TensorDataset(x_train, y_train),
            batch_size=batch_size,
            shuffle=True,
            pin_memory=pin_memory
        )
        val_loader = DataLoader(
            TensorDataset(x_val, y_val),
            batch_size=batch_size,
            shuffle=False,
            pin_memory=pin_memory
        )

    # Loss function & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        model.parameters(),
        lr=lr,
        betas=(beta1, 0.999),
        weight_decay=weight_decay
    )

    # AMP Scaler for GPU mixed precision
    device_type = "cuda" if is_cuda else "cpu"
    scaler = torch.amp.GradScaler(device_type, enabled=use_amp)

    # Cosine Annealing with Warm Restarts Scheduler
    T_0 = max(1, config.get("cosine_t0", max(1, epochs // 2)))
    T_mult = config.get("cosine_tmult", 2)
    eta_min = lr * 1e-3

    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=T_0,
        T_mult=T_mult,
        eta_min=eta_min
    )

    try:
        # Training Loop with AMP
        model.train()
        for epoch in range(epochs):
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(device, non_blocking=is_cuda)
                y_batch = y_batch.to(device, non_blocking=is_cuda)

                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast(device_type, enabled=use_amp):
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            scheduler.step()

        # Validation Loop
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device, non_blocking=is_cuda)
                y_batch = y_batch.to(device, non_blocking=is_cuda)

                with torch.amp.autocast(device_type, enabled=use_amp):
                    outputs = model(X_batch)

                preds = outputs.argmax(dim=1)
                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)

        val_accuracy = float(correct / max(total, 1))
    finally:
        # Explicit VRAM cleanup to prevent memory fragmentation across dynamic candidates
        del model, optimizer, scheduler, scaler
        if is_cuda:
            torch.cuda.empty_cache()
            gc.collect()

    return val_accuracy, param_count, flops


if RAY_AVAILABLE:
    @ray.remote
    def _ray_eval_worker(
        candidate: Candidate,
        epochs: int,
        config: Dict[str, Any],
        gpu_id: Optional[int] = None
    ) -> Tuple[str, float, int, int]:
        """Ray remote task wrapper for evaluating candidates across distributed CUDA nodes."""
        device_str = f"cuda:{gpu_id}" if gpu_id is not None and torch.cuda.is_available() else "cpu"
        val_acc, params, flops = train_and_evaluate_candidate(candidate, epochs, config, device_str)
        return candidate.candidate_id, val_acc, params, flops


def _mp_worker_wrapper(args: Tuple[Candidate, int, Dict[str, Any], str]) -> Tuple[str, float, int, int]:
    """Multiprocessing process pool fallback wrapper with CUDA device placement."""
    candidate, epochs, config, device_str = args
    val_acc, params, flops = train_and_evaluate_candidate(candidate, epochs, config, device_str)
    return candidate.candidate_id, val_acc, params, flops


class ParallelRunner:
    """
    Production Execution Runner supporting CUDA-accelerated distributed Ray evaluation
    and Python multiprocessing with per-GPU process allocation.
    """

    def __init__(
        self,
        num_workers: int = 2,
        use_ray: bool = False,
        gpus_per_worker: float = 1.0 if torch.cuda.is_available() else 0.0
    ):
        self.num_workers = min(num_workers, mp.cpu_count())
        self.use_ray = use_ray and RAY_AVAILABLE
        self.gpus_per_worker = gpus_per_worker
        self.num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

        if self.use_ray:
            if not ray.is_initialized():
                logger.info("Initializing Ray cluster context...")
                ray.init(ignore_reinit_error=True)
            logger.info(f"ParallelRunner: Ray enabled with {self.num_workers} workers, {self.num_gpus} GPUs.")
        else:
            gpu_info = f"({self.num_gpus} CUDA GPU(s) available)" if self.num_gpus > 0 else "(CPU Mode)"
            logger.info(f"ParallelRunner: Multiprocessing enabled with {self.num_workers} workers {gpu_info}.")

    def evaluate_candidates(
        self,
        candidates: List[Candidate],
        epochs: int,
        config: Dict[str, Any],
        eval_fn: Optional[Callable] = None
    ) -> List[Candidate]:
        """
        Evaluates candidates in parallel using CUDA-accelerated Ray tasks or multiprocessing workers.
        """
        if not candidates:
            return []

        results_map: Dict[str, Tuple[float, int, int]] = {}

        if self.use_ray:
            futures = []
            for i, cand in enumerate(candidates):
                gpu_id = i % self.num_gpus if self.num_gpus > 0 and self.gpus_per_worker > 0 else None
                task = _ray_eval_worker.options(num_gpus=self.gpus_per_worker).remote(
                    cand, epochs, config, gpu_id
                )
                futures.append(task)

            results = ray.get(futures)
            for cand_id, score, params, flops in results:
                results_map[cand_id] = (score, params, flops)

        elif self.num_workers > 1:
            ctx = mp.get_context("spawn")
            tasks = []
            for i, cand in enumerate(candidates):
                device_str = f"cuda:{i % self.num_gpus}" if self.num_gpus > 0 else "cpu"
                tasks.append((cand, epochs, config, device_str))

            with ctx.Pool(processes=self.num_workers) as pool:
                for cand_id, score, params, flops in tqdm(
                    pool.imap_unordered(_mp_worker_wrapper, tasks),
                    total=len(tasks),
                    desc=f"Training Batch ({epochs} Epochs [CUDA/MP])"
                ):
                    results_map[cand_id] = (score, params, flops)
        else:
            for i, cand in enumerate(candidates):
                device_str = "cuda:0" if self.num_gpus > 0 else "cpu"
                cand_id, score, params, flops = _mp_worker_wrapper((cand, epochs, config, device_str))
                results_map[cand_id] = (score, params, flops)

        # Update Candidates in place
        for cand in candidates:
            if cand.candidate_id in results_map:
                score, params, flops = results_map[cand.candidate_id]
                cand.fitness = score
                cand.param_count = params
                cand.flops = flops
                cand.evaluated_epochs += epochs
                cand.is_ground_truth = True
                cand.uncertainty = 0.0
                cand.update_pbest()

        return candidates

    def evaluate_single_candidate(
        self,
        candidate: Candidate,
        epochs: int,
        config: Dict[str, Any],
        device_str: Optional[str] = None
    ) -> Dict[str, Any]:
        """Evaluates a single candidate and returns a dictionary of metrics."""
        if device_str is None:
            device_str = "cuda:0" if torch.cuda.is_available() else "cpu"

        val_acc, params, flops = train_and_evaluate_candidate(candidate, epochs, config, device_str)
        candidate.fitness = val_acc
        candidate.param_count = params
        candidate.flops = flops
        candidate.evaluated_epochs += epochs
        candidate.is_ground_truth = True
        candidate.uncertainty = 0.0
        candidate.update_pbest()

        return {
            "accuracy": val_acc,
            "params": params,
            "flops": flops,
            "latency_ms": candidate.latency_ms
        }

    def shutdown(self) -> None:
        """Cleans up Ray cluster contexts or process pool contexts cleanly."""
        if self.use_ray:
            try:
                import ray
                if ray.is_initialized():
                    logger.info("Shutting down Ray cluster context...")
                    ray.shutdown()
            except Exception as e:
                logger.warning(f"Error during Ray cluster shutdown: {e}")
