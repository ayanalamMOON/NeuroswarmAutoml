"""
Model Evaluation Metrics Module with CUDA Acceleration.

Provides parameter counting, FLOP estimation, and GPU-accelerated accuracy computation utilities
for dynamic neural network architectures.
"""

from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import numpy as np


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """
    Counts the total number of parameters in a PyTorch model.

    Args:
        model: PyTorch nn.Module instance.
        trainable_only: If True, count only parameters requiring gradients.

    Returns:
        int: Total parameter count.
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def estimate_flops(
    model: nn.Module,
    input_shape: Tuple[int, ...] = (1, 3, 32, 32),
    device: Optional[Union[str, torch.device]] = None,
) -> int:
    """
    Estimates FLOPs for a forward pass using a hook-based approach.

    Args:
        model: PyTorch nn.Module instance.
        input_shape: Input tensor shape (batch, channels, height, width).
        device: Target device for the dummy forward pass.

    Returns:
        int: Estimated total FLOPs.
    """
    total_flops = 0
    hooks = []

    def _hook_fn(module, input, output):
        nonlocal total_flops
        if isinstance(module, nn.Conv2d):
            batch_size = input[0].shape[0]
            out_h, out_w = output.shape[2], output.shape[3]
            kernel_ops = (
                module.kernel_size[0]
                * module.kernel_size[1]
                * (module.in_channels // module.groups)
            )
            total_flops += (
                2 * kernel_ops * module.out_channels * out_h * out_w * batch_size
            )
        elif isinstance(module, nn.Linear):
            batch_size = input[0].shape[0] if input[0].dim() > 1 else 1
            total_flops += 2 * module.in_features * module.out_features * batch_size

    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            hooks.append(module.register_forward_hook(_hook_fn))

    model.eval()
    if device is None:
        device = (
            next(model.parameters()).device
            if list(model.parameters())
            else torch.device("cpu")
        )
    else:
        device = torch.device(device)

    dummy_input = torch.randn(*input_shape, device=device)
    with torch.no_grad():
        model(dummy_input)

    for hook in hooks:
        hook.remove()

    return max(total_flops, 1000)


def compute_accuracy(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: Optional[Union[str, torch.device]] = None,
    use_amp: bool = True,
) -> float:
    """
    Computes top-1 classification accuracy on a given dataloader with CUDA and AMP support.

    Args:
        model: Trained PyTorch nn.Module.
        dataloader: PyTorch DataLoader yielding (inputs, targets) batches.
        device: Device to run inference on (defaults to model device or CUDA if available).
        use_amp: Whether to use Automatic Mixed Precision during evaluation.

    Returns:
        float: Accuracy as a fraction in [0.0, 1.0].
    """
    model.eval()
    if device is None:
        device = (
            next(model.parameters()).device
            if list(model.parameters())
            else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        )
    else:
        device = torch.device(device)

    is_cuda = device.type == "cuda"
    enable_amp = use_amp and is_cuda

    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device, non_blocking=is_cuda)
            targets = targets.to(device, non_blocking=is_cuda)

            with torch.amp.autocast("cuda", enabled=enable_amp):
                outputs = model(inputs)

            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

    return float(correct / max(total, 1))
