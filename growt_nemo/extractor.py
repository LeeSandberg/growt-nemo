"""Hook-based feature extraction from PyTorch models for Growt auditing."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

logger = logging.getLogger("growt_nemo")


def extract_features(
    model: nn.Module,
    dataloader: DataLoader,  # type: ignore[type-arg]
    layer_name: Optional[str] = None,
    max_samples: int = 5000,
    device: Optional[torch.device] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract feature vectors from a model layer using forward hooks.

    Registers a temporary forward hook on the target layer, runs inference
    over the dataloader, and collects the intermediate activations.

    Args:
        model: Trained PyTorch / NeMo model.
        dataloader: DataLoader yielding ``(inputs, labels)`` tuples.
        layer_name: Dot-separated layer path (e.g. ``"encoder.layer.11"``).
            If ``None``, the penultimate layer is auto-detected.
        max_samples: Maximum number of samples to extract.
        device: Device to run inference on.  Defaults to the model's current device.

    Returns:
        Tuple of ``(features, labels)`` as numpy arrays.
        ``features`` has shape ``(N, D)`` and ``labels`` has shape ``(N,)``.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    target_layer = _resolve_layer(model, layer_name)
    logger.info(
        "Extracting features from layer: %s (max_samples=%d)",
        target_layer.__class__.__name__,
        max_samples,
    )

    features_list: list[torch.Tensor] = []
    labels_list: list[torch.Tensor] = []
    collected = 0

    hook_output: list[torch.Tensor] = []

    def hook_fn(
        _module: nn.Module,
        _input: tuple[torch.Tensor, ...],
        output: torch.Tensor | tuple[torch.Tensor, ...],
    ) -> None:
        hook_output.clear()
        if isinstance(output, torch.Tensor):
            hook_output.append(output.detach())
        elif isinstance(output, tuple) and len(output) > 0:
            hook_output.append(output[0].detach())

    handle = target_layer.register_forward_hook(hook_fn)

    try:
        with torch.no_grad():
            for batch in dataloader:
                if collected >= max_samples:
                    break

                inputs = batch[0].to(device)
                batch_labels = batch[1]
                model(inputs)

                if hook_output:
                    feat = hook_output[0]
                    # Flatten spatial dims if present (e.g. CNN feature maps)
                    if feat.dim() > 2:
                        feat = feat.mean(dim=list(range(2, feat.dim())))
                    features_list.append(feat.cpu())
                    labels_list.append(batch_labels)
                    collected += feat.shape[0]
    finally:
        handle.remove()

    all_features = torch.cat(features_list, dim=0)[:max_samples]
    all_labels = torch.cat(labels_list, dim=0)[:max_samples]

    logger.info(
        "Extracted features: shape=%s, labels: shape=%s",
        all_features.shape,
        all_labels.shape,
    )
    return all_features.numpy(), all_labels.numpy()


def _resolve_layer(model: nn.Module, layer_name: Optional[str]) -> nn.Module:
    """Find the target layer for feature extraction.

    Args:
        model: The PyTorch model.
        layer_name: Dot-separated layer path, or ``None`` to auto-detect
            the penultimate layer.

    Returns:
        The resolved ``nn.Module``.

    Raises:
        ValueError: If penultimate layer cannot be auto-detected.
        AttributeError: If the specified layer path does not exist.
    """
    if layer_name:
        parts = layer_name.split(".")
        module: nn.Module = model
        for part in parts:
            if part.isdigit():
                module = module[int(part)]  # type: ignore[index]
            else:
                module = getattr(module, part)
        return module

    # Auto-detect penultimate layer: second-to-last direct child
    children = list(model.children())
    if len(children) >= 2:
        return children[-2]

    # For models with named modules, find the last non-classifier layer
    modules = list(model.named_modules())
    for _name, module in reversed(modules):
        if not isinstance(module, (nn.Linear, nn.Softmax, nn.LogSoftmax)):
            if list(module.parameters()):
                return module

    raise ValueError(
        "Could not auto-detect penultimate layer. "
        "Please specify layer_name explicitly."
    )
