"""Parameter serialization, deserialization, vectorization, and inspection utilities.

Provides dependency-free (torch and numpy only) model parameter extraction
for federated aggregation (Flower FedAvg), parameter flattening for
secure aggregation / differential privacy, and model inspection.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence
import numpy as np
import torch
import torch.nn as nn


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Return total number of parameters in the model.

    Args:
        model: PyTorch model.
        trainable_only: If True, count only parameters with requires_grad=True.

    Returns:
        Integer count of parameters.
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def get_parameters(model: nn.Module) -> list[np.ndarray]:
    """Extract model parameters and buffers in deterministic state_dict order.

    Returns:
        List of NumPy ndarrays, float32, detached CPU copies.
    """
    return [
        val.detach().cpu().numpy().astype(np.float32, copy=True)
        for val in model.state_dict().values()
    ]


def set_parameters(model: nn.Module, params: list[np.ndarray]) -> None:
    """Load model parameters and buffers from a list of NumPy arrays.

    Uses model.state_dict() order and keys. Performs strict length and shape validation.
    Leaves no gradient side effects.

    Args:
        model: Target PyTorch model to update.
        params: Ordered list of NumPy ndarrays matching model.state_dict().

    Raises:
        ValueError: If parameter list length or individual tensor shapes do not match.
    """
    curr_state = model.state_dict()
    state_keys = list(curr_state.keys())
    if len(params) != len(state_keys):
        raise ValueError(
            f"Parameter count mismatch: provided {len(params)} parameter arrays, "
            f"but model state_dict expects {len(state_keys)} entries."
        )

    new_state: dict[str, torch.Tensor] = {}
    for key, arr in zip(state_keys, params):
        expected_tensor = curr_state[key]
        expected_shape = tuple(expected_tensor.shape)
        actual_shape = tuple(arr.shape)

        if actual_shape != expected_shape:
            raise ValueError(
                f"Shape mismatch for parameter '{key}': expected {expected_shape}, "
                f"got array shape {actual_shape}."
            )

        new_state[key] = torch.as_tensor(
            arr,
            device=expected_tensor.device,
            dtype=expected_tensor.dtype,
        )

    with torch.no_grad():
        model.load_state_dict(new_state, strict=True)


def parameters_to_vector(
    parameters: Sequence[torch.Tensor | np.ndarray],
) -> torch.Tensor:
    """Flatten a collection of tensors or numpy arrays into a single 1D float32 tensor.

    Args:
        parameters: Sequence of tensors or numpy arrays.

    Returns:
        1D float32 CPU PyTorch tensor.
    """
    tensors = []
    for p in parameters:
        if isinstance(p, np.ndarray):
            tensors.append(torch.from_numpy(p).view(-1).float())
        else:
            tensors.append(p.detach().cpu().view(-1).float())

    if not tensors:
        return torch.empty(0, dtype=torch.float32)

    return torch.cat(tensors, dim=0)


def vector_to_parameters(vec: torch.Tensor, model: nn.Module) -> None:
    """Load flattened 1D vector back into model.parameters() in-place.

    Args:
        vec: 1D PyTorch tensor containing concatenated parameter elements.
        model: Model whose parameters will be updated.

    Raises:
        ValueError: If vector size does not match total parameter count.
    """
    total_params = sum(p.numel() for p in model.parameters())
    if vec.numel() != total_params:
        raise ValueError(
            f"Vector size mismatch: vector has {vec.numel()} elements, "
            f"but model.parameters() has {total_params} elements."
        )

    pointer = 0
    with torch.no_grad():
        for p in model.parameters():
            numel = p.numel()
            slice_data = vec[pointer : pointer + numel].view_as(p)
            p.copy_(slice_data.to(device=p.device, dtype=p.dtype))
            pointer += numel


def model_size_mb(model: nn.Module) -> float:
    """Calculate total in-memory size of model parameters and buffers in megabytes (MB).

    Args:
        model: PyTorch model.

    Returns:
        Float size in MB.
    """
    total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    total_bytes += sum(b.numel() * b.element_size() for b in model.buffers())
    return float(total_bytes / (1024.0 * 1024.0))


def _detect_norm_types(model: nn.Module) -> str:
    """Identify normalization layer types used in the model."""
    norms = set()
    for m in model.modules():
        name = type(m).__name__
        if "GroupNorm" in name:
            norms.add("GroupNorm")
        elif "InstanceNorm" in name:
            norms.add("InstanceNorm")
        elif "BatchNorm" in name:
            norms.add("BatchNorm")
    if not norms:
        return "None"
    return "/".join(sorted(norms))


def log_model_summary(model: nn.Module, logger: logging.Logger | Any) -> dict[str, Any]:
    """Log model architecture summary without recording filesystem paths.

    Args:
        model: PyTorch model.
        logger: Standard Python logger or compatible logging interface.

    Returns:
        Dictionary of summary statistics.
    """
    name = type(model).__name__
    n_params = count_parameters(model, trainable_only=False)
    n_trainable = count_parameters(model, trainable_only=True)
    size_mb = model_size_mb(model)
    norm_type = _detect_norm_types(model)

    summary = {
        "model_name": name,
        "total_parameters": n_params,
        "trainable_parameters": n_trainable,
        "size_mb": round(size_mb, 4),
        "norm_type": norm_type,
    }

    logger.info(
        "Model Summary: arch=%s, params=%d (trainable=%d), size=%.2f MB, norm=%s",
        name,
        n_params,
        n_trainable,
        size_mb,
        norm_type,
    )

    return summary
