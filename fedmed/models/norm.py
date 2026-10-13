"""Federated normalization policies, guards, and shape validators.

In federated learning, Batch Normalization is banned because:
1. Non-IID client distributions cause running mean and variance to diverge.
2. FedAvg averages weights, but running statistics are not weights, making naive
   aggregation mathematically ungrounded.
3. Per-sample gradient clipping (required for Differential Privacy in subsequent weeks)
   is incompatible with cross-sample BatchNorm batch statistics.

This module provides:
- ``assert_no_batchnorm``: Recursively inspects PyTorch models and raises if any BatchNorm
  layer is present.
- ``validate_roi_for_model``: Validates that data ROI dimensions are divisible by the model's
  total downsampling factor (product of strides).
- ``validate_group_norm_channels``: Ensures GroupNorm num_groups divides every channel count.
- ``get_norm_layer_spec``: Translates configuration norm definitions into MONAI/PyTorch specifications.
"""

from __future__ import annotations

from functools import reduce
from typing import Any, Mapping, Sequence
import torch.nn as nn

_BANNED_BATCHNORM_CLASSES = (
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.SyncBatchNorm,
    nn.LazyBatchNorm1d,
    nn.LazyBatchNorm2d,
    nn.LazyBatchNorm3d,
)


def assert_no_batchnorm(model: nn.Module) -> None:
    """Scan model modules and raise ValueError if any BatchNorm layer is found.

    Args:
        model: The PyTorch neural network module to inspect.

    Raises:
        ValueError: If any BatchNorm layer is present, detailing layer paths and types.
    """
    offending = []
    for name, module in model.named_modules():
        if isinstance(module, _BANNED_BATCHNORM_CLASSES):
            offending.append((name, type(module).__name__))

    if offending:
        details = ", ".join(f"'{name}' ({cls})" for name, cls in offending)
        raise ValueError(
            f"BatchNorm is strictly prohibited in FedMed due to non-IID drift, FedAvg "
            f"incompatibility, and Differential Privacy clipping failure. "
            f"Found offending layer(s): {details}. Use GroupNorm or InstanceNorm instead."
        )


def validate_roi_for_model(cfg: Mapping[str, Any]) -> None:
    """Verify that every data.roi_size dimension is divisible by the product of model strides.

    Args:
        cfg: Configuration dictionary containing optional 'data.roi_size' and 'model.strides'.

    Raises:
        ValueError: If any ROI dimension is not divisible by the stride product.
    """
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})

    roi_size = data_cfg.get("roi_size")
    strides = model_cfg.get("strides")

    if roi_size is None or strides is None:
        return

    # Total downsampling factor is the product of all downsampling strides
    total_downsample = reduce(lambda a, b: a * b, strides, 1)

    bad_dims = []
    for idx, dim in enumerate(roi_size):
        if dim % total_downsample != 0:
            bad_dims.append((idx, dim))

    if bad_dims:
        bad_strs = [f"dim[{i}]={d}" for i, d in bad_dims]
        raise ValueError(
            f"ROI dimensions must each be divisible by the total downsampling factor "
            f"(product of model.strides={list(strides)} = {total_downsample}). "
            f"Failing dimension(s): {', '.join(bad_strs)}. Required multiple: {total_downsample}."
        )


def validate_group_norm_channels(channels: Sequence[int], num_groups: int) -> None:
    """Verify that num_groups evenly divides every channel count in the model.

    Args:
        channels: Sequence of channel counts per level.
        num_groups: Number of groups requested for GroupNorm.

    Raises:
        ValueError: If any channel count is not divisible by num_groups.
    """
    if num_groups < 1:
        raise ValueError(f"num_groups must be a positive integer, got: {num_groups}")

    bad_channels = [c for c in channels if c % num_groups != 0]
    if bad_channels:
        raise ValueError(
            f"GroupNorm num_groups={num_groups} does not divide all channel widths. "
            f"Offending channel widths: {bad_channels}. "
            f"All channel widths must be divisible by {num_groups}."
        )


def get_norm_layer_spec(norm_cfg: Any) -> tuple | str:
    """Convert configuration norm entry into MONAI norm specification.

    Args:
        norm_cfg: String ('instance', 'group') or Mapping {'type': ..., 'num_groups': ...}.

    Returns:
        Specification usable by MONAI components.

    Raises:
        ValueError: If 'batch' norm is requested or unrecognized norm type.
    """
    if isinstance(norm_cfg, Mapping):
        norm_type = str(norm_cfg.get("type", "instance")).lower()
        if "batch" in norm_type:
            # BatchNorm running statistics are a problem under non-IID client data,
            # are not meaningfully averaged by FedAvg, and are incompatible with
            # per-sample gradient clipping (differential privacy).
            raise ValueError(
                "model.norm.type='batch' is prohibited. BatchNorm running statistics "
                "diverge under non-IID client data, are not meaningfully averaged by FedAvg, "
                "and are incompatible with per-sample gradient clipping (differential privacy). "
                "Use 'group' or 'instance' instead."
            )
        if norm_type == "group":
            num_groups = int(norm_cfg.get("num_groups", 8))
            return ("GROUP", {"num_groups": num_groups})
        elif norm_type == "instance":
            return "INSTANCE"
        else:
            raise ValueError(f"Unknown norm type '{norm_type}'; expected 'group' or 'instance'.")

    norm_str = str(norm_cfg).lower()
    if "batch" in norm_str:
        # BatchNorm running statistics are a problem under non-IID client data,
        # are not meaningfully averaged by FedAvg, and are incompatible with
        # per-sample gradient clipping (differential privacy).
        raise ValueError(
            "BatchNorm is prohibited in FedMed. BatchNorm running statistics "
            "diverge under non-IID client data, are not meaningfully averaged by FedAvg, "
            "and are incompatible with per-sample gradient clipping (differential privacy). "
            "Use 'group' or 'instance' instead."
        )
    if norm_str == "group":
        return ("GROUP", {"num_groups": 8})
    elif norm_str == "instance":
        return "INSTANCE"
    else:
        return norm_str.upper()
