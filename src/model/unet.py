"""MONAI 3D U-Net factory for brain tumor MRI segmentation."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence
import torch
import torch.nn as nn
from monai.networks.nets import UNet

from fedmed.models.norm import assert_no_batchnorm
from fedmed.utils.logging import get_logger

logger = get_logger(__name__)


def build_unet(cfg: Mapping[str, Any], is_he: bool = False) -> nn.Module:
    """Build a 3D U-Net configured for multi-modal brain tumor segmentation.

    Architectural Decisions & Federated Learning Rationale:
    1. Normalization: InstanceNorm or GroupNorm is strictly enforced.
       BatchNorm is prohibited because running statistics (mean and variance)
       diverge under non-IID client distributions across hospital silos.
       Furthermore, FedAvg averages weight parameters, not empirical batch statistics,
       and per-sample differential privacy gradient clipping requires independence
       between samples that BatchNorm violates.
    2. Input Modalities: 4 MRI channels (T1, T1ce, T2, FLAIR).
    3. Output Compartments: 3 binary tumor channels (TC: Tumor Core, WT: Whole Tumor, ET: Enhancing Tumor).

    Args:
        cfg: Unified configuration dictionary.
        is_he: If True, uses the lightweight homomorphic encryption channel profile.

    Returns:
        Configured PyTorch nn.Module.
    """
    model_cfg = cfg.get("model", {})
    in_channels = int(model_cfg.get("in_channels", 4))
    out_channels = int(model_cfg.get("out_channels", 3))
    dropout = float(model_cfg.get("dropout", 0.0))
    strides = tuple(model_cfg.get("strides", [2, 2]))

    # Determine channel progression
    is_debug = cfg.get("debug", False)
    if is_he or (cfg.get("he", {}).get("enabled", False) and cfg.get("he", {}).get("profile") == "small"):
        channels = tuple(model_cfg.get("he_channels", [4, 8, 16]))
    elif is_debug:
        channels = tuple(model_cfg.get("debug_channels", [4, 8, 16]))
    else:
        channels = tuple(model_cfg.get("channels", [16, 32, 64]))

    norm_type = str(model_cfg.get("norm", "INSTANCE")).upper()
    if "BATCH" in norm_type:
        raise ValueError("BatchNorm is strictly prohibited in FedMed due to non-IID statistical drift.")

    if norm_type in ("INSTANCE", "IN"):
        norm = ("INSTANCE", {"affine": True})
    elif norm_type in ("GROUP", "GN"):
        num_groups = int(model_cfg.get("num_groups", min(4, channels[0])))
        norm = ("GROUP", {"num_groups": num_groups, "affine": True})
    else:
        norm = ("INSTANCE", {"affine": True})

    model = UNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=channels,
        strides=strides,
        num_res_units=0,
        norm=norm,
        dropout=dropout,
    )

    # Runtime assertion to ensure no BatchNorm leaked into any submodule
    assert_no_batchnorm(model)
    logger.info(
        "Built UNet3D (channels=%s, norm=%s, in=%s, out=%s, params=%s)",
        channels,
        norm[0],
        in_channels,
        out_channels,
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )
    return model


__all__ = ["build_unet"]
