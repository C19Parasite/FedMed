"""Model factory for FedMed segmentation architectures.

Provides ``build_model(cfg)`` which constructs a 3-D segmentation network
from a validated configuration dict.  Supported architectures:

- ``monai_unet``:    MONAI ``UNet`` with configurable depth / norm.
- ``segresnet``:     MONAI ``SegResNet`` with GroupNorm by default.
- ``dynunet``:       MONAI ``DynUNet`` with optional deep supervision.
- ``unet3d_custom``: From-scratch reference 3-D U-Net (see ``unet3d.py``).

Normalization policy
--------------------
Only GroupNorm and InstanceNorm are allowed. BatchNorm is **banned** because:

1. Running statistics (mean / variance) drift under non-IID client data.
2. FedAvg averages weights, but running stats are not weights; averaging them
   produces inconsistent and drifted statistics.
3. Per-sample gradient clipping (required for differential privacy in later
   weeks) is incompatible with batch normalization statistics.

The ``assert_no_batchnorm`` guard runs at the end of every ``build_model`` call
to ensure custom or third-party defaults cannot introduce BatchNorm.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn as nn
from monai.networks.nets import DynUNet, SegResNet, UNet

from fedmed.models.norm import (
    assert_no_batchnorm,
    validate_group_norm_channels,
    validate_roi_for_model,
)
from fedmed.models.unet3d import UNet3DCustom
from fedmed.utils.logging import get_logger

logger = get_logger(__name__)


def _norm_spec(cfg: Mapping[str, Any]) -> tuple | str:
    """Return a MONAI-compatible norm spec from the config ``model.norm`` section.

    Returns:
        A string like ``"INSTANCE"`` or a tuple like ``("ADAPTIVE_GROUP", {"num_groups": 8})``.
    """
    norm_cfg = cfg["model"].get("norm", {})
    if isinstance(norm_cfg, str):
        norm_type = norm_cfg.upper()
        if "BATCH" in norm_type:
            raise ValueError(
                "BatchNorm is banned in FedMed (non-IID drift, FedAvg incompatibility, "
                "DP incompatibility). Use 'group' or 'instance'."
            )
        if norm_type == "GROUP":
            return ("ADAPTIVE_GROUP", {"num_groups": 8})
        return norm_type

    norm_type = str(norm_cfg.get("type", "instance")).lower()
    if norm_type == "batch" or "batch" in norm_type:
        # BatchNorm running statistics are a problem under non-IID client data,
        # are not meaningfully averaged by FedAvg, and are incompatible with
        # per-sample gradient clipping (differential privacy).
        raise ValueError(
            "model.norm.type='batch' is prohibited in FedMed. "
            "BatchNorm running statistics diverge under non-IID client data, "
            "are not meaningfully averaged by FedAvg, and are incompatible "
            "with per-sample gradient clipping (differential privacy). "
            "Use 'group' or 'instance' instead."
        )

    if norm_type == "group":
        num_groups = int(norm_cfg.get("num_groups", 8))
        return ("ADAPTIVE_GROUP", {"num_groups": num_groups})
    elif norm_type == "instance":
        return "INSTANCE"
    else:
        raise ValueError(
            f"Unknown norm type '{norm_type}'; expected 'group' or 'instance'."
        )


def _apply_init(model: nn.Module, init_name: str) -> None:
    """Apply weight initialisation to Conv3d / ConvTranspose3d layers.

    Choices:
        - ``kaiming``: Kaiming He normal (fan_out, relu) - standard default for
          ReLU-family activations in 3D medical segmentation.
        - ``xavier``: Xavier / Glorot uniform.
        - ``none``: Skip custom initialization.

    Biases are zeroed deterministically.
    """
    if init_name == "none":
        return

    for module in model.modules():
        if isinstance(module, (nn.Conv3d, nn.ConvTranspose3d)):
            if init_name == "kaiming":
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
            elif init_name == "xavier":
                nn.init.xavier_uniform_(module.weight)
            else:
                raise ValueError(f"Unknown init '{init_name}'; use kaiming|xavier|none")
            if module.bias is not None:
                nn.init.zeros_(module.bias)


def _build_monai_unet(cfg: Mapping[str, Any]) -> nn.Module:
    m = cfg["model"]
    norm = _norm_spec(cfg)
    channels = list(m.get("channels", [16, 32, 64, 128, 256]))
    num_groups = m.get("norm", {}).get("num_groups", 8) if isinstance(m.get("norm"), dict) else 8
    norm_type = m.get("norm", {}).get("type", "instance") if isinstance(m.get("norm"), dict) else "instance"
    if norm_type == "group":
        validate_group_norm_channels(channels, num_groups)

    return UNet(
        spatial_dims=3,
        in_channels=int(m.get("in_channels", 4)),
        out_channels=int(m.get("out_channels", 3)),
        channels=channels,
        strides=list(m.get("strides", [2, 2, 2, 2])),
        kernel_size=int(m.get("kernel_size", 3)),
        num_res_units=int(m.get("num_res_units", 0)),
        act=str(m.get("act", "PRELU")),
        norm=norm,
        dropout=float(m.get("dropout", 0.0)),
    )


def _build_segresnet(cfg: Mapping[str, Any]) -> nn.Module:
    m = cfg["model"]
    norm = _norm_spec(cfg)
    num_groups = m.get("norm", {}).get("num_groups", 8) if isinstance(m.get("norm"), dict) else 8

    # SegResNet accepts norm as string or tuple
    # If norm is ADAPTIVE_GROUP, SegResNet native GroupNorm works cleanly
    if isinstance(norm, tuple) and "GROUP" in str(norm[0]):
        seg_norm: tuple | str = ("GROUP", {"num_groups": num_groups})
    else:
        seg_norm = norm

    return SegResNet(
        spatial_dims=3,
        init_filters=int(m.get("init_filters", 16)),
        in_channels=int(m.get("in_channels", 4)),
        out_channels=int(m.get("out_channels", 3)),
        dropout_prob=float(m.get("dropout", 0.0)) or None,
        norm=seg_norm,
        num_groups=num_groups,
        blocks_down=tuple(m.get("blocks_down", (1, 2, 2, 4))),
        blocks_up=tuple(m.get("blocks_up", (1, 1, 1))),
    )


def _build_dynunet(cfg: Mapping[str, Any]) -> nn.Module:
    """Build DynUNet.

    Output contract for deep supervision:
    When ``deep_supervision=True`` and in ``.train()`` mode, forward() returns
    a stacked tensor of shape ``[B, DS+1, C, D, H, W]`` where DS is deep_supr_num.
    When in ``.eval()`` mode or deep_supervision=False, forward() returns ``[B, C, D, H, W]``.
    """
    m = cfg["model"]
    norm = _norm_spec(cfg)
    strides = list(m.get("strides", [1, 2, 2, 2]))
    if len(strides) < 3:
        strides = [1] + strides if len(strides) == 2 else strides + [2] * (3 - len(strides))

    ks_raw = m.get("kernel_size", 3)
    if isinstance(ks_raw, int):
        kernel_size = [ks_raw] * len(strides)
    else:
        kernel_size = list(ks_raw)
        if len(kernel_size) < len(strides):
            kernel_size = kernel_size + [kernel_size[-1]] * (len(strides) - len(kernel_size))
        elif len(kernel_size) > len(strides):
            kernel_size = kernel_size[:len(strides)]

    upsample_kernel_size = strides[1:]

    filters = m.get("channels")
    if filters is not None:
        filters = list(filters)
        if len(filters) < len(strides):
            for _ in range(len(strides) - len(filters)):
                filters.append(filters[-1] * 2)
        elif len(filters) > len(strides):
            filters = filters[:len(strides)]

    deep_supervision = bool(m.get("deep_supervision", False))
    num_up_layers = len(strides) - 1
    configured_ds = int(m.get("deep_supr_num", 1))
    deep_supr_num = max(1, min(configured_ds, max(1, num_up_layers - 1))) if deep_supervision else 1

    norm_name: tuple | str
    if isinstance(norm, tuple):
        norm_name = norm
    elif isinstance(norm, str) and norm.upper() == "INSTANCE":
        norm_name = ("INSTANCE", {"affine": True})
    else:
        norm_name = norm

    return DynUNet(
        spatial_dims=3,
        in_channels=int(m.get("in_channels", 4)),
        out_channels=int(m.get("out_channels", 3)),
        kernel_size=kernel_size,
        strides=strides,
        upsample_kernel_size=upsample_kernel_size,
        filters=filters,
        dropout=float(m.get("dropout", 0.0)),
        norm_name=norm_name,
        deep_supervision=deep_supervision,
        deep_supr_num=deep_supr_num,
        res_block=bool(m.get("use_residual", False)),
    )


def _build_custom_unet(cfg: Mapping[str, Any]) -> nn.Module:
    m = cfg["model"]
    norm_cfg = m.get("norm", {})
    norm_type = norm_cfg.get("type", "instance") if isinstance(norm_cfg, dict) else str(norm_cfg).lower()
    num_groups = norm_cfg.get("num_groups", 8) if isinstance(norm_cfg, dict) else 8

    if "batch" in norm_type:
        raise ValueError(
            "BatchNorm is banned in FedMed (non-IID drift, FedAvg, DP incompatibility)."
        )

    channels = m.get("channels")
    strides = m.get("strides")

    return UNet3DCustom(
        in_channels=int(m.get("in_channels", 4)),
        out_channels=int(m.get("out_channels", 3)),
        channels=list(channels) if channels else None,
        strides=list(strides) if strides else None,
        depth=int(m.get("depth", 4)),
        base_channels=int(m.get("base_channels", 16)),
        norm_type=norm_type,
        num_groups=num_groups,
        use_residual=bool(m.get("use_residual", True)),
        dropout=float(m.get("dropout", 0.0)),
    )


_BUILDERS = {
    "monai_unet": _build_monai_unet,
    "segresnet": _build_segresnet,
    "dynunet": _build_dynunet,
    "unet3d_custom": _build_custom_unet,
}


def build_model(cfg: Mapping[str, Any]) -> nn.Module:
    """Construct a segmentation model from a validated configuration mapping.

    Supported architectures in cfg['model']['name']:
    - 'monai_unet'
    - 'segresnet'
    - 'dynunet'
    - 'unet3d_custom'

    Args:
        cfg: Validated experiment configuration.

    Returns:
        Instantiated nn.Module ready for training or evaluation.

    Raises:
        ValueError: On unknown architecture, banned BatchNorm, or invalid shapes/groups.
    """
    model_cfg = cfg["model"]
    norm_cfg = model_cfg.get("norm", {})
    norm_type = norm_cfg.get("type", "") if isinstance(norm_cfg, dict) else str(norm_cfg)
    if "batch" in norm_type.lower():
        # BatchNorm running statistics are a problem under non-IID client data,
        # are not meaningfully averaged by FedAvg, and are incompatible with
        # per-sample gradient clipping (differential privacy).
        raise ValueError(
            "BatchNorm is prohibited in FedMed due to federated weight drift; "
            "use INSTANCE or GROUP normalization."
        )

    model_name = str(model_cfg["name"]).lower()
    builder = _BUILDERS.get(model_name)
    if builder is None:
        raise ValueError(
            f"Unknown model name '{model_name}'; expected one of {sorted(_BUILDERS)}"
        )

    # Validate ROI divisibility against downsampling factor
    validate_roi_for_model(cfg)

    # Instantiate model
    model = builder(cfg)

    # Apply deterministic weight initialization
    init_cfg = model_cfg.get("init", {})
    init_name = str(init_cfg.get("name", "none")).lower() if isinstance(init_cfg, dict) else "none"
    _apply_init(model, init_name)

    # Strict assertion: no BatchNorm layer allowed in the network
    assert_no_batchnorm(model)

    return model
