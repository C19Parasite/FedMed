"""3D U-Net architecture for multi-modal brain tumor MRI segmentation."""

from __future__ import annotations

from typing import Any, Mapping, Sequence
import torch
import torch.nn as nn
from monai.networks.nets import UNet
from monai.networks.layers.factories import Norm

from fedmed.utils.logging import get_logger

logger = get_logger(__name__)


class AdaptiveGroupNorm(nn.Module):
    """GroupNorm that falls back to the largest valid group divisor if num_channels is not divisible.

    In MONAI's UNet with num_res_units > 0, an upsampling normalization layer is
    instantiated on the final output channels (e.g. 3 classes). PyTorch requires
    num_channels to be divisible by num_groups. AdaptiveGroupNorm selects the
    largest divisor of num_channels <= num_groups (e.g. 1 for 3 channels),
    allowing standard GroupNorm on all internal layers while preventing runtime crashes.
    """

    def __init__(self, num_groups: int, num_channels: int, **kwargs: Any) -> None:
        super().__init__()
        g = num_groups
        while g > 1 and num_channels % g != 0:
            g -= 1
        self.gn = nn.GroupNorm(num_groups=g, num_channels=num_channels, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gn(x)


# Register with MONAI Norm factory if not already present
try:
    Norm.add_factory_class("ADAPTIVE_GROUP", AdaptiveGroupNorm)
except Exception:
    pass


class UNet3D(nn.Module):
    """3D U-Net segmentation network using InstanceNorm or GroupNorm for federated stability.

    Adheres strictly to the FedMed requirement of using InstanceNorm or GroupNorm
    instead of BatchNorm to prevent statistical drift during federated averaging.

    Args:
        in_channels: Number of input MRI channels (typically 4: T1, T1ce, T2, FLAIR).
        out_channels: Number of output tumor classes (typically 3: TC, WT, ET).
        channels: Sequence of feature channels per resolution level.
        strides: Sequence of convolution strides per downsampling step.
        norm: Normalization layer type ('INSTANCE', 'GROUP', tuple, or mapping).
        dropout: Dropout probability.
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 3,
        channels: Sequence[int] = (16, 32, 64, 128),
        strides: Sequence[int] = (2, 2, 2),
        norm: str | tuple | Mapping[str, Any] = "INSTANCE",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        norm_spec: str | tuple
        if isinstance(norm, Mapping):
            norm_type = str(norm.get("type", "instance")).upper()
            if "BATCH" in norm_type:
                raise ValueError(
                    "BatchNorm is prohibited in FedMed due to federated weight drift; "
                    "use INSTANCE or GROUP normalization."
                )
            if norm_type == "GROUP":
                num_groups = int(norm.get("num_groups", 8))
                norm_spec = ("ADAPTIVE_GROUP", {"num_groups": num_groups})
            else:
                norm_spec = norm_type
        elif isinstance(norm, tuple):
            if norm[0].upper() == "GROUP":
                norm_spec = ("ADAPTIVE_GROUP", norm[1])
            else:
                norm_spec = norm
        else:
            norm_upper = str(norm).upper()
            if "BATCH" in norm_upper:
                raise ValueError(
                    "BatchNorm is prohibited in FedMed due to federated weight drift; "
                    "use INSTANCE or GROUP normalization."
                )
            if norm_upper == "GROUP":
                norm_spec = ("ADAPTIVE_GROUP", {"num_groups": 8})
            else:
                norm_spec = norm_upper

        self.net = UNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=tuple(channels),
            strides=tuple(strides),
            num_res_units=2,
            norm=norm_spec,
            dropout=dropout,
        )
        logger.info(
            "Initialized UNet3D with in_channels=%s, out_channels=%s, norm=%s",
            in_channels,
            out_channels,
            norm_spec,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through 3D U-Net.

        Args:
            x: Input tensor of shape (B, in_channels, D, H, W).

        Returns:
            Logits tensor of shape (B, out_channels, D, H, W).
        """
        return self.net(x)


def get_model(cfg: Mapping[str, Any]) -> UNet3D:
    """Build and return UNet3D from validated configuration mapping.

    Args:
        cfg: Validated config mapping containing 'model' section.

    Returns:
        Instantiated UNet3D module.
    """
    model_cfg = cfg["model"]
    norm_cfg = model_cfg["norm"]
    if isinstance(norm_cfg, Mapping):
        norm_type = str(norm_cfg.get("type", "instance")).upper()
        if norm_type == "GROUP":
            norm: str | tuple = ("ADAPTIVE_GROUP", {"num_groups": int(norm_cfg.get("num_groups", 8))})
        else:
            norm = norm_type
    else:
        norm = str(norm_cfg)

    return UNet3D(
        in_channels=int(model_cfg["in_channels"]),
        out_channels=int(model_cfg["out_channels"]),
        channels=model_cfg["channels"],
        strides=model_cfg["strides"],
        norm=norm,
        dropout=float(model_cfg.get("dropout", 0.1)),
    )
