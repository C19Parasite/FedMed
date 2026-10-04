"""3D U-Net architecture for multi-modal brain tumor MRI segmentation."""

from __future__ import annotations

from typing import Any, Mapping, Sequence
import torch
import torch.nn as nn
from monai.networks.nets import UNet

from fedmed.utils.logging import get_logger

logger = get_logger(__name__)


class UNet3D(nn.Module):
    """3D U-Net segmentation network using InstanceNorm for federated stability.

    Adheres strictly to the FedMed requirement of using InstanceNorm or GroupNorm
    instead of BatchNorm to prevent statistical drift during federated averaging.

    Args:
        in_channels: Number of input MRI channels (typically 4: T1, T1ce, T2, FLAIR).
        out_channels: Number of output tumor classes (typically 3: TC, WT, ET).
        channels: Sequence of feature channels per resolution level.
        strides: Sequence of convolution strides per downsampling step.
        norm: Normalization layer type ('INSTANCE' or 'GROUP').
        dropout: Dropout probability.
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 3,
        channels: Sequence[int] = (16, 32, 64, 128),
        strides: Sequence[int] = (2, 2, 2),
        norm: str = "INSTANCE",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        norm_upper = norm.upper()
        if "BATCH" in norm_upper:
            raise ValueError(
                "BatchNorm is prohibited in FedMed due to federated weight drift; "
                "use INSTANCE or GROUP normalization."
            )

        self.net = UNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=tuple(channels),
            strides=tuple(strides),
            num_res_units=2,
            norm=norm_upper,
            dropout=dropout,
        )
        logger.info(
            "Initialized UNet3D with in_channels=%s, out_channels=%s, norm=%s",
            in_channels,
            out_channels,
            norm_upper,
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
    return UNet3D(
        in_channels=int(model_cfg["in_channels"]),
        out_channels=int(model_cfg["out_channels"]),
        channels=model_cfg["channels"],
        strides=model_cfg["strides"],
        norm=str(model_cfg["norm"]),
        dropout=float(model_cfg.get("dropout", 0.1)),
    )
