"""3D U-Net architecture for multi-modal brain tumor MRI segmentation.

Contains:
1. UNet3DCustom: A transparent, from-scratch 3D U-Net reference architecture
   with configurable depth, channels, residual blocks, and federated-safe normalization
   (GroupNorm or InstanceNorm only, BatchNorm strictly prohibited).
2. UNet3D: Backwards-compatible MONAI-based 3D U-Net wrapper.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence
import torch
import torch.nn as nn
from monai.networks.nets import UNet
from monai.networks.layers.factories import Norm

from fedmed.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# AdaptiveGroupNorm for MONAI compatibility
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# From-scratch 3D U-Net Reference Architecture
# ---------------------------------------------------------------------------

def _make_norm(norm_type: str, num_channels: int, num_groups: int = 8) -> nn.Module:
    """Create a federated-safe normalization layer.

    Raises ValueError if BatchNorm is requested.
    """
    norm_lower = norm_type.lower()
    if "batch" in norm_lower:
        raise ValueError(
            "model.norm.type='batch' is prohibited. BatchNorm running statistics "
            "diverge under non-IID client data, are not meaningfully averaged by FedAvg, "
            "and are incompatible with per-sample gradient clipping (differential privacy). "
            "Use 'group' or 'instance' instead."
        )

    if norm_lower == "group":
        if num_channels % num_groups != 0:
            # Fall back to largest divisor <= num_groups for channels not divisible
            g = num_groups
            while g > 1 and num_channels % g != 0:
                g -= 1
            return nn.GroupNorm(num_groups=g, num_channels=num_channels)
        return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)
    elif norm_lower == "instance":
        return nn.InstanceNorm3d(num_channels, affine=True)
    else:
        raise ValueError(f"Unknown norm_type '{norm_type}'; expected 'group' or 'instance'.")


class ConvBlock3D(nn.Module):
    """Two consecutive 3D convolutions with normalization, activation, and optional residual connection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_type: str = "instance",
        num_groups: int = 8,
        use_residual: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.use_residual = use_residual

        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = _make_norm(norm_type, out_channels, num_groups)
        self.act1 = nn.PReLU()

        self.dropout = nn.Dropout3d(dropout) if dropout > 0.0 else nn.Identity()

        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = _make_norm(norm_type, out_channels, num_groups)
        self.act2 = nn.PReLU()

        if self.use_residual:
            if in_channels != out_channels:
                self.residual_proj: nn.Module = nn.Sequential(
                    nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
                    _make_norm(norm_type, out_channels, num_groups),
                )
            else:
                self.residual_proj = nn.Identity()
        else:
            self.residual_proj = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.residual_proj(x) if self.use_residual else None

        out = self.act1(self.norm1(self.conv1(x)))
        out = self.dropout(out)
        out = self.norm2(self.conv2(out))

        if self.use_residual and res is not None:
            out = out + res

        return self.act2(out)


class EncoderStage3D(nn.Module):
    """Encoder resolution level: ConvBlock3D followed by downsampling."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 2,
        norm_type: str = "instance",
        num_groups: int = 8,
        use_residual: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.conv_block = ConvBlock3D(
            in_channels=in_channels,
            out_channels=out_channels,
            norm_type=norm_type,
            num_groups=num_groups,
            use_residual=use_residual,
            dropout=dropout,
        )
        self.stride = stride
        if stride > 1:
            self.downsample: nn.Module = nn.Conv3d(
                out_channels, out_channels, kernel_size=stride, stride=stride, bias=False
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        skip = self.conv_block(x)
        down = self.downsample(skip)
        return down, skip


class DecoderStage3D(nn.Module):
    """Decoder resolution level: ConvTranspose3d upsampling, skip concatenation, and ConvBlock3D."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        stride: int = 2,
        norm_type: str = "instance",
        num_groups: int = 8,
        use_residual: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.upsample = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size=stride,
            stride=stride,
            bias=False,
        )
        self.conv_block = ConvBlock3D(
            in_channels=out_channels + skip_channels,
            out_channels=out_channels,
            norm_type=norm_type,
            num_groups=num_groups,
            use_residual=use_residual,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        up = self.upsample(x)
        # Handle small spatial dimension mismatches via padding if needed
        if up.shape[2:] != skip.shape[2:]:
            diff = [s - u for s, u in zip(skip.shape[2:], up.shape[2:])]
            pad = [
                diff[2] // 2, diff[2] - diff[2] // 2,
                diff[1] // 2, diff[1] - diff[1] // 2,
                diff[0] // 2, diff[0] - diff[0] // 2,
            ]
            up = nn.functional.pad(up, pad)
        concat = torch.cat([skip, up], dim=1)
        return self.conv_block(concat)


class UNet3DCustom(nn.Module):
    """From-scratch, transparent 3D U-Net reference architecture for brain tumor MRI segmentation.

    Features:
    - Modular encoder, bottleneck, and decoder paths.
    - Configurable depth, base channels, and channel progressions.
    - Skip connections via channel-wise concatenation.
    - Optional residual connections in convolution blocks.
    - Strict federated normalization enforcement (InstanceNorm or GroupNorm; BatchNorm is banned).

    Args:
        in_channels: Number of input MRI channels (typically 4: T1, T1ce, T2, FLAIR).
        out_channels: Number of output tumor classes (typically 3: TC, WT, ET).
        channels: Sequence of channel counts per level (e.g. (16, 32, 64, 128)).
            If None, computed from base_channels and depth.
        strides: Sequence of downsampling strides per encoder step (default: (2, 2, 2)).
        depth: Number of resolution levels if channels is None (default: 4).
        base_channels: Starting channel width if channels is None (default: 16).
        norm_type: Normalization type ('instance' or 'group').
        num_groups: Number of groups for GroupNorm (default: 8).
        use_residual: Whether to enable residual skip connections within conv blocks.
        dropout: Spatial dropout probability.
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 3,
        channels: Sequence[int] | None = None,
        strides: Sequence[int] | None = None,
        depth: int = 4,
        base_channels: int = 16,
        norm_type: str = "instance",
        num_groups: int = 8,
        use_residual: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        norm_type_lower = norm_type.lower()
        if "batch" in norm_type_lower:
            raise ValueError(
                "BatchNorm is prohibited in FedMed due to federated weight drift; "
                "use INSTANCE or GROUP normalization."
            )

        if channels is not None:
            self.channel_list = list(channels)
        else:
            self.channel_list = [base_channels * (2**i) for i in range(depth)]

        if len(self.channel_list) < 2:
            raise ValueError(f"channels must contain at least 2 levels, got: {self.channel_list}")

        num_down = len(self.channel_list) - 1
        if strides is not None:
            self.strides = list(strides)
            if len(self.strides) < num_down:
                self.strides += [2] * (num_down - len(self.strides))
        else:
            self.strides = [2] * num_down

        # Encoder stages
        self.encoder_stages = nn.ModuleList()
        current_in = in_channels
        for i in range(num_down):
            stage = EncoderStage3D(
                in_channels=current_in,
                out_channels=self.channel_list[i],
                stride=self.strides[i],
                norm_type=norm_type,
                num_groups=num_groups,
                use_residual=use_residual,
                dropout=dropout,
            )
            self.encoder_stages.append(stage)
            current_in = self.channel_list[i]

        # Bottleneck
        self.bottleneck = ConvBlock3D(
            in_channels=current_in,
            out_channels=self.channel_list[-1],
            norm_type=norm_type,
            num_groups=num_groups,
            use_residual=use_residual,
            dropout=dropout,
        )

        # Decoder stages
        self.decoder_stages = nn.ModuleList()
        bottleneck_out = self.channel_list[-1]
        for i in range(num_down - 1, -1, -1):
            skip_ch = self.channel_list[i]
            out_ch = self.channel_list[i]
            stage = DecoderStage3D(
                in_channels=bottleneck_out,
                skip_channels=skip_ch,
                out_channels=out_ch,
                stride=self.strides[i],
                norm_type=norm_type,
                num_groups=num_groups,
                use_residual=use_residual,
                dropout=dropout,
            )
            self.decoder_stages.append(stage)
            bottleneck_out = out_ch

        # Final 1x1x1 convolution projecting to tumor class channels
        self.final_conv = nn.Conv3d(self.channel_list[0], out_channels, kernel_size=1)

        logger.info(
            "Initialized UNet3DCustom (depth=%d, channels=%s, norm=%s, residual=%s)",
            len(self.channel_list),
            self.channel_list,
            norm_type,
            use_residual,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through custom 3D U-Net.

        Args:
            x: Input tensor of shape (B, in_channels, D, H, W).

        Returns:
            Output logits tensor of shape (B, out_channels, D, H, W).
        """
        skips: list[torch.Tensor] = []
        feat = x
        for stage in self.encoder_stages:
            feat, skip = stage(feat)
            skips.append(skip)

        feat = self.bottleneck(feat)

        for stage, skip in zip(self.decoder_stages, reversed(skips)):
            feat = stage(feat, skip)

        return self.final_conv(feat)


# ---------------------------------------------------------------------------
# Backwards-compatible MONAI UNet wrapper
# ---------------------------------------------------------------------------

class UNet3D(nn.Module):
    """3D U-Net segmentation network using InstanceNorm or GroupNorm for federated stability."""

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
        return self.net(x)


def get_model(cfg: Mapping[str, Any]) -> UNet3D:
    """Build and return UNet3D from validated configuration mapping."""
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
