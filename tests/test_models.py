"""Unit tests for 3D U-Net model architecture and federated normalization rules."""

import pytest
import torch

from fedmed.models.unet3d import UNet3D, get_model
from fedmed.utils.config import load_config


def test_unet3d_forward_pass_shape() -> None:
    """Verify that UNet3D maps (B, 4, D, H, W) to (B, 3, D, H, W)."""
    model = UNet3D(
        in_channels=4,
        out_channels=3,
        channels=(8, 16, 32),
        strides=(2, 2),
        norm="INSTANCE",
    )
    x = torch.randn(2, 4, 16, 16, 16)
    out = model(x)
    assert out.shape == (2, 3, 16, 16, 16)


def test_unet3d_rejects_batchnorm() -> None:
    """Verify that BatchNorm raises ValueError due to federated normalization rule."""
    with pytest.raises(ValueError, match="BatchNorm is prohibited"):
        UNet3D(norm="BATCH")


def test_get_model_from_config() -> None:
    """Verify get_model instantiates network using baseline.yaml parameters."""
    cfg = load_config("configs/baseline.yaml")
    model = get_model(cfg)
    assert isinstance(model, UNet3D)

    x = torch.randn(1, 4, 32, 32, 32)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 3, 32, 32, 32)
