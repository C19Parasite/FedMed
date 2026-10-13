"""Unit tests for 3D U-Net model architecture and federated normalization rules."""

import pytest
import torch
import torch.nn as nn

from fedmed.models.norm import (
    assert_no_batchnorm,
    get_norm_layer_spec,
    validate_group_norm_channels,
    validate_roi_for_model,
)
from fedmed.models.unet3d import UNet3D, UNet3DCustom, get_model
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


def test_assert_no_batchnorm_passes_on_clean_model() -> None:
    """Verify assert_no_batchnorm succeeds when model uses only Instance/Group norms."""
    clean_model = nn.Sequential(
        nn.Conv3d(4, 16, 3, padding=1),
        nn.InstanceNorm3d(16),
        nn.ReLU(),
        nn.Conv3d(16, 16, 3, padding=1),
        nn.GroupNorm(4, 16),
    )
    assert_no_batchnorm(clean_model)


def test_assert_no_batchnorm_fails_on_batchnorm_layer() -> None:
    """Verify assert_no_batchnorm raises ValueError with details on model containing BatchNorm3d."""
    dirty_model = nn.Sequential(
        nn.Conv3d(4, 16, 3, padding=1),
        nn.BatchNorm3d(16),
        nn.ReLU(),
    )
    with pytest.raises(ValueError, match="BatchNorm is strictly prohibited"):
        assert_no_batchnorm(dirty_model)


def test_validate_roi_for_model_success() -> None:
    """Verify validate_roi_for_model passes when ROI is divisible by total downsampling."""
    cfg = {
        "data": {"roi_size": [64, 64, 64]},
        "model": {"strides": [2, 2, 2, 2]},  # product = 16. 64 % 16 == 0
    }
    validate_roi_for_model(cfg)


def test_validate_roi_for_model_failure() -> None:
    """Verify validate_roi_for_model raises ValueError when any dimension is not divisible."""
    cfg = {
        "data": {"roi_size": [64, 50, 64]},
        "model": {"strides": [2, 2, 2, 2]},  # product = 16. 50 % 16 != 0
    }
    with pytest.raises(ValueError, match="ROI dimensions must each be divisible"):
        validate_roi_for_model(cfg)


def test_validate_group_norm_channels() -> None:
    """Verify validate_group_norm_channels ensures channels are divisible by num_groups."""
    # Valid: 16, 32, 64 all divisible by 8
    validate_group_norm_channels([16, 32, 64], 8)

    # Invalid: 18 not divisible by 8
    with pytest.raises(ValueError, match="does not divide all channel widths"):
        validate_group_norm_channels([16, 18, 64], 8)


def test_get_norm_layer_spec_rejects_batchnorm() -> None:
    """Verify get_norm_layer_spec raises ValueError when BatchNorm is configured."""
    with pytest.raises(ValueError, match="prohibited"):
        get_norm_layer_spec("batch")

    with pytest.raises(ValueError, match="prohibited"):
        get_norm_layer_spec({"type": "batch"})


def test_unet3d_custom_forward_pass_shape() -> None:
    """Verify UNet3DCustom reference architecture produces correct output tensor shape."""
    model = UNet3DCustom(
        in_channels=4,
        out_channels=3,
        channels=[8, 16, 32],
        strides=[2, 2],
        norm_type="instance",
        use_residual=True,
    )
    assert_no_batchnorm(model)
    x = torch.randn(1, 4, 16, 16, 16)
    out = model(x)
    assert out.shape == (1, 3, 16, 16, 16)


def test_unet3d_custom_rejects_batchnorm() -> None:
    """Verify UNet3DCustom strictly rejects BatchNorm with clear ValueError."""
    with pytest.raises(ValueError, match="BatchNorm is prohibited"):
        UNet3DCustom(norm_type="batch")


def test_unet3d_custom_group_norm_and_no_residual() -> None:
    """Verify UNet3DCustom works with GroupNorm and without residual connections."""
    model = UNet3DCustom(
        in_channels=4,
        out_channels=3,
        channels=[8, 16],
        strides=[2],
        norm_type="group",
        num_groups=4,
        use_residual=False,
    )
    assert_no_batchnorm(model)
    x = torch.randn(2, 4, 16, 16, 16)
    out = model(x)
    assert out.shape == (2, 3, 16, 16, 16)
