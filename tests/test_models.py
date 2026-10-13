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


from fedmed.models.factory import build_model
from fedmed.utils.seed import seed_everything


@pytest.mark.parametrize("arch_name", ["monai_unet", "segresnet", "dynunet", "unet3d_custom"])
def test_build_model_forward_all_architectures(arch_name: str) -> None:
    """Verify build_model builds all 4 architectures and returns correct tensor shapes."""
    cfg = {
        "data": {"roi_size": [32, 32, 32]},
        "model": {
            "name": arch_name,
            "in_channels": 4,
            "out_channels": 3,
            "channels": [8, 16, 32],
            "strides": [2, 2],
            "num_res_units": 0,
            "kernel_size": 3,
            "act": "PRELU",
            "dropout": 0.0,
            "norm": {"type": "instance"},
            "init": {"name": "kaiming"},
            # Arch specifics
            "init_filters": 8,
            "blocks_down": [1, 2, 2],
            "blocks_up": [1, 1],
            "deep_supervision": False,
            "use_residual": True,
            "depth": 3,
            "base_channels": 8,
        },
    }

    model = build_model(cfg)
    assert_no_batchnorm(model)
    model.eval()

    x = torch.randn(1, 4, 32, 32, 32)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 3, 32, 32, 32)


def test_build_model_dynunet_deep_supervision() -> None:
    """Verify DynUNet deep_supervision output shape contract in train vs eval mode."""
    cfg = {
        "data": {"roi_size": [32, 32, 32]},
        "model": {
            "name": "dynunet",
            "in_channels": 4,
            "out_channels": 3,
            "channels": [8, 16, 32],
            "strides": [2, 2],
            "kernel_size": 3,
            "norm": {"type": "instance"},
            "deep_supervision": True,
            "deep_supr_num": 1,
        },
    }
    model = build_model(cfg)

    x = torch.randn(1, 4, 32, 32, 32)

    # Train mode: returns stacked outputs [B, DS+1, C, D, H, W]
    model.train()
    out_train = model(x)
    assert out_train.ndim == 6
    assert out_train.shape[0] == 1  # batch
    assert out_train.shape[1] == 2  # deep_supr_num + 1
    assert out_train.shape[2] == 3  # out_channels

    # Eval mode: returns standard single tensor [B, C, D, H, W]
    model.eval()
    with torch.no_grad():
        out_eval = model(x)
    assert out_eval.shape == (1, 3, 32, 32, 32)


def test_build_model_rejects_batchnorm() -> None:
    """Verify build_model raises ValueError when norm type is batch."""
    cfg = {
        "model": {
            "name": "monai_unet",
            "in_channels": 4,
            "out_channels": 3,
            "channels": [8, 16],
            "strides": [2],
            "norm": {"type": "batch"},
        }
    }
    with pytest.raises(ValueError, match="BatchNorm is prohibited"):
        build_model(cfg)


def test_build_model_seed_determinism() -> None:
    """Verify identical seeds produce identical initial weights, different seeds differ."""
    cfg = {
        "model": {
            "name": "unet3d_custom",
            "in_channels": 4,
            "out_channels": 3,
            "channels": [4, 8],
            "strides": [2],
            "norm": {"type": "instance"},
            "init": {"name": "kaiming"},
        }
    }

    seed_everything(100)
    m1 = build_model(cfg)

    seed_everything(100)
    m2 = build_model(cfg)

    seed_everything(200)
    m3 = build_model(cfg)

    # m1 and m2 weights must be identical
    for p1, p2 in zip(m1.parameters(), m2.parameters()):
        assert torch.allclose(p1, p2)

    # m1 and m3 weights must differ
    diff = any(not torch.allclose(p1, p3) for p1, p3 in zip(m1.parameters(), m3.parameters()))
    assert diff


def test_build_model_overfit_synthetic_batch() -> None:
    """Verify 40 optimizer steps on one synthetic batch reduces loss by at least 30%."""
    cfg = {
        "model": {
            "name": "unet3d_custom",
            "in_channels": 4,
            "out_channels": 3,
            "channels": [4, 8],
            "strides": [2],
            "norm": {"type": "instance"},
            "init": {"name": "kaiming"},
        }
    }
    seed_everything(42)
    model = build_model(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    criterion = nn.BCEWithLogitsLoss()

    x = torch.randn(1, 4, 16, 16, 16)
    y = (torch.rand(1, 3, 16, 16, 16) > 0.5).float()

    model.train()
    initial_loss = float("inf")
    final_loss = float("inf")

    for step in range(40):
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        if step == 0:
            initial_loss = loss.item()
        final_loss = loss.item()

    reduction = (initial_loss - final_loss) / initial_loss
    assert reduction >= 0.30, f"Expected >= 30% reduction, got {reduction * 100:.2f}%"
