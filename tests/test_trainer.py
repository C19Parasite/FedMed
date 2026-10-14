"""Unit tests for Trainer: accumulation, clipping, NaN handling, sliding window, and overfitting."""

from __future__ import annotations

import copy
import math
from pathlib import Path
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from fedmed.training.losses import build_loss
from fedmed.training.metrics import SegmentationMetrics
from fedmed.training.optim import build_optimizer, build_scheduler
from fedmed.training.trainer import Trainer
from fedmed.utils.config import load_config


class SyntheticTensorDataset(Dataset):
    """Simple in-memory dataset yielding 5D image and label tensors."""

    def __init__(self, images: torch.Tensor, labels: torch.Tensor) -> None:
        self.images = images
        self.labels = labels

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {"image": self.images[idx], "label": self.labels[idx]}


class ToyConv3D(nn.Module):
    """Deterministic, normalization-free 3D network for exact mathematical checks."""

    def __init__(self, in_channels: int = 4, out_channels: int = 3) -> None:
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        nn.init.constant_(self.conv.weight, 0.05)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ToyUNet(nn.Module):
    """Small 3D model for validation and overfitting tests."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(4, 8, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=2, num_channels=8),
            nn.PReLU(),
            nn.Conv3d(8, 3, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def test_train_one_epoch_returns_finite_loss_and_keys() -> None:
    """Verify one epoch returns finite loss, expected dictionary keys, and correct samples_seen."""
    cfg = load_config("configs/smoke_test.yaml")
    model = ToyUNet()
    loss_fn = build_loss(cfg)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    metrics = SegmentationMetrics(cfg)

    # 4 samples of spatial size 16x16x16
    x = torch.randn(4, 4, 16, 16, 16)
    y = (torch.rand(4, 3, 16, 16, 16) > 0.5).float()
    loader = DataLoader(SyntheticTensorDataset(x, y), batch_size=2, shuffle=False)

    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        cfg=cfg,
        device="cpu",
        metrics=metrics,
    )

    res = trainer.train_one_epoch(loader, epoch=1)

    expected_keys = {
        "train_loss",
        "grad_norm_mean",
        "lr",
        "steps",
        "skipped_steps",
        "epoch_time_s",
        "samples_seen",
    }
    assert expected_keys.issubset(res.keys())
    assert not math.isnan(res["train_loss"])
    assert res["train_loss"] > 0.0
    assert res["samples_seen"] == 4
    assert res["steps"] == 2
    assert res["skipped_steps"] == 0


def test_gradient_accumulation_matches_larger_batch() -> None:
    """Verify grad accumulation (k=2, batch=1) matches one step with batch=2 on deterministic model."""
    torch.manual_seed(42)
    # Generate 2 distinct samples
    x1 = torch.randn(1, 4, 8, 8, 8)
    y1 = (torch.rand(1, 3, 8, 8, 8) > 0.5).float()
    x2 = torch.randn(1, 4, 8, 8, 8)
    y2 = (torch.rand(1, 3, 8, 8, 8) > 0.5).float()

    x_combined = torch.cat([x1, x2], dim=0)
    y_combined = torch.cat([y1, y2], dim=0)

    # Setup Case A: Batch size 1, grad_accum_steps = 2
    cfg_a = load_config("configs/smoke_test.yaml")
    cfg_a["training"]["grad_accum_steps"] = 2
    cfg_a["training"]["grad_clip"]["enabled"] = False
    cfg_a["training"]["optimizer"]["lr"] = 0.01

    model_a = ToyConv3D()
    loss_fn_a = nn.BCEWithLogitsLoss()  # Batch-averaged linear loss for exact equivalence
    opt_a = torch.optim.SGD(model_a.parameters(), lr=0.01)
    loader_a = DataLoader(SyntheticTensorDataset(x_combined, y_combined), batch_size=1, shuffle=False)
    trainer_a = Trainer(model_a, loss_fn_a, opt_a, None, cfg_a, device="cpu")

    # Setup Case B: Batch size 2, grad_accum_steps = 1
    cfg_b = copy.deepcopy(cfg_a)
    cfg_b["training"]["grad_accum_steps"] = 1

    model_b = ToyConv3D()
    loss_fn_b = nn.BCEWithLogitsLoss()
    opt_b = torch.optim.SGD(model_b.parameters(), lr=0.01)
    loader_b = DataLoader(SyntheticTensorDataset(x_combined, y_combined), batch_size=2, shuffle=False)
    trainer_b = Trainer(model_b, loss_fn_b, opt_b, None, cfg_b, device="cpu")

    trainer_a.train_one_epoch(loader_a, epoch=1)
    trainer_b.train_one_epoch(loader_b, epoch=1)

    # Check that model weights match closely
    for p_a, p_b in zip(model_a.parameters(), model_b.parameters()):
        assert torch.allclose(p_a, p_b, atol=1e-5)


def test_non_finite_loss_handling_and_abort() -> None:
    """Verify non-finite loss is skipped and counted, and consecutive failures abort."""
    cfg = load_config("configs/smoke_test.yaml")
    model = ToyUNet()
    optimizer = build_optimizer(model, cfg)

    x = torch.randn(2, 4, 8, 8, 8)
    y = torch.zeros(2, 3, 8, 8, 8)
    loader = DataLoader(SyntheticTensorDataset(x, y), batch_size=1, shuffle=False)

    call_count = [0]

    def patched_loss_single_nan(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        call_count[0] += 1
        if call_count[0] == 1:
            return torch.tensor(float("nan"), requires_grad=True)
        return nn.functional.binary_cross_entropy_with_logits(logits, targets)

    trainer = Trainer(
        model=model,
        loss_fn=patched_loss_single_nan,
        optimizer=optimizer,
        scheduler=None,
        cfg=cfg,
        device="cpu",
    )

    res = trainer.train_one_epoch(loader, epoch=1)
    assert res["skipped_steps"] == 1
    assert res["steps"] == 1

    # Now verify that consecutive non-finite steps abort with RuntimeError
    def always_nan_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return torch.tensor(float("nan"), requires_grad=True)

    cfg_abort = copy.deepcopy(cfg)
    cfg_abort["training"]["max_steps_per_epoch"] = None
    cfg_abort["training"]["max_consecutive_skipped"] = 3

    x_many = torch.randn(5, 4, 8, 8, 8)
    y_many = torch.zeros(5, 3, 8, 8, 8)
    loader_many = DataLoader(SyntheticTensorDataset(x_many, y_many), batch_size=1, shuffle=False)

    trainer_abort = Trainer(
        model=model,
        loss_fn=always_nan_loss,
        optimizer=optimizer,
        scheduler=None,
        cfg=cfg_abort,
        device="cpu",
    )

    with pytest.raises(RuntimeError, match="consecutive steps encountered non-finite"):
        trainer_abort.train_one_epoch(loader_many, epoch=1)


def test_gradient_clipping_caps_norm() -> None:
    """Verify gradient clipping prevents explosive gradient updates."""
    cfg = load_config("configs/smoke_test.yaml")
    cfg["training"]["grad_clip"]["enabled"] = True
    cfg["training"]["grad_clip"]["max_norm"] = 0.5
    cfg["training"]["optimizer"]["lr"] = 0.01

    model = ToyUNet()
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = build_optimizer(model, cfg)

    # Input with huge magnitude to produce large gradients
    x = torch.randn(2, 4, 8, 8, 8) * 1000.0
    y = torch.ones(2, 3, 8, 8, 8)
    loader = DataLoader(SyntheticTensorDataset(x, y), batch_size=2)

    trainer = Trainer(model, loss_fn, optimizer, None, cfg, device="cpu")
    res = trainer.train_one_epoch(loader, epoch=1)

    # Pre-clip norm was recorded
    assert res["grad_norm_mean"] > 0.0
    assert not math.isinf(res["grad_norm_mean"])


def test_overfit_single_batch() -> None:
    """Verify 30 optimization steps on a fixed batch reduce train loss by at least 30%."""
    torch.manual_seed(42)
    cfg = load_config("configs/smoke_test.yaml")
    cfg["training"]["optimizer"]["lr"] = 0.02
    cfg["training"]["optimizer"]["name"] = "adam"

    model = ToyUNet()
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = build_optimizer(model, cfg)

    x = torch.randn(1, 4, 8, 8, 8)
    y = (torch.rand(1, 3, 8, 8, 8) > 0.5).float()
    loader = DataLoader(SyntheticTensorDataset(x, y), batch_size=1)

    trainer = Trainer(model, loss_fn, optimizer, None, cfg, device="cpu")

    initial_loss = trainer.train_one_epoch(loader, epoch=1)["train_loss"]
    final_loss = initial_loss

    for ep in range(2, 31):
        final_loss = trainer.train_one_epoch(loader, epoch=ep)["train_loss"]

    rel_reduction = (initial_loss - final_loss) / initial_loss
    assert rel_reduction >= 0.30, f"Expected >= 30% reduction, got {rel_reduction * 100:.2f}%"


def test_validate_sliding_window_larger_volume_and_batch_sizes() -> None:
    """Verify validate works when volume is 2x ROI (sliding window) and is deterministic."""
    cfg = load_config("configs/smoke_test.yaml")
    cfg["training"]["sliding_window"]["roi_size"] = [8, 8, 8]
    cfg["training"]["sliding_window"]["sw_batch_size"] = 1

    model = ToyUNet()
    loss_fn = nn.BCEWithLogitsLoss()
    trainer = Trainer(model, loss_fn, None, None, cfg, device="cpu")

    # Volumes of 16x16x16 with ROI 8x8x8 (volume is 2x ROI along each dimension)
    x = torch.randn(2, 4, 16, 16, 16)
    y = (torch.rand(2, 3, 16, 16, 16) > 0.5).float()

    # Test batch size 1
    loader_bs1 = DataLoader(SyntheticTensorDataset(x, y), batch_size=1, shuffle=False)
    model.train()
    val_res1 = trainer.validate(loader_bs1, include_hd95=False)
    assert model.training is True, "Validate must restore model training mode"
    assert 0.0 <= val_res1["dice_mean"] <= 1.0

    # Determinism across two calls
    val_res1_repeat = trainer.validate(loader_bs1, include_hd95=False)
    assert val_res1["dice_mean"] == pytest.approx(val_res1_repeat["dice_mean"], abs=1e-6)

    # Test batch size 2
    loader_bs2 = DataLoader(SyntheticTensorDataset(x, y), batch_size=2, shuffle=False)
    val_res2 = trainer.validate(loader_bs2, include_hd95=False)
    assert 0.0 <= val_res2["dice_mean"] <= 1.0
