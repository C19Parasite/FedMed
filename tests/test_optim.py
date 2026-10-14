"""Unit tests for optimizer and scheduler factories."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from fedmed.training.optim import build_optimizer, build_scheduler, get_lr
from fedmed.utils.config import load_config


class ToyModelWithNorm(nn.Module):
    """Small model containing conv, group norm, linear, and bias parameters."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv3d(4, 8, kernel_size=3, bias=True)
        self.norm = nn.GroupNorm(num_groups=2, num_channels=8)
        self.fc = nn.Linear(8, 3, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = x.mean(dim=(2, 3, 4))
        return self.fc(x)


def test_optimizer_weight_decay_parameter_groups() -> None:
    """Verify biases and normalization weights are excluded from weight decay."""
    model = ToyModelWithNorm()
    cfg = load_config("configs/baseline.yaml")
    cfg["training"]["optimizer"]["weight_decay"] = 0.05

    optimizer = build_optimizer(model, cfg)

    # Must have 2 parameter groups: decay and no_decay
    assert len(optimizer.param_groups) == 2
    decay_group = optimizer.param_groups[0]
    no_decay_group = optimizer.param_groups[1]

    assert decay_group["weight_decay"] == 0.05
    assert no_decay_group["weight_decay"] == 0.0

    # Conv weight and fc weight must be in decay group
    decay_params = set(decay_group["params"])
    assert model.conv.weight in decay_params
    assert model.fc.weight in decay_params

    # Biases and GroupNorm weight/bias must be in no-decay group
    no_decay_params = set(no_decay_group["params"])
    assert model.conv.bias in no_decay_params
    assert model.fc.bias in no_decay_params
    assert model.norm.weight in no_decay_params
    assert model.norm.bias in no_decay_params


def test_scheduler_warmup_then_cosine_decay() -> None:
    """Verify linear warmup followed by cosine annealing decay at known epochs."""
    model = ToyModelWithNorm()
    cfg = load_config("configs/baseline.yaml")
    cfg["training"]["epochs"] = 4
    cfg["training"]["optimizer"]["lr"] = 1.0
    cfg["training"]["scheduler"]["name"] = "cosine"
    cfg["training"]["scheduler"]["warmup_epochs"] = 2
    cfg["training"]["scheduler"]["min_lr"] = 0.1

    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    assert scheduler is not None

    # Epoch 0 (init): warmup starts at min_lr / base_lr = 0.1
    assert get_lr(optimizer) == pytest.approx(0.1, abs=1e-4)

    # Step after epoch 1: midway through warmup
    optimizer.step()
    scheduler.step()
    assert get_lr(optimizer) == pytest.approx(0.55, abs=1e-4)

    # Step after epoch 2: warmup complete, reached base_lr = 1.0
    optimizer.step()
    scheduler.step()
    assert get_lr(optimizer) == pytest.approx(1.0, abs=1e-4)

    # Step after epoch 3: cosine decay midway
    optimizer.step()
    scheduler.step()
    # T_max = 2 (epochs 3 and 4)
    # Cosine after 1 step of 2: 0.1 + 0.5 * (1.0 - 0.1) * (1 + cos(pi/2)) = 0.1 + 0.45 = 0.55
    assert get_lr(optimizer) == pytest.approx(0.55, abs=1e-4)

    # Step after epoch 4: reached min_lr = 0.1
    optimizer.step()
    scheduler.step()
    assert get_lr(optimizer) == pytest.approx(0.1, abs=1e-4)


def test_scheduler_poly_decay_endpoint_equals_min_lr() -> None:
    """Verify polynomial decay endpoint reaches min_lr exactly."""
    model = ToyModelWithNorm()
    cfg = load_config("configs/baseline.yaml")
    cfg["training"]["epochs"] = 4
    cfg["training"]["optimizer"]["lr"] = 1.0
    cfg["training"]["scheduler"]["name"] = "poly"
    cfg["training"]["scheduler"]["warmup_epochs"] = 1
    cfg["training"]["scheduler"]["min_lr"] = 0.05
    cfg["training"]["scheduler"]["poly_power"] = 1.0

    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    assert scheduler is not None

    # Step through all epochs
    for _ in range(4):
        optimizer.step()
        scheduler.step()

    # Final lr must equal min_lr exactly
    assert get_lr(optimizer) == pytest.approx(0.05, abs=1e-5)


def test_scheduler_none_returns_none() -> None:
    """Verify scheduler name 'none' returns None."""
    model = ToyModelWithNorm()
    cfg = load_config("configs/baseline.yaml")
    cfg["training"]["scheduler"]["name"] = "none"

    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    assert scheduler is None
