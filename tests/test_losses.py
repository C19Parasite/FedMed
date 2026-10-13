"""Unit tests for segmentation loss functions and deep supervision wrapper."""

import pytest
import torch

from fedmed.training.losses import DeepSupervisionLoss, build_loss


@pytest.mark.parametrize("loss_name", ["dice_ce", "dice_focal", "dice"])
def test_loss_scalar_finite_and_differentiable(loss_name: str) -> None:
    """Verify loss outputs scalar, finite tensor with backward non-zero gradients."""
    cfg = {
        "loss": {
            "name": loss_name,
            "lambda_dice": 1.0,
            "lambda_ce": 1.0,
            "lambda_focal": 1.0,
            "smooth_nr": 1e-5,
            "smooth_dr": 1e-5,
            "channel_weights": [1.0, 1.0, 1.0],
        }
    }
    criterion = build_loss(cfg)

    logits = torch.randn(2, 3, 16, 16, 16, requires_grad=True)
    targets = (torch.rand(2, 3, 16, 16, 16) > 0.5).float()

    loss = criterion(logits, targets)

    # Must be scalar, finite, requiring gradient
    assert loss.ndim == 0
    assert torch.isfinite(loss).item()
    assert loss.requires_grad

    # Backward pass produces non-zero gradients
    loss.backward()
    assert logits.grad is not None
    assert torch.sum(torch.abs(logits.grad)).item() > 0.0


@pytest.mark.parametrize("loss_name", ["dice_ce", "dice_focal", "dice"])
def test_loss_empty_labels_is_finite(loss_name: str) -> None:
    """Verify loss remains finite and non-NaN when targets are all zero (no tumor)."""
    cfg = {"loss": {"name": loss_name, "smooth_nr": 1e-5, "smooth_dr": 1e-5}}
    criterion = build_loss(cfg)

    logits = torch.randn(2, 3, 16, 16, 16)
    empty_targets = torch.zeros(2, 3, 16, 16, 16)

    loss = criterion(logits, empty_targets)
    assert torch.isfinite(loss).item()


def test_loss_near_zero_on_perfect_match() -> None:
    """Verify loss approaches zero when logits strongly predict binary ground truth."""
    cfg = {"loss": {"name": "dice", "smooth_nr": 1e-5, "smooth_dr": 1e-5}}
    criterion = build_loss(cfg)

    targets = torch.zeros(1, 3, 16, 16, 16)
    targets[0, :, 2:14, 2:14, 2:14] = 1.0

    # Logits strongly positive (+10.0) where target is 1, strongly negative (-10.0) where 0
    logits = torch.where(targets == 1.0, torch.tensor(10.0), torch.tensor(-10.0))

    loss = criterion(logits, targets)
    assert loss.item() < 0.01, f"Expected loss near zero for perfect match, got {loss.item()}"


def test_deep_supervision_wrapper() -> None:
    """Verify DeepSupervisionLoss weights sum to 1.0, handles 6D tensors, and is a no-op on 5D."""
    cfg = {"loss": {"name": "dice_ce"}}
    criterion = build_loss(cfg)
    assert isinstance(criterion, DeepSupervisionLoss)

    # 1. Verification of weights computation
    weights = criterion._compute_weights(n_stages=4, device=torch.device("cpu"))
    assert weights.shape[0] == 4
    assert torch.isclose(weights.sum(), torch.tensor(1.0))
    # Halving progression check
    assert torch.isclose(weights[0], 2.0 * weights[1], atol=1e-5)

    # 2. 6D stacked tensor from DynUNet [B, N, C, D, H, W]
    logits_6d = torch.randn(2, 3, 3, 16, 16, 16, requires_grad=True)
    targets = (torch.rand(2, 3, 16, 16, 16) > 0.5).float()

    loss_6d = criterion(logits_6d, targets)
    assert loss_6d.ndim == 0
    assert torch.isfinite(loss_6d).item()
    loss_6d.backward()
    assert logits_6d.grad is not None

    # 3. No-op verification on standard 5D tensor
    logits_5d = torch.randn(2, 3, 16, 16, 16)
    loss_wrapped = criterion(logits_5d, targets)
    loss_direct = criterion.base_loss(logits_5d, targets)
    assert torch.isclose(loss_wrapped, loss_direct)


def test_unknown_loss_name_raises() -> None:
    """Verify build_loss raises ValueError on unknown loss function name."""
    cfg = {"loss": {"name": "unknown_loss"}}
    with pytest.raises(ValueError, match="Unsupported loss name"):
        build_loss(cfg)
