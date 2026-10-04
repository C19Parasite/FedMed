"""Unit tests for losses, metrics, and centralized training workflow."""

from pathlib import Path
import pytest
import torch

from fedmed.data.loaders import get_loaders
from fedmed.models.unet3d import UNet3D
from fedmed.training.losses import get_loss
from fedmed.training.metrics import compute_dice_score, evaluate_brats_dice
from fedmed.training.trainer import CentralizedTrainer
from fedmed.utils.config import load_config


def test_dice_metric_perfect_and_zero_overlap() -> None:
    """Verify Dice metric yields 1.0 for perfect overlap and 0.0 for disjoint sets."""
    mask_a = torch.ones((1, 1, 8, 8, 8))
    mask_b = torch.ones((1, 1, 8, 8, 8))
    mask_c = torch.zeros((1, 1, 8, 8, 8))

    assert compute_dice_score(mask_a, mask_b) == pytest.approx(1.0, rel=1e-3)
    assert compute_dice_score(mask_a, mask_c) == pytest.approx(0.0, abs=1e-4)


def test_evaluate_brats_dice_keys() -> None:
    """Verify evaluate_brats_dice computes all three sub-region scores."""
    logits = torch.randn(2, 3, 8, 8, 8)
    targets = (torch.rand(2, 3, 8, 8, 8) > 0.5).float()

    scores = evaluate_brats_dice(logits, targets)
    assert "dice_tc" in scores
    assert "dice_wt" in scores
    assert "dice_et" in scores
    assert "dice_mean" in scores
    for val in scores.values():
        assert 0.0 <= val <= 1.0


def test_centralized_trainer_single_step(tmp_path: Path) -> None:
    """Verify CentralizedTrainer completes training and saves checkpoint on synthetic data."""
    cfg = load_config("configs/baseline.yaml")
    cfg["data"]["root_dir"] = str(tmp_path / "data")
    cfg["data"]["splits_path"] = str(tmp_path / "data" / "splits.json")
    cfg["data"]["metadata_path"] = str(tmp_path / "data" / "metadata.json")
    cfg["data"]["cache_dir"] = str(tmp_path / "data" / "cache")
    cfg["data"]["roi_size"] = [16, 16, 16]
    cfg["data"]["synthetic"]["num_patients"] = 2
    cfg["data"]["synthetic"]["spatial_size"] = [16, 16, 16]
    cfg["training"]["save_dir"] = str(tmp_path / "checkpoints")

    model = UNet3D(
        in_channels=4,
        out_channels=3,
        channels=(4, 8),
        strides=(2,),
        norm="INSTANCE",
    )
    loaders = get_loaders(cfg)
    trainer = CentralizedTrainer(model=model, cfg=cfg, device="cpu")

    loss = trainer.train_epoch(loaders["train"])
    assert isinstance(loss, float)
    assert loss > 0.0

    val_res = trainer.evaluate(loaders["val"])
    assert "dice_mean" in val_res

    # Checkpoint saving test
    ckpt_file = trainer.save_checkpoint("test_ckpt.pt", epoch=1, metric=0.5)
    assert ckpt_file.is_file()
