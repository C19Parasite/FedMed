"""Unit tests for training resumption and bit-reproducibility with RNG state."""

from __future__ import annotations

import csv
from pathlib import Path
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from fedmed.training.checkpoint import load_checkpoint
from fedmed.training.losses import build_loss
from fedmed.training.optim import build_optimizer, build_scheduler
from fedmed.training.run import create_run_dir
from fedmed.training.trainer import Trainer
from fedmed.utils.config import load_config
from fedmed.utils.seed import set_seed


class SyntheticDataset(Dataset):
    def __init__(self, data: torch.Tensor, targets: torch.Tensor) -> None:
        self.data = data
        self.targets = targets

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {"image": self.data[idx], "label": self.targets[idx]}


class ResumeToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv3d(4, 3, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


def test_resume_identical_weights_and_clean_csv(tmp_path: Path) -> None:
    """Verify train 4 epochs straight vs train 2, interrupt, resume to 4 yields IDENTICAL weights and 4 CSV rows."""
    set_seed(42)
    x_train = torch.randn(4, 4, 8, 8, 8)
    y_train = (torch.rand(4, 3, 8, 8, 8) > 0.5).float()
    x_val = torch.randn(2, 4, 8, 8, 8)
    y_val = (torch.rand(2, 3, 8, 8, 8) > 0.5).float()

    train_loader = DataLoader(SyntheticDataset(x_train, y_train), batch_size=2, shuffle=False)
    val_loader = DataLoader(SyntheticDataset(x_val, y_val), batch_size=1, shuffle=False)

    # 1. Straight run for 4 epochs
    set_seed(100)
    cfg_straight = load_config("configs/smoke_test.yaml")
    cfg_straight["training"]["epochs"] = 4
    cfg_straight["training"]["val_every_n_epochs"] = 1
    cfg_straight["training"]["early_stopping"]["enabled"] = False
    run_dir_straight = tmp_path / "straight_run"
    cfg_straight["training"]["logging"]["output_dir"] = str(tmp_path)
    cfg_straight["training"]["logging"]["run_name"] = "straight_run"

    model_straight = ResumeToyModel()
    opt_straight = build_optimizer(model_straight, cfg_straight)
    sched_straight = build_scheduler(opt_straight, cfg_straight)
    loss_straight = build_loss(cfg_straight)

    trainer_straight = Trainer(
        model=model_straight,
        loss_fn=loss_straight,
        optimizer=opt_straight,
        scheduler=sched_straight,
        cfg=cfg_straight,
        device="cpu",
        run_dir=run_dir_straight,
    )
    trainer_straight.fit(train_loader, val_loader, start_epoch=0)

    # 2. Segment A: Train 2 epochs and stop
    set_seed(100)
    cfg_resume = load_config("configs/smoke_test.yaml")
    cfg_resume["training"]["epochs"] = 2
    cfg_resume["training"]["val_every_n_epochs"] = 1
    cfg_resume["training"]["early_stopping"]["enabled"] = False
    run_dir_resume = tmp_path / "resumed_run"
    cfg_resume["training"]["logging"]["output_dir"] = str(tmp_path)
    cfg_resume["training"]["logging"]["run_name"] = "resumed_run"

    model_resume = ResumeToyModel()
    opt_resume = build_optimizer(model_resume, cfg_resume)
    sched_resume = build_scheduler(opt_resume, cfg_resume)
    loss_resume = build_loss(cfg_resume)

    trainer_part1 = Trainer(
        model=model_resume,
        loss_fn=loss_resume,
        optimizer=opt_resume,
        scheduler=sched_resume,
        cfg=cfg_resume,
        device="cpu",
        run_dir=run_dir_resume,
    )
    trainer_part1.fit(train_loader, val_loader, start_epoch=0)

    # 3. Segment B: Resume from epoch 2 to 4
    cfg_resume["training"]["epochs"] = 4
    cfg_resume["training"]["resume"]["enabled"] = True
    cfg_resume["training"]["resume"]["path"] = str(run_dir_resume / "checkpoints" / "latest.pt")

    # Re-initialize trainer and load checkpoint
    model_part2 = ResumeToyModel()
    opt_part2 = build_optimizer(model_part2, cfg_resume)
    sched_part2 = build_scheduler(opt_part2, cfg_resume)

    trainer_part2 = Trainer(
        model=model_part2,
        loss_fn=loss_resume,
        optimizer=opt_part2,
        scheduler=sched_part2,
        cfg=cfg_resume,
        device="cpu",
        run_dir=run_dir_resume,
    )
    meta = trainer_part2.load_checkpoint(run_dir_resume / "checkpoints" / "latest.pt")
    assert meta["epoch"] == 2

    trainer_part2.fit(train_loader, val_loader, start_epoch=meta["epoch"])

    # Assert that final weights match exactly
    for p_straight, p_resumed in zip(model_straight.parameters(), model_part2.parameters()):
        assert torch.allclose(p_straight, p_resumed, atol=1e-6)

    # Assert CSV has exactly 4 data rows and no duplicates
    csv_file = run_dir_resume / "results" / "metrics.csv"
    with csv_file.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 4
    epochs = [int(r["epoch"]) for r in rows]
    assert epochs == [1, 2, 3, 4]
