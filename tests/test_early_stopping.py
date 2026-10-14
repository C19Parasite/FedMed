"""Unit tests for early stopping patience and min_delta validation logic."""

from __future__ import annotations

from pathlib import Path
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from fedmed.training.trainer import Trainer
from fedmed.utils.config import load_config


class DummyDataset(Dataset):
    def __len__(self) -> int:
        return 2

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "image": torch.zeros(4, 8, 8, 8),
            "label": torch.zeros(3, 8, 8, 8),
        }


class DummyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv3d(4, 3, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


def test_early_stopping_patience_in_validation_rounds(tmp_path: Path) -> None:
    """Verify early stopping patience is counted in validation rounds, stopping at expected epoch."""
    cfg = load_config("configs/smoke_test.yaml")
    cfg["training"]["epochs"] = 10
    cfg["training"]["val_every_n_epochs"] = 2
    cfg["training"]["early_stopping"]["enabled"] = True
    cfg["training"]["early_stopping"]["patience"] = 2
    cfg["training"]["early_stopping"]["min_delta"] = 0.01
    cfg["training"]["early_stopping"]["monitor"] = "val_dice_mean"
    cfg["training"]["early_stopping"]["mode"] = "max"

    model = DummyModel()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.BCEWithLogitsLoss()

    loader = DataLoader(DummyDataset(), batch_size=1)
    run_dir = tmp_path / "es_test"

    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=opt,
        scheduler=None,
        cfg=cfg,
        device="cpu",
        run_dir=run_dir,
    )

    # Scripted validation metric sequence for epochs:
    # val_every_n_epochs is 2:
    # Epoch 2: 0.80 -> improvement! patience=0
    # Epoch 4: 0.805 -> < min_delta (0.01), no improvement, patience=1
    # Epoch 6: 0.79 -> no improvement, patience=2 -> triggers STOP!
    # Training must stop at epoch 6, not reaching epochs 7-10.
    scripted_dice = {
        2: 0.80,
        4: 0.805,
        6: 0.79,
        8: 0.85,
        10: 0.90,
    }

    original_validate = trainer.validate

    def patched_validate(l: Any, include_hd95: bool = False) -> dict[str, Any]:
        val_res = original_validate(l, include_hd95=include_hd95)
        # Identify current epoch from len(trainer.optimizer.param_groups) or call count
        # Or look at last row in history
        current_ep = getattr(trainer, "_current_epoch_tracker", 2)
        val_res["val_dice_mean"] = scripted_dice.get(current_ep, 0.5)
        return val_res

    trainer.validate = patched_validate  # type: ignore

    # Track epoch in train_one_epoch
    original_train = trainer.train_one_epoch

    def patched_train_one_epoch(l: Any, epoch: int = 1) -> dict[str, Any]:
        trainer._current_epoch_tracker = epoch  # type: ignore
        return original_train(l, epoch=epoch)

    trainer.train_one_epoch = patched_train_one_epoch  # type: ignore

    res = trainer.fit(loader, loader, start_epoch=0)

    # Must stop at epoch 6
    assert len(res["history"]) == 6
    assert res["best_epoch"] == 2
    assert res["best_metric"] == 0.80
