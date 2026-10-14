"""Privacy and modularity test: verifies centralized pipeline executes without FL or DP dependencies."""

from __future__ import annotations

import sys
from pathlib import Path
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class BlockedImports:
    """Context manager blocking specific module imports via sys.modules poisoning."""

    def __init__(self, blocked: list[str]) -> None:
        self.blocked = blocked
        self.original_modules: dict[str, Any] = {}

    def __enter__(self) -> None:
        for name in self.blocked:
            self.original_modules[name] = sys.modules.get(name)
            sys.modules[name] = None  # Causes ModuleNotFoundError on import attempt

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        for name in self.blocked:
            if self.original_modules[name] is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = self.original_modules[name]


class DummyDataset(Dataset):
    def __len__(self) -> int:
        return 2

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "image": torch.zeros(4, 8, 8, 8),
            "label": torch.zeros(3, 8, 8, 8),
        }


def test_centralized_training_with_blocked_fl_and_privacy_libs() -> None:
    """Verify fedmed.training runs with tenseal, fastapi, and flwr blocked from import."""
    with BlockedImports(["tenseal", "fastapi", "flwr"]):
        # Confirm they are indeed blocked
        with pytest.raises(ModuleNotFoundError):
            import flwr  # type: ignore

        with pytest.raises(ModuleNotFoundError):
            import tenseal  # type: ignore

        with pytest.raises(ModuleNotFoundError):
            import fastapi  # type: ignore

        # Importing and running Trainer must succeed completely
        from fedmed.training.losses import build_loss
        from fedmed.training.optim import build_optimizer
        from fedmed.training.trainer import Trainer
        from fedmed.utils.config import load_config

        cfg = load_config("configs/smoke_test.yaml")
        model = nn.Conv3d(4, 3, 1)
        optimizer = build_optimizer(model, cfg)
        loss_fn = nn.BCEWithLogitsLoss()

        trainer = Trainer(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=None,
            cfg=cfg,
            device="cpu",
        )

        loader = DataLoader(DummyDataset(), batch_size=1)
        res = trainer.train_one_epoch(loader, epoch=1)

        assert "train_loss" in res
        assert res["steps"] > 0
