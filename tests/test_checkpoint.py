"""Unit tests for checkpoint persistence, atomic writing, and validation."""

from __future__ import annotations

import logging
from pathlib import Path
import pytest
import torch
import torch.nn as nn

from fedmed.training.checkpoint import (
    load_checkpoint,
    prune_checkpoints,
    save_checkpoint,
)


class SimpleModel(nn.Module):
    def __init__(self, in_features: int = 4, out_features: int = 2) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def test_checkpoint_save_load_roundtrip(tmp_path: Path) -> None:
    """Verify save/load round-trip restores identical weights and optimizer state."""
    model = SimpleModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Perform a dummy training step to give optimizer non-empty state
    x = torch.randn(2, 4)
    loss = model(x).sum()
    loss.backward()
    optimizer.step()

    ckpt_path = tmp_path / "latest.pt"
    save_checkpoint(
        path=ckpt_path,
        model=model,
        optimizer=optimizer,
        epoch=3,
        best_metric=0.75,
        cfg_hash="abc123456789",
        is_best=False,
    )

    # Instantiate fresh model and optimizer
    model_loaded = SimpleModel()
    opt_loaded = torch.optim.AdamW(model_loaded.parameters(), lr=1e-3)

    meta = load_checkpoint(
        ckpt_path,
        model=model_loaded,
        optimizer=opt_loaded,
        expected_cfg_hash="abc123456789",
    )

    assert meta["epoch"] == 3
    assert meta["best_metric"] == 0.75
    assert meta["cfg_hash"] == "abc123456789"

    for p1, p2 in zip(model.parameters(), model_loaded.parameters()):
        assert torch.allclose(p1, p2)

    # Check optimizer state restoration
    assert len(opt_loaded.state) > 0


def test_checkpoint_best_contains_no_optimizer(tmp_path: Path) -> None:
    """Verify best.pt is a small deployment artifact with no optimizer state."""
    model = SimpleModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    best_path = tmp_path / "best.pt"
    save_checkpoint(
        path=best_path,
        model=model,
        optimizer=optimizer,
        epoch=5,
        best_metric=0.88,
        cfg_hash="xyz987654321",
        is_best=True,
    )

    raw_payload = torch.load(best_path, weights_only=True)
    assert "model_state_dict" in raw_payload
    assert "optimizer_state_dict" not in raw_payload
    assert "scheduler_state_dict" not in raw_payload


def test_checkpoint_atomic_write_leaves_no_partial_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify atomic write leaves no partial file on simulated disk failure."""
    model = SimpleModel()
    target_path = tmp_path / "atomic_test.pt"

    def fail_rename(src: Any, dst: Any) -> None:
        raise OSError("Simulated disk error during atomic rename")

    monkeypatch.setattr("os.replace", fail_rename)

    with pytest.raises(OSError, match="Simulated disk error"):
        save_checkpoint(path=target_path, model=model)

    assert not target_path.exists(), "Target file must not exist if write was interrupted"
    tmp_files = list(tmp_path.glob("*.tmp*"))
    assert len(tmp_files) == 0, "Temporary files must be cleaned up on failure"


def test_checkpoint_prune_keep_last_k(tmp_path: Path) -> None:
    """Verify keep_last_k retains the most recent periodic checkpoints and deletes older ones."""
    model = SimpleModel()
    for ep in range(1, 6):
        ckpt_path = tmp_path / f"epoch_{ep}.pt"
        save_checkpoint(path=ckpt_path, model=model, epoch=ep)

    # Also create best.pt and latest.pt
    save_checkpoint(tmp_path / "best.pt", model=model, is_best=True)
    save_checkpoint(tmp_path / "latest.pt", model=model)

    deleted = prune_checkpoints(tmp_path, keep_last_k=2)
    assert len(deleted) == 3

    remaining_files = {p.name for p in tmp_path.glob("*.pt")}
    assert "epoch_4.pt" in remaining_files
    assert "epoch_5.pt" in remaining_files
    assert "epoch_1.pt" not in remaining_files
    assert "epoch_2.pt" not in remaining_files
    assert "epoch_3.pt" not in remaining_files
    assert "best.pt" in remaining_files
    assert "latest.pt" in remaining_files


def test_checkpoint_cfg_hash_mismatch_warns(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Verify configuration hash mismatch issues a warning rather than failing."""
    model = SimpleModel()
    ckpt_path = tmp_path / "hash_warn.pt"
    save_checkpoint(path=ckpt_path, model=model, cfg_hash="hash_alpha")

    with caplog.at_level(logging.WARNING):
        load_checkpoint(ckpt_path, model=model, expected_cfg_hash="hash_beta")

    assert "Checkpoint config hash mismatch" in caplog.text


def test_checkpoint_shape_mismatch_raises(tmp_path: Path) -> None:
    """Verify model shape mismatch raises ValueError with informative message."""
    model_small = SimpleModel(in_features=4, out_features=2)
    ckpt_path = tmp_path / "small.pt"
    save_checkpoint(path=ckpt_path, model=model_small)

    model_large = SimpleModel(in_features=8, out_features=2)
    with pytest.raises(ValueError, match="Model architecture shape mismatch for parameter"):
        load_checkpoint(ckpt_path, model=model_large)
