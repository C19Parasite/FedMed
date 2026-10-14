"""Unit tests for metrics CSV logger schema, empty cells, and resume truncation."""

from __future__ import annotations

import csv
from pathlib import Path
import pytest

from fedmed.training.csv_logger import METRICS_CSV_COLUMNS, MetricsCSVLogger


def test_csv_logger_header_exact_match(tmp_path: Path) -> None:
    """Verify metrics CSV header matches METRICS_CSV_COLUMNS exactly."""
    csv_file = tmp_path / "metrics.csv"
    logger = MetricsCSVLogger(csv_file)

    with csv_file.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)

    assert header == METRICS_CSV_COLUMNS


def test_csv_logger_empty_cells_for_skipped_metrics(tmp_path: Path) -> None:
    """Verify epochs without validation or HD95 leave cells empty (not 0 or NaN)."""
    csv_file = tmp_path / "metrics.csv"
    logger = MetricsCSVLogger(csv_file)

    # Log training-only epoch (no validation)
    logger.log_epoch({
        "epoch": 1,
        "train_loss": 0.4521,
        "lr": 0.0002,
        "grad_norm_mean": 0.8123,
        "epoch_time_s": 12.5,
        "samples_seen": 20,
        "is_best": False,
    })

    with csv_file.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        row = next(reader)

    assert row["epoch"] == "1"
    assert row["train_loss"] == "0.452100"
    assert row["val_loss"] == ""
    assert row["val_dice_mean"] == ""
    assert row["val_hd95_mean"] == ""
    assert row["is_best"] == "False"


def test_csv_logger_is_best_flag(tmp_path: Path) -> None:
    """Verify is_best flag marks True/False appropriately."""
    csv_file = tmp_path / "metrics.csv"
    logger = MetricsCSVLogger(csv_file)

    logger.log_epoch({
        "epoch": 1,
        "train_loss": 0.5,
        "val_loss": 0.4,
        "val_dice_mean": 0.8,
        "is_best": True,
    })
    logger.log_epoch({
        "epoch": 2,
        "train_loss": 0.4,
        "val_loss": 0.45,
        "val_dice_mean": 0.78,
        "is_best": False,
    })

    with csv_file.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 2
    assert rows[0]["is_best"] == "True"
    assert rows[1]["is_best"] == "False"


def test_csv_logger_resume_truncation(tmp_path: Path) -> None:
    """Verify on resume that rows with epoch >= start_epoch are truncated to prevent duplicates."""
    csv_file = tmp_path / "metrics.csv"

    # Pre-populate CSV with 4 epochs
    logger1 = MetricsCSVLogger(csv_file)
    for ep in range(1, 5):
        logger1.log_epoch({"epoch": ep, "train_loss": 0.5 - 0.1 * ep, "is_best": ep == 1})

    with csv_file.open("r", encoding="utf-8") as f:
        assert len(list(csv.reader(f))) == 5  # 1 header + 4 rows

    # Resume at epoch 3 (so epochs 3 and 4 should be truncated)
    logger2 = MetricsCSVLogger(csv_file, start_epoch=3)
    logger2.log_epoch({"epoch": 3, "train_loss": 0.18, "is_best": True})
    logger2.log_epoch({"epoch": 4, "train_loss": 0.12, "is_best": True})

    with csv_file.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 4
    epochs = [int(r["epoch"]) for r in rows]
    assert epochs == [1, 2, 3, 4]
    assert rows[2]["train_loss"] == "0.180000"
    assert rows[3]["train_loss"] == "0.120000"
