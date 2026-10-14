"""Metrics CSV logger and optional TensorBoard logger for experiment tracking.

Header format is fixed and parsed by downstream tools and the Week 3 React dashboard:
epoch,train_loss,val_loss,val_dice_tc,val_dice_wt,val_dice_et,val_dice_mean,val_hd95_tc,val_hd95_wt,val_hd95_et,val_hd95_mean,lr,grad_norm_mean,epoch_time_s,samples_seen,is_best
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping, Sequence
import torch.nn as nn

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None  # type: ignore

from fedmed.utils.logging import get_logger

logger = get_logger(__name__)

# Fixed schema for results/metrics.csv - NEVER reorder columns
METRICS_CSV_COLUMNS: list[str] = [
    "epoch",
    "train_loss",
    "val_loss",
    "val_dice_tc",
    "val_dice_wt",
    "val_dice_et",
    "val_dice_mean",
    "val_hd95_tc",
    "val_hd95_wt",
    "val_hd95_et",
    "val_hd95_mean",
    "lr",
    "grad_norm_mean",
    "epoch_time_s",
    "samples_seen",
    "is_best",
]


class MetricsCSVLogger:
    """Logger tracking per-epoch metrics to a structured CSV file and TensorBoard.

    Args:
        csv_path: Destination path for metrics.csv.
        start_epoch: Epoch to resume from; if > 0, truncates any existing rows with epoch >= start_epoch.
        tensorboard_dir: Optional directory for TensorBoard event logs.
    """

    def __init__(
        self,
        csv_path: str | Path,
        start_epoch: int = 0,
        tensorboard_dir: str | Path | None = None,
    ) -> None:
        self.csv_path = Path(csv_path).resolve()
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.writer: Any | None = None

        if tensorboard_dir and SummaryWriter is not None:
            tb_dir = Path(tensorboard_dir).resolve()
            tb_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir=str(tb_dir))

        self._init_csv(start_epoch=start_epoch)

    def _init_csv(self, start_epoch: int) -> None:
        """Initialize CSV file, handling header creation or resume truncation."""
        if not self.csv_path.is_file() or self.csv_path.stat().st_size == 0:
            with self.csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(METRICS_CSV_COLUMNS)
            return

        # Resume mode: truncate rows with epoch >= start_epoch
        if start_epoch > 0:
            rows_to_keep: list[list[str]] = []
            with self.csv_path.open("r", newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if header:
                    rows_to_keep.append(header)
                epoch_idx = 0
                for row in reader:
                    if not row:
                        continue
                    try:
                        row_epoch = int(row[epoch_idx])
                        if row_epoch < start_epoch:
                            rows_to_keep.append(row)
                    except (ValueError, IndexError):
                        pass

            with self.csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerows(rows_to_keep)

    def log_epoch(self, metrics: Mapping[str, Any]) -> None:
        """Append one epoch's metric dictionary to the CSV file and TensorBoard.

        Unspecified validation or HD95 metrics are written as empty strings (not 0 or NaN).

        Args:
            metrics: Mapping containing metrics corresponding to METRICS_CSV_COLUMNS.
        """
        row: list[str] = []
        for col in METRICS_CSV_COLUMNS:
            val = metrics.get(col)
            if val is None or val == "":
                row.append("")
            elif col == "is_best":
                row.append(str(bool(val)))
            elif col == "epoch" or col == "samples_seen":
                row.append(str(int(val)))
            elif isinstance(val, (int, float)):
                # Format float cleanly without scientific notation unless very small
                if abs(val) < 1e-4 and val != 0.0:
                    row.append(f"{val:.6e}")
                else:
                    row.append(f"{val:.6f}")
            else:
                row.append(str(val))

        # Atomic append and flush
        with self.csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row)
            f.flush()

        # Log to TensorBoard if enabled
        if self.writer is not None and "epoch" in metrics:
            epoch = int(metrics["epoch"])
            for key, val in metrics.items():
                if val is None or val == "" or key in ("epoch", "is_best"):
                    continue
                try:
                    num_val = float(val)
                    if key.startswith("train_"):
                        tag = f"train/{key[len('train_'):]}"
                    elif key.startswith("val_"):
                        tag = f"val/{key[len('val_'):]}"
                    else:
                        tag = f"metrics/{key}"
                    self.writer.add_scalar(tag, num_val, global_step=epoch)
                except (ValueError, TypeError):
                    pass
            self.writer.flush()

    def log_model_histograms(self, model: nn.Module, epoch: int) -> None:
        """Log parameter weight histograms to TensorBoard."""
        if self.writer is None:
            return
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.writer.add_histogram(f"weights/{name}", param.detach().cpu(), global_step=epoch)
        self.writer.flush()

    def close(self) -> None:
        """Flush and close TensorBoard writer."""
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
            self.writer = None
