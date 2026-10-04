"""Centralized baseline trainer for 3D brain tumor MRI segmentation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from fedmed.training.losses import get_loss
from fedmed.training.metrics import evaluate_brats_dice
from fedmed.utils.logging import get_logger

logger = get_logger(__name__)


class CentralizedTrainer:
    """Trainer orchestrating 3D U-Net baseline optimization and validation.

    Args:
        model: PyTorch segmentation model.
        cfg: Validated experiment configuration mapping.
        device: Target execution device ('cuda' or 'cpu').
    """

    def __init__(
        self,
        model: nn.Module,
        cfg: Mapping[str, Any],
        device: Optional[torch.device | str] = None,
    ) -> None:
        self.cfg = cfg
        train_cfg = cfg["training"]

        if device is None:
            configured_dev = train_cfg.get("device", "cuda")
            self.device = torch.device(
                "cuda" if (configured_dev == "cuda" and torch.cuda.is_available()) else "cpu"
            )
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)
        self.criterion = get_loss(cfg)

        lr = float(train_cfg.get("lr", 2e-4))
        weight_decay = float(train_cfg.get("weight_decay", 1e-5))
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )

        self.save_dir = Path(train_cfg.get("save_dir", "checkpoints"))
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.val_interval = int(train_cfg.get("val_interval", 1))
        self.best_dice = -1.0

        logger.info(
            "Initialized CentralizedTrainer on device=%s with lr=%s",
            self.device,
            lr,
        )

    def train_epoch(self, train_loader: DataLoader) -> float:
        """Run one training epoch over all batches.

        Args:
            train_loader: Training DataLoader.

        Returns:
            Average training loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(images)
            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss

    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Evaluate model across validation set and compute mean BraTS metrics.

        Args:
            val_loader: Validation DataLoader.

        Returns:
            Dictionary of validation metrics.
        """
        self.model.eval()
        total_metrics: Dict[str, float] = {
            "dice_tc": 0.0,
            "dice_wt": 0.0,
            "dice_et": 0.0,
            "dice_mean": 0.0,
            "val_loss": 0.0,
        }
        num_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)

                logits = self.model(images)
                loss = self.criterion(logits, labels)

                batch_dice = evaluate_brats_dice(logits, labels)
                for k, v in batch_dice.items():
                    total_metrics[k] += v
                total_metrics["val_loss"] += loss.item()
                num_batches += 1

        count = max(num_batches, 1)
        return {k: v / count for k, v in total_metrics.items()}

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
    ) -> Dict[str, Any]:
        """Execute full training and evaluation schedule.

        Args:
            train_loader: DataLoader for training cases.
            val_loader: DataLoader for validation cases.
            epochs: Total training epochs.

        Returns:
            Summary dictionary of training results and best metrics.
        """
        logger.info("Beginning training for %s epochs", epochs)
        history = []

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            logger.info("Epoch %s/%s - Train Loss: %.4f", epoch, epochs, train_loss)

            if epoch % self.val_interval == 0:
                val_metrics = self.evaluate(val_loader)
                mean_dice = val_metrics["dice_mean"]
                logger.info(
                    "Epoch %s/%s - Val Loss: %.4f, Mean Dice: %.4f (TC: %.4f, WT: %.4f, ET: %.4f)",
                    epoch,
                    epochs,
                    val_metrics["val_loss"],
                    mean_dice,
                    val_metrics["dice_tc"],
                    val_metrics["dice_wt"],
                    val_metrics["dice_et"],
                )

                if mean_dice > self.best_dice:
                    self.best_dice = mean_dice
                    self.save_checkpoint("best_metric_model.pt", epoch, mean_dice)

                history.append({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    **val_metrics,
                })

        # Save final state
        self.save_checkpoint("latest_model.pt", epochs, self.best_dice)
        logger.info("Training complete. Best Validation Mean Dice: %.4f", self.best_dice)
        return {"best_dice": self.best_dice, "history": history}

    def save_checkpoint(self, filename: str, epoch: int, metric: float) -> Path:
        """Persist model weights and metadata to disk.

        Args:
            filename: Destination filename.
            epoch: Current training epoch.
            metric: Best metric value achieved.

        Returns:
            Path to saved checkpoint.
        """
        path = self.save_dir / filename
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "metric": metric,
            },
            path,
        )
        logger.info("Saved checkpoint %s (epoch=%s, metric=%.4f)", filename, epoch, metric)
        return path
