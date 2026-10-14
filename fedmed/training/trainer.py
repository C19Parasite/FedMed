"""Centralized baseline trainer and federated reusable training harness.

Provides:
- Trainer: Complete training, validation, and fitting pipeline designed for reuse by Flower clients.
- CentralizedTrainer: Backwards-compatible baseline trainer.
"""

from __future__ import annotations

import math
from pathlib import Path
import time
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from monai.inferers import sliding_window_inference

from fedmed.models.params import get_parameters, set_parameters
from fedmed.training.checkpoint import (
    load_checkpoint as load_ckpt_file,
    prune_checkpoints,
    save_checkpoint as save_ckpt_file,
)
from fedmed.training.csv_logger import MetricsCSVLogger
from fedmed.training.losses import build_loss, get_loss
from fedmed.training.metrics import SegmentationMetrics, evaluate_brats_dice
from fedmed.training.optim import build_optimizer, build_scheduler, get_lr
from fedmed.training.run import config_hash, create_run_dir
from fedmed.utils.logging import get_logger

logger = get_logger(__name__)


class Trainer:
    """Centralized training and evaluation engine for 3D brain tumor MRI segmentation.

    Designed for reuse across both standalone centralized execution and
    federated hospital client training rounds (Prompt 5).

    Args:
        model: PyTorch segmentation model.
        loss_fn: Loss function (e.g. DeepSupervisionLoss wrapping DiceCELoss).
        optimizer: PyTorch optimizer instance.
        scheduler: Optional learning rate scheduler.
        cfg: Validated experiment configuration mapping.
        device: Target execution device ('cuda' or 'cpu').
        logger: Optional Python logger.
        metrics: Optional SegmentationMetrics instance.
        run_dir: Optional root run directory.
        on_epoch_end: Optional list of callback functions called with epoch metric records.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: Optimizer,
        scheduler: LRScheduler | None,
        cfg: Mapping[str, Any],
        device: torch.device | str,
        logger: Any | None = None,
        metrics: SegmentationMetrics | None = None,
        run_dir: Path | str | None = None,
        on_epoch_end: Sequence[Callable[[dict[str, Any]], None]] | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.cfg = cfg
        self.logger = logger or get_logger(__name__)
        self.metrics = metrics or SegmentationMetrics(cfg)
        self.run_dir = Path(run_dir).resolve() if run_dir is not None else None
        self.on_epoch_end: list[Callable[[dict[str, Any]], None]] = (
            list(on_epoch_end) if on_epoch_end is not None else []
        )

        train_cfg = cfg.get("training", {})
        amp_cfg = train_cfg.get("amp", {})
        self.amp_enabled = bool(amp_cfg.get("enabled", False))
        dtype_str = str(amp_cfg.get("dtype", "float16")).lower()
        self.amp_dtype = torch.float16 if dtype_str == "float16" else torch.bfloat16

        # Auto-disable AMP with warning on CPU
        if self.amp_enabled and self.device.type != "cuda":
            self.logger.warning("AMP is enabled in config but device is CPU; disabling AMP.")
            self.amp_enabled = False

        # GradScaler is required only when AMP is enabled AND dtype is float16 on CUDA
        self.use_scaler = (
            self.amp_enabled
            and self.amp_dtype == torch.float16
            and self.device.type == "cuda"
        )
        if self.use_scaler:
            self.scaler = torch.amp.GradScaler(device="cuda", enabled=True)
        else:
            self.scaler = None

        self.logger.info(
            "Trainer initialized on device=%s (AMP=%s, GradScaler=%s)",
            self.device,
            self.amp_enabled,
            self.use_scaler,
        )

    def _autocast_context(self) -> Any:
        """Create device-appropriate autocast context manager."""
        dev_type = "cuda" if self.device.type == "cuda" else "cpu"
        return torch.amp.autocast(
            device_type=dev_type,
            dtype=self.amp_dtype,
            enabled=self.amp_enabled,
        )

    def train_one_epoch(
        self,
        loader: DataLoader | Iterable[Any],
        epoch: int = 1,
    ) -> dict[str, Any]:
        """Execute one training epoch over the given loader.

        Handles AMP autocast, GradScaler, gradient accumulation, and clipping.

        Args:
            loader: Training data loader yielding batches with 'image' and 'label'.
            epoch: Current epoch index (for logging).

        Returns:
            Dictionary containing train_loss, grad_norm_mean, lr, steps,
            skipped_steps, epoch_time_s, and samples_seen.
        """
        self.model.train()
        train_cfg = self.cfg.get("training", {})
        data_cfg = self.cfg.get("data", {})

        grad_accum_steps = max(1, int(train_cfg.get("grad_accum_steps", 1)))
        grad_clip_cfg = train_cfg.get("grad_clip", {})
        clip_enabled = bool(grad_clip_cfg.get("enabled", True))
        max_norm = float(grad_clip_cfg.get("max_norm", 1.0))
        max_steps = train_cfg.get("max_steps_per_epoch")
        pin_memory = bool(data_cfg.get("pin_memory", False))
        max_consecutive_skipped = int(train_cfg.get("max_consecutive_skipped", 5))

        start_time = time.time()
        total_loss = 0.0
        total_grad_norm = 0.0
        steps = 0
        skipped_steps = 0
        consecutive_skipped = 0
        samples_seen = 0
        accum_count = 0

        self.optimizer.zero_grad()

        num_batches = len(loader) if hasattr(loader, "__len__") else None

        for batch_idx, batch in enumerate(loader):
            if max_steps is not None and batch_idx >= max_steps:
                break

            if isinstance(batch, (list, tuple)):
                images, labels = batch[0], batch[1]
            else:
                images = batch["image"]
                labels = batch["label"]

            images = images.to(self.device, non_blocking=pin_memory)
            labels = labels.to(self.device, non_blocking=pin_memory)
            batch_size = images.shape[0]
            samples_seen += batch_size

            with self._autocast_context():
                logits = self.model(images)
                loss = self.loss_fn(logits, labels)
                loss_scaled = loss / grad_accum_steps

            loss_val = float(loss.item())

            # Non-finite loss check
            if math.isnan(loss_val) or math.isinf(loss_val):
                self.logger.warning(
                    "Non-finite loss %s at epoch %d, batch %d; skipping step",
                    loss_val,
                    epoch,
                    batch_idx,
                )
                self.optimizer.zero_grad()
                skipped_steps += 1
                consecutive_skipped += 1
                accum_count = 0
                if consecutive_skipped >= max_consecutive_skipped:
                    raise RuntimeError(
                        f"Training aborted: {consecutive_skipped} consecutive steps encountered non-finite loss."
                    )
                continue

            if self.scaler is not None:
                self.scaler.scale(loss_scaled).backward()
            else:
                loss_scaled.backward()

            accum_count += 1
            is_last_batch = (batch_idx + 1 == num_batches) if num_batches else False
            if max_steps is not None and batch_idx + 1 == max_steps:
                is_last_batch = True

            # Step optimizer upon reaching gradient accumulation steps or final batch
            if accum_count == grad_accum_steps or is_last_batch:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)

                # Pre-clip global gradient norm
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=max_norm if clip_enabled else float("inf"),
                )
                if isinstance(grad_norm, torch.Tensor):
                    grad_norm_val = float(grad_norm.item())
                else:
                    grad_norm_val = float(grad_norm)

                # Non-finite gradient norm check
                if math.isnan(grad_norm_val) or math.isinf(grad_norm_val):
                    self.logger.warning(
                        "Non-finite grad norm %s at epoch %d, batch %d; skipping optimizer step",
                        grad_norm_val,
                        epoch,
                        batch_idx,
                    )
                    self.optimizer.zero_grad()
                    skipped_steps += 1
                    consecutive_skipped += 1
                    accum_count = 0
                    if consecutive_skipped >= max_consecutive_skipped:
                        raise RuntimeError(
                            f"Training aborted: {consecutive_skipped} consecutive steps encountered non-finite grad norm."
                        )
                    continue

                consecutive_skipped = 0

                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad()
                total_loss += loss_val
                total_grad_norm += grad_norm_val
                steps += 1
                accum_count = 0

        epoch_time = time.time() - start_time
        mean_loss = total_loss / max(1, steps)
        mean_grad_norm = total_grad_norm / max(1, steps)
        current_lr = get_lr(self.optimizer)

        return {
            "train_loss": mean_loss,
            "grad_norm_mean": mean_grad_norm,
            "lr": current_lr,
            "steps": steps,
            "skipped_steps": skipped_steps,
            "epoch_time_s": epoch_time,
            "samples_seen": samples_seen,
        }

    def validate(
        self,
        loader: DataLoader | Iterable[Any],
        include_hd95: bool = False,
    ) -> dict[str, Any]:
        """Evaluate model across validation dataset with sliding window inference.

        Args:
            loader: Validation DataLoader.
            include_hd95: Whether to calculate Hausdorff 95 distance.

        Returns:
            Dictionary containing Dice metrics, optional HD95 metrics,
            val_loss, n_volumes, and inference_time_s.
        """
        was_training = self.model.training
        self.model.eval()

        train_cfg = self.cfg.get("training", {})
        sw_cfg = train_cfg.get("sliding_window", {})
        data_cfg = self.cfg.get("data", {})

        roi_size = sw_cfg.get("roi_size", data_cfg.get("roi_size", [96, 96, 96]))
        sw_batch_size = int(sw_cfg.get("sw_batch_size", 1))
        overlap = float(sw_cfg.get("overlap", 0.25))
        mode = str(sw_cfg.get("mode", "gaussian")).lower()
        progress = bool(sw_cfg.get("progress", False))
        pin_memory = bool(data_cfg.get("pin_memory", False))

        self.metrics.reset()
        start_time = time.time()
        total_loss = 0.0
        n_volumes = 0

        with torch.no_grad():
            for batch in loader:
                if isinstance(batch, (list, tuple)):
                    images, labels = batch[0], batch[1]
                else:
                    images = batch["image"]
                    labels = batch["label"]

                images = images.to(self.device, non_blocking=pin_memory)
                labels = labels.to(self.device, non_blocking=pin_memory)
                b_size = images.shape[0]

                with self._autocast_context():
                    logits = sliding_window_inference(
                        inputs=images,
                        roi_size=roi_size,
                        sw_batch_size=sw_batch_size,
                        predictor=self.model,
                        overlap=overlap,
                        mode=mode,
                        progress=progress,
                    )
                    loss = self.loss_fn(logits, labels)

                total_loss += float(loss.item()) * b_size
                n_volumes += b_size
                self.metrics.update(logits, labels)

        results = self.metrics.compute(include_hd95=include_hd95)
        results["val_loss"] = total_loss / max(1, n_volumes)
        results["n_volumes"] = n_volumes
        results["inference_time_s"] = time.time() - start_time

        # Restore original training mode
        if was_training:
            self.model.train()

        return results

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        start_epoch: int = 0,
    ) -> dict[str, Any]:
        """Execute full training schedule with early stopping and checkpoint persistence.

        Catches KeyboardInterrupt to save state safely.

        Args:
            train_loader: DataLoader for training cases.
            val_loader: DataLoader for validation cases.
            start_epoch: Epoch to start/resume from (default: 0).

        Returns:
            Dictionary containing best_epoch, best_metrics, and full training history.
        """
        train_cfg = self.cfg.get("training", {})
        metrics_cfg = self.cfg.get("metrics", {})
        total_epochs = int(train_cfg.get("epochs", 5))
        val_every_n = max(1, int(train_cfg.get("val_every_n_epochs", 1)))

        hd95_cfg = metrics_cfg.get("hd95", {})
        hd95_enabled = bool(hd95_cfg.get("enabled", False))
        hd95_every = int(hd95_cfg.get("every_n_epochs", 5))

        es_cfg = train_cfg.get("early_stopping", {})
        es_enabled = bool(es_cfg.get("enabled", True))
        es_patience = int(es_cfg.get("patience", 10))
        es_min_delta = float(es_cfg.get("min_delta", 0.0001))
        es_monitor = str(es_cfg.get("monitor", "val_dice_mean"))
        es_mode = str(es_cfg.get("mode", "max")).lower()

        ckpt_cfg = train_cfg.get("checkpoint", {})
        save_every_n = max(1, int(ckpt_cfg.get("save_every_n_epochs", 1)))
        keep_last_k = max(1, int(ckpt_cfg.get("keep_last_k", 3)))

        # Determine run directory and checkpoint folder
        if self.run_dir is None:
            self.run_dir = create_run_dir(self.cfg)
        ckpt_dir = self.run_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        results_dir = self.run_dir / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

        csv_path = results_dir / "metrics.csv"
        tb_dir = self.run_dir / "tensorboard" if train_cfg.get("logging", {}).get("tensorboard", True) else None
        first_new_epoch = (start_epoch + 1) if start_epoch > 0 else 0
        csv_logger = MetricsCSVLogger(csv_path=csv_path, start_epoch=first_new_epoch, tensorboard_dir=tb_dir)

        cfg_h = config_hash(self.cfg)
        best_monitored_val = -float("inf") if es_mode == "max" else float("inf")
        best_epoch = start_epoch
        best_metrics: dict[str, Any] = {}
        history: list[dict[str, Any]] = []
        validation_rounds_no_improve = 0

        self.logger.info(
            "Beginning fit schedule from epoch %d to %d (monitoring=%s, mode=%s)",
            start_epoch + 1,
            total_epochs,
            es_monitor,
            es_mode,
        )

        try:
            for epoch in range(start_epoch + 1, total_epochs + 1):
                train_res = self.train_one_epoch(train_loader, epoch=epoch)

                # Validation decision
                is_last_epoch = (epoch == total_epochs)
                do_val = (epoch % val_every_n == 0) or is_last_epoch
                do_hd95 = (
                    do_val
                    and hd95_enabled
                    and ((epoch % hd95_every == 0) or is_last_epoch)
                )

                val_res: dict[str, Any] = {}
                is_best = False

                if do_val:
                    val_res = self.validate(val_loader, include_hd95=do_hd95)
                    monitored_val = float(val_res.get(es_monitor, 0.0))

                    if es_mode == "max":
                        improved = monitored_val >= (best_monitored_val + es_min_delta)
                    else:
                        improved = monitored_val <= (best_monitored_val - es_min_delta)

                    if improved:
                        is_best = True
                        best_monitored_val = monitored_val
                        best_epoch = epoch
                        best_metrics = dict(val_res)
                        validation_rounds_no_improve = 0

                        # Save deployable best.pt (weights and metadata only)
                        save_ckpt_file(
                            path=ckpt_dir / "best.pt",
                            model=self.model,
                            epoch=epoch,
                            best_metric=best_monitored_val,
                            cfg_hash=cfg_h,
                            is_best=True,
                        )
                        self.logger.info(
                            "Epoch %d: New best %s = %.4f -> saved %s",
                            epoch,
                            es_monitor,
                            best_monitored_val,
                            ckpt_dir / "best.pt",
                        )
                    else:
                        validation_rounds_no_improve += 1

                # Step scheduler after training epoch
                if self.scheduler is not None:
                    self.scheduler.step()

                # Assemble row for CSV and TensorBoard
                row = {
                    "epoch": epoch,
                    "train_loss": train_res["train_loss"],
                    "val_loss": val_res.get("val_loss"),
                    "val_dice_tc": val_res.get("dice_tc"),
                    "val_dice_wt": val_res.get("dice_wt"),
                    "val_dice_et": val_res.get("dice_et"),
                    "val_dice_mean": val_res.get("dice_mean"),
                    "val_hd95_tc": val_res.get("hd95_tc"),
                    "val_hd95_wt": val_res.get("hd95_wt"),
                    "val_hd95_et": val_res.get("hd95_et"),
                    "val_hd95_mean": val_res.get("hd95_mean"),
                    "lr": train_res["lr"],
                    "grad_norm_mean": train_res["grad_norm_mean"],
                    "epoch_time_s": train_res["epoch_time_s"],
                    "samples_seen": train_res["samples_seen"],
                    "is_best": is_best,
                }
                csv_logger.log_epoch(row)
                history.append(row)

                # Save latest.pt every epoch
                save_ckpt_file(
                    path=ckpt_dir / "latest.pt",
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    scaler=self.scaler,
                    epoch=epoch,
                    best_metric=best_monitored_val,
                    history=history,
                    cfg_hash=cfg_h,
                    is_best=False,
                )

                # Periodic checkpointing and pruning
                if epoch % save_every_n == 0:
                    save_ckpt_file(
                        path=ckpt_dir / f"epoch_{epoch}.pt",
                        model=self.model,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        scaler=self.scaler,
                        epoch=epoch,
                        best_metric=best_monitored_val,
                        history=history,
                        cfg_hash=cfg_h,
                        is_best=False,
                    )
                    prune_checkpoints(ckpt_dir, keep_last_k=keep_last_k)

                # Invoke callbacks
                for callback in self.on_epoch_end:
                    try:
                        callback(row)
                    except Exception as exc:
                        self.logger.warning("on_epoch_end callback failed: %s", exc)

                # Early stopping check (counted in validation rounds)
                if es_enabled and do_val and validation_rounds_no_improve >= es_patience:
                    self.logger.info(
                        "Early stopping triggered after %d validation rounds without improvement.",
                        es_patience,
                    )
                    break

        except KeyboardInterrupt:
            self.logger.warning("Training interrupted by user! Persisting state to latest.pt...")
            save_ckpt_file(
                path=ckpt_dir / "latest.pt",
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.scaler,
                epoch=epoch,
                best_metric=best_monitored_val,
                history=history,
                cfg_hash=cfg_h,
                is_best=False,
            )
            csv_logger.close()
            resume_cmd = (
                f"python scripts/train_baseline.py --config "
                f"{self.cfg.get('_config_path', 'configs/baseline.yaml')} --resume"
            )
            self.logger.info("Training interrupted. Resume with:\n  %s", resume_cmd)
            raise

        csv_logger.close()
        return {
            "best_epoch": best_epoch,
            "best_metric": best_monitored_val,
            "best_metrics": best_metrics,
            "history": history,
        }

    # Federated Learning reuse helpers (Prompt 5)
    def get_model_parameters(self) -> list[np.ndarray]:
        """Extract model parameters as a list of NumPy arrays for federated aggregation."""
        return get_parameters(self.model)

    def set_model_parameters(self, parameters: list[np.ndarray]) -> None:
        """Update model parameters from federated aggregation."""
        set_parameters(self.model, parameters)

    def state_dict_for_checkpoint(self) -> dict[str, Any]:
        """Extract full serializable state dictionary."""
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict() if self.optimizer else None,
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "scaler": self.scaler.state_dict() if self.scaler else None,
        }

    def load_checkpoint(self, path: str | Path) -> dict[str, Any]:
        """Load state into model, optimizer, scheduler, and scaler."""
        return load_ckpt_file(
            path=path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            expected_cfg_hash=config_hash(self.cfg),
            map_location=self.device,
        )


class CentralizedTrainer:
    """Backwards-compatible wrapper for legacy tests and scripts.

    Maintains the original CentralizedTrainer interface.
    """

    def __init__(
        self,
        model: nn.Module,
        cfg: Mapping[str, Any],
        device: Optional[torch.device | str] = None,
    ) -> None:
        self.cfg = cfg
        train_cfg = cfg.get("training", {})

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
        """Run one training epoch over all batches."""
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

        return total_loss / max(num_batches, 1)

    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Evaluate model across validation set."""
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
        """Execute legacy training and evaluation schedule."""
        logger.info("Beginning training for %s epochs", epochs)
        history = []

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            logger.info("Epoch %s/%s - Train Loss: %.4f", epoch, epochs, train_loss)

            if epoch % self.val_interval == 0:
                val_metrics = self.evaluate(val_loader)
                mean_dice = val_metrics["dice_mean"]
                if mean_dice > self.best_dice:
                    self.best_dice = mean_dice
                    self.save_checkpoint("best_metric_model.pt", epoch, mean_dice)

                history.append({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    **val_metrics,
                })

        self.save_checkpoint("latest_model.pt", epochs, self.best_dice)
        return {"best_dice": self.best_dice, "history": history}

    def save_checkpoint(self, filename: str, epoch: int, metric: float) -> Path:
        """Persist model weights and metadata to disk."""
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
        return path
