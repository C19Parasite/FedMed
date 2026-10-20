"""Flower client representing an isolated hospital node for cross-silo FL."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Tuple
import flwr as fl
import numpy as np
import torch
import torch.nn as nn
from monai.data import DataLoader

from fedmed.models.params import get_parameters as extract_model_parameters, set_parameters as apply_model_parameters
from fedmed.training.losses import get_loss
from fedmed.training.metrics import evaluate_brats_dice
from fedmed.utils.logging import get_logger

logger = get_logger(__name__)


class HospitalClient(fl.client.NumPyClient):
    """Flower NumPyClient implementing local hospital model training and evaluation.

    Preserves strict data isolation: only model weight updates and aggregated
    validation metrics cross the network boundary; raw scans stay in the silo.

    Args:
        hospital_id: Identifier for the hospital node (e.g. 'hospital_1').
        model: PyTorch segmentation network (InstanceNorm UNet3D).
        train_loader: Private training DataLoader for this hospital silo.
        val_loader: Private validation DataLoader for this hospital silo.
        cfg: Configuration mapping.
        device: Execution device ('cuda' or 'cpu').
    """

    def __init__(
        self,
        hospital_id: str,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: Mapping[str, Any],
        device: Optional[torch.device | str] = None,
    ) -> None:
        self.hospital_id = hospital_id
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg

        if device is None:
            configured_dev = cfg.get("training", {}).get("device", "cpu")
            self.device = torch.device(
                "cuda" if (configured_dev == "cuda" and torch.cuda.is_available()) else "cpu"
            )
        else:
            self.device = torch.device(device)

        self.model.to(self.device)
        self.criterion = get_loss(cfg)

        lr = float(cfg.get("training", {}).get("lr", 2e-4))
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)

        logger.info(
            "Initialized HospitalClient [%s] on device=%s with %s train batches",
            self.hospital_id,
            self.device,
            len(self.train_loader),
        )

    def get_parameters(self, config: Dict[str, Any]) -> List[np.ndarray]:
        """Extract model parameters as a list of NumPy arrays.

        Args:
            config: Server-provided configuration dictionary.

        Returns:
            List of model weights in NumPy format.
        """
        return extract_model_parameters(self.model)

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Update local model weights with received parameters.

        Args:
            parameters: List of weight tensors in NumPy format.
        """
        apply_model_parameters(self.model, parameters)

    def _extract_batch_tensors(self, batch: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract image and label tensors robustly from collated batch, list, or tuple."""
        if isinstance(batch, dict):
            return batch["image"].to(self.device), batch["label"].to(self.device)
        if isinstance(batch, (tuple, list)):
            if len(batch) == 2 and isinstance(batch[0], torch.Tensor) and isinstance(batch[1], torch.Tensor):
                return batch[0].to(self.device), batch[1].to(self.device)
            # In case of uncollated list of dict samples
            if all(isinstance(item, dict) for item in batch):
                images = torch.stack([item["image"] for item in batch]).to(self.device)
                labels = torch.stack([item["label"] for item in batch]).to(self.device)
                return images, labels
        raise TypeError(f"Unexpected batch type: {type(batch)}")

    def fit(
        self, parameters: List[np.ndarray], config: Dict[str, Any]
    ) -> Tuple[List[np.ndarray], int, Dict[str, Any]]:
        """Train local model on hospital data for specified local epochs.

        Args:
            parameters: Global weights received from central server.
            config: Training instructions from server.

        Returns:
            Tuple of (updated weights, num_samples, training metrics).
        """
        self.set_parameters(parameters)
        self.model.train()

        local_epochs = int(config.get("local_epochs", 1))
        total_loss = 0.0
        num_examples = 0

        for _ in range(local_epochs):
            for batch in self.train_loader:
                images, labels = self._extract_batch_tensors(batch)

                self.optimizer.zero_grad(set_to_none=True)
                logits = self.model(images)
                loss = self.criterion(logits, labels)
                loss.backward()
                self.optimizer.step()

                batch_size = images.shape[0]
                total_loss += loss.item() * batch_size
                num_examples += batch_size

        avg_loss = total_loss / max(num_examples, 1)
        logger.info(
            "Node [%s] completed fit: examples=%s, loss=%.4f",
            self.hospital_id,
            num_examples,
            avg_loss,
        )
        return self.get_parameters(config={}), num_examples, {"train_loss": avg_loss}

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict[str, Any]
    ) -> Tuple[float, int, Dict[str, Any]]:
        """Evaluate received global model weights on local validation set.

        Args:
            parameters: Global weights received from central server.
            config: Evaluation instructions from server.

        Returns:
            Tuple of (val_loss, num_samples, evaluation metrics dict).
        """
        self.set_parameters(parameters)
        self.model.eval()

        total_loss = 0.0
        total_metrics = {"dice_tc": 0.0, "dice_wt": 0.0, "dice_et": 0.0, "dice_mean": 0.0}
        num_examples = 0
        num_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                images, labels = self._extract_batch_tensors(batch)

                logits = self.model(images)
                loss = self.criterion(logits, labels)

                batch_metrics = evaluate_brats_dice(logits, labels)
                for k, v in batch_metrics.items():
                    total_metrics[k] += v

                batch_size = images.shape[0]
                total_loss += loss.item() * batch_size
                num_examples += batch_size
                num_batches += 1

        avg_loss = total_loss / max(num_examples, 1)
        count = max(num_batches, 1)
        mean_metrics = {k: v / count for k, v in total_metrics.items()}
        mean_metrics["val_loss"] = avg_loss

        logger.info(
            "Node [%s] eval: loss=%.4f, mean_dice=%.4f",
            self.hospital_id,
            avg_loss,
            mean_metrics["dice_mean"],
        )
        return avg_loss, num_examples, mean_metrics
