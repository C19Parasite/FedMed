"""Flower NumPyClient implementing private hospital node training with ProtectionPipeline."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Mapping, Optional, Tuple
import flwr as fl
import numpy as np
import torch
import torch.nn as nn
from monai.data import DataLoader

from fedmed.models.params import get_parameters, set_parameters
from fedmed.training.losses import get_loss
from fedmed.utils.logging import get_logger
from src.model.metrics import compute_dice_score, evaluate_dataset
from src.privacy.hooks import ProtectionPipeline

logger = get_logger(__name__)


class FedMedClient(fl.client.NumPyClient):
    """Flower NumPyClient representing an isolated hospital node with integrated privacy pipelines.

    Strict Data Isolation:
      - Raw MRI scans, annotations, and patient IDs NEVER leave this process.
      - Only protected weight updates and aggregated scalars are transmitted.
    """

    def __init__(
        self,
        hospital_id: str,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: Mapping[str, Any],
        device: torch.device | str = "cpu",
        dropout_round: Optional[int] = None,
    ) -> None:
        self.hospital_id = hospital_id
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.device = torch.device(device)
        self.dropout_round = dropout_round
        self.model.to(self.device)

        self.criterion = get_loss(cfg)
        lr = float(cfg.get("training", {}).get("lr", 2e-4))
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)

        # Extensible privacy protection pipeline
        self.pipeline = ProtectionPipeline(cfg, hospital_id=hospital_id)
        self.initial_weights: Optional[List[np.ndarray]] = None

        logger.info(
            "Initialized FedMedClient [%s] on device=%s (%d train batches, %d val batches)",
            self.hospital_id,
            self.device,
            len(self.train_loader),
            len(self.val_loader),
        )

    def get_parameters(self, config: Dict[str, Any]) -> List[np.ndarray]:
        """Extract current model parameters."""
        return get_parameters(self.model)

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Update model weights."""
        set_parameters(self.model, parameters)

    def fit(
        self, parameters: List[np.ndarray], config: Dict[str, Any]
    ) -> Tuple[List[np.ndarray], int, Dict[str, Any]]:
        """Perform local training on private hospital data and apply privacy hooks."""
        server_round = config.get("server_round", 1)
        if self.dropout_round is not None and int(server_round) == int(self.dropout_round):
            logger.error("[%s] SIMULATING ABRUPT CRASH in round %s", self.hospital_id, server_round)
            raise RuntimeError(f"Simulated network dropout for {self.hospital_id} in round {server_round}")

        local_epochs = config.get("local_epochs", int(self.cfg.get("training", {}).get("epochs", 1)))

        # Cache initial weights before training
        model_params = get_parameters(self.model)
        if len(parameters) == len(model_params):
            self.set_parameters(parameters)
        elif self.pipeline.he_enabled and self.pipeline.hospital_context is not None and len(parameters) > 0:
            # Reconstruct averaged weights from previous round's aggregated ciphertext
            try:
                raw_chunks = [arr.tobytes() for arr in parameters]
                flat_w, shapes_and_types = flatten_tensors(model_params)
                total_n = int(config.get("total_n", 6))
                averaged_deltas, _ = decrypt_and_average(
                    raw_chunks, shapes_and_types, len(flat_w), total_n, self.pipeline.hospital_context
                )
                new_w = [w + d for w, d in zip(model_params, averaged_deltas)]
                set_parameters(self.model, new_w)
            except Exception as e:
                logger.warning("[%s] Could not decrypt incoming parameters: %s", self.hospital_id, e)

        self.initial_weights = [w.copy() for w in get_parameters(self.model)]

        self.model.train()
        total_loss = 0.0
        total_batches = 0

        for epoch in range(local_epochs):
            for batch in self.train_loader:
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                total_loss += float(loss.item())
                total_batches += 1

        avg_train_loss = total_loss / max(1, total_batches)
        trained_weights = get_parameters(self.model)
        num_examples = len(self.train_loader.dataset)

        # Run outbound ProtectionPipeline: compute_update -> clip -> noise -> encrypt -> send
        outbound_params, meta = self.pipeline.process_outbound(
            initial_weights=self.initial_weights,
            trained_weights=trained_weights,
            num_examples=num_examples,
        )

        metrics: Dict[str, Any] = {
            "hospital_id": self.hospital_id,
            "train_loss": round(avg_train_loss, 4),
            "round": int(server_round),
            "num_examples": int(num_examples),
        }
        if meta.get("he_applied"):
            metrics["he_applied"] = True

        logger.info(
            "[%s] Completed fit round %d: train_loss=%.4f (n=%d)",
            self.hospital_id,
            server_round,
            avg_train_loss,
            num_examples,
        )
        return outbound_params, num_examples, metrics

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict[str, Any]
    ) -> Tuple[float, int, Dict[str, Any]]:
        """Evaluate local model on hospital's private validation split."""
        model_params = get_parameters(self.model)
        if len(parameters) == len(model_params):
            self.set_parameters(parameters)

        val_loss, dice_metrics = evaluate_dataset(
            model=self.model,
            loader=self.val_loader,
            device=self.device,
            criterion=self.criterion,
        )

        num_examples = len(self.val_loader.dataset)
        metrics: Dict[str, Any] = {
            "hospital_id": self.hospital_id,
            "val_loss": round(val_loss, 4),
            "dice_mean": dice_metrics.get("dice_mean", 0.0),
            "dice_tc": dice_metrics.get("dice_tc", 0.0),
            "dice_wt": dice_metrics.get("dice_wt", 0.0),
            "dice_et": dice_metrics.get("dice_et", 0.0),
        }
        return val_loss, num_examples, metrics


__all__ = ["FedMedClient"]
