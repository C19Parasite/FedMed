"""Custom Flower Strategy with BraTS Centralized Evaluation, Dropout Resilience, and CKKS HE Aggregation."""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union
import flwr as fl
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Metrics,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
import numpy as np
import torch
import torch.nn as nn
from monai.data import DataLoader

from fedmed.models.params import get_parameters, set_parameters
from fedmed.training.losses import get_loss
from fedmed.utils.logging import get_logger
from src.metrics.publisher import publish_round_metrics
from src.model.metrics import evaluate_dataset
from src.privacy.dp import RenyiDPAccountant
from src.privacy.he import homomorphic_aggregate_chunks, load_server_context

logger = get_logger(__name__)


class FedMedStrategy(fl.server.strategy.FedAvg):
    """Federated Averaging strategy supporting BraTS centralized evaluation, node dropout resilience, and CKKS HE."""

    def __init__(
        self,
        model: nn.Module,
        global_val_loader: DataLoader,
        cfg: Mapping[str, Any],
        criterion: Optional[nn.Module] = None,
        device: torch.device | str = "cpu",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.model = model
        self.global_val_loader = global_val_loader
        self.cfg = cfg
        self.device = torch.device(device)
        self.criterion = criterion or get_loss(cfg)

        self.best_dice = -1.0
        self.round_start_time = time.perf_counter()

        self.total_clients = len(cfg.get("hospitals", [1, 2, 3]))
        self.dp_enabled = bool(cfg.get("dp", {}).get("enabled", False))
        self.he_enabled = bool(cfg.get("he", {}).get("enabled", False))
        self.tls_enabled = bool(cfg.get("tls", {}).get("enabled", False))

        self.noise_multiplier = float(cfg.get("dp", {}).get("noise_multiplier", 1.0))
        self.delta = float(cfg.get("dp", {}).get("delta", 1e-5))
        self.dp_accountant = (
            RenyiDPAccountant(noise_multiplier=self.noise_multiplier, delta=self.delta)
            if self.dp_enabled
            else None
        )

        self.server_he_context = None
        if self.he_enabled:
            ctx_path = cfg.get("he", {}).get("public_context", "keys/server_context.tenseal")
            try:
                self.server_he_context = load_server_context(ctx_path)
                logger.info("Server loaded public CKKS context from %s (cannot decrypt)", ctx_path)
            except Exception as e:
                logger.warning("Could not load server public HE context: %s", e)

        # Store latest client training metrics
        self.latest_hospital_metrics: Dict[str, Dict[str, float]] = {}

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: fl.server.client_manager.ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the upcoming fit round."""
        self.round_start_time = time.perf_counter()
        config: Dict[str, Scalar] = {
            "server_round": server_round,
            "local_epochs": int(self.cfg.get("training", {}).get("epochs", 1)),
            "lr": float(self.cfg.get("training", {}).get("lr", 2e-4)),
            "he_enabled": self.he_enabled,
            "dp_enabled": self.dp_enabled,
        }
        fit_ins = FitIns(parameters, config)

        if client_manager.num_available() < self.min_available_clients:
            logger.info(
                "Waiting for %d client(s) to connect (currently available: %d)...",
                self.min_available_clients,
                client_manager.num_available(),
            )
            client_manager.wait_for(num_clients=self.min_available_clients, timeout=30)

        sample_size, min_num_clients = self.num_fit_clients(client_manager.num_available())
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )
        return [(client, fit_ins) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate client updates, resilient to client failures, with optional CKKS ciphertext summation."""
        if not results:
            logger.warning("Round %d: No client results received!", server_round)
            return None, {}

        num_responding = len(results)
        if failures:
            logger.warning(
                "Round %d: %d client failure(s) detected. Resiliently renormalizing weights over %d responding clients.",
                server_round,
                len(failures),
                num_responding,
            )

        # Record hospital metrics reported by clients
        self.latest_hospital_metrics.clear()
        total_bytes_received = 0

        for client_proxy, fit_res in results:
            client_id = fit_res.metrics.get("hospital_id", client_proxy.cid)
            self.latest_hospital_metrics[str(client_id)] = {
                "train_loss": float(fit_res.metrics.get("train_loss", 0.0)),
                "num_examples": float(fit_res.num_examples),
            }
            # Approximate wire bytes from parameters
            for tensor in fit_res.parameters.tensors:
                total_bytes_received += len(tensor)

        # 1. Homomorphic Encryption Ciphertext Summation (Phase 6)
        if self.he_enabled and self.server_he_context is not None:
            logger.info("Round %d: Aggregating on ciphertext using TenSEAL homomorphic addition...", server_round)
            client_chunks_list = []
            total_n = 0
            for _, fit_res in results:
                ndarrays = parameters_to_ndarrays(fit_res.parameters)
                chunks = [arr.tobytes() for arr in ndarrays]
                client_chunks_list.append(chunks)
                total_n += fit_res.num_examples

            agg_chunks, agg_time = homomorphic_aggregate_chunks(client_chunks_list, self.server_he_context)
            logger.info("Ciphertext aggregation complete in %.3fs for %d chunks (total_n=%d)", agg_time, len(agg_chunks), total_n)

            # Package aggregated chunks as uint8 ndarrays
            agg_ndarrays = [np.frombuffer(c, dtype=np.uint8) for c in agg_chunks]
            aggregated_params = ndarrays_to_parameters(agg_ndarrays)

            metrics = {
                "participating_clients": num_responding,
                "total_n": total_n,
                "he_applied": True,
            }
            return aggregated_params, metrics

        # 2. Standard Plaintext Weighted FedAvg (Phase 3 & Phase 5)
        total_examples = sum(fit_res.num_examples for _, fit_res in results)
        if total_examples == 0:
            return None, {}

        # Weighted average of model parameters
        weighted_weights: List[np.ndarray] = []
        for client_idx, (_, fit_res) in enumerate(results):
            client_weights = parameters_to_ndarrays(fit_res.parameters)
            weight_factor = fit_res.num_examples / total_examples
            if client_idx == 0:
                weighted_weights = [w * weight_factor for w in client_weights]
            else:
                for idx, w in enumerate(client_weights):
                    weighted_weights[idx] += w * weight_factor

        aggregated_params = ndarrays_to_parameters(weighted_weights)
        metrics = {"participating_clients": num_responding, "total_examples": total_examples}
        return aggregated_params, metrics

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Centralized evaluation on server's private global validation holdout."""
        if server_round == 0:
            return None

        # If HE was enabled, the server parameters are ciphertext chunks;
        # in that setting, client-side decryption occurs on the nodes, but for server-side
        # validation we evaluate the model if plaintext weights are available.
        if not (self.he_enabled and self.server_he_context is not None):
            weights = parameters_to_ndarrays(parameters)
            set_parameters(self.model, weights)

        val_loss, dice_metrics = evaluate_dataset(
            model=self.model,
            loader=self.global_val_loader,
            device=self.device,
            criterion=self.criterion,
        )

        global_val_dice = dice_metrics.get("dice_mean", 0.0)
        round_duration = time.perf_counter() - self.round_start_time

        # Save checkpoint if new best
        checkpoint_dir = self.cfg.get("server", {}).get("checkpoint_dir", "checkpoints/federated")
        import os
        os.makedirs(checkpoint_dir, exist_ok=True)
        ckpt_path = f"{checkpoint_dir}/round_{server_round}.pt"
        torch.save(
            {
                "round": server_round,
                "state_dict": self.model.state_dict(),
                "val_dice": global_val_dice,
                "val_loss": val_loss,
            },
            ckpt_path,
        )

        if global_val_dice > self.best_dice:
            self.best_dice = global_val_dice
            torch.save(
                {
                    "round": server_round,
                    "state_dict": self.model.state_dict(),
                    "val_dice": global_val_dice,
                    "val_loss": val_loss,
                },
                f"{checkpoint_dir}/best.pt",
            )
            logger.info("Saved new best federated checkpoint (Dice=%.4f) to %s/best.pt", self.best_dice, checkpoint_dir)

        # Differential privacy budget
        epsilon = self.dp_accountant.compute_epsilon(server_round) if self.dp_accountant else 0.0

        # Publish metrics through single publication function
        publish_round_metrics(
            round_num=server_round,
            global_val_dice=global_val_dice,
            global_val_loss=val_loss,
            hospital_metrics=self.latest_hospital_metrics,
            participating_clients=len(self.latest_hospital_metrics),
            total_clients=self.total_clients,
            round_duration_sec=round_duration,
            tls_enabled=self.tls_enabled,
            he_enabled=self.he_enabled,
            dp_enabled=self.dp_enabled,
            epsilon=epsilon,
            sigma=self.noise_multiplier if self.dp_enabled else 0.0,
            csv_path=self.cfg.get("results", {}).get("metrics_csv", "results/metrics.csv"),
            json_path=self.cfg.get("results", {}).get("metrics_json", "results/metrics.json"),
            extra={"dice_breakdown": dice_metrics},
        )

        return val_loss, {
            "dice_mean": global_val_dice,
            "dice_tc": dice_metrics.get("dice_tc", 0.0),
            "dice_wt": dice_metrics.get("dice_wt", 0.0),
            "dice_et": dice_metrics.get("dice_et", 0.0),
        }


__all__ = ["FedMedStrategy"]
