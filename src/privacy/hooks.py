"""Extensible ProtectionPipeline hook architecture for Privacy-Preserving Federated Learning.

Execution Order:
  Outbound: compute_update -> clip (DP) -> add_gaussian_noise (DP) -> encrypt (CKKS HE) -> send
  Inbound:  receive -> decrypt (CKKS HE) -> divide by total_n -> apply to model
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import numpy as np
import torch
import torch.nn as nn

from fedmed.models.params import get_parameters as extract_model_parameters, set_parameters as apply_model_parameters
from fedmed.utils.logging import get_logger
from src.privacy.dp import add_gaussian_noise, clip_update_norm
from src.privacy.he import (
    decrypt_and_average,
    encrypt_update,
    flatten_tensors,
    load_hospital_context,
    unflatten_tensors,
)

logger = get_logger(__name__)


class ProtectionPipeline:
    """Manages the outbound and inbound transformation hooks for client updates."""

    def __init__(self, cfg: Mapping[str, Any], hospital_id: str = "hospital_1") -> None:
        self.cfg = cfg
        self.hospital_id = hospital_id

        self.dp_enabled = bool(cfg.get("dp", {}).get("enabled", False))
        self.he_enabled = bool(cfg.get("he", {}).get("enabled", False))

        self.dp_clip_norm = float(cfg.get("dp", {}).get("clip_norm", 1.0))
        self.dp_noise_multiplier = float(cfg.get("dp", {}).get("noise_multiplier", 1.0))

        self.hospital_context = None
        if self.he_enabled:
            ctx_path = cfg.get("he", {}).get("secret_context", "keys/hospital_context.tenseal")
            try:
                self.hospital_context = load_hospital_context(ctx_path)
                logger.info("[%s] ProtectionPipeline loaded hospital CKKS secret context from %s", hospital_id, ctx_path)
            except Exception as e:
                logger.warning("[%s] Could not load HE context (%s); HE will be disabled locally", hospital_id, e)
                self.he_enabled = False

    def process_outbound(
        self,
        initial_weights: List[np.ndarray],
        trained_weights: List[np.ndarray],
        num_examples: int,
    ) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        """Outbound pipeline hook: compute_update -> clip -> noise -> encrypt -> send.

        Args:
            initial_weights: Model weights before local training.
            trained_weights: Model weights after local training.
            num_examples: Local hospital sample count (n_i).

        Returns:
            Tuple of (parameters_to_send, metadata_dictionary).
        """
        # 1. compute_update: delta = trained - initial
        deltas = [trained - initial for initial, trained in zip(initial_weights, trained_weights)]

        meta: Dict[str, Any] = {
            "hospital_id": self.hospital_id,
            "num_examples": num_examples,
            "dp_applied": False,
            "he_applied": False,
        }

        # 2. Differential Privacy: clip L2 norm
        if self.dp_enabled:
            deltas, original_norm = clip_update_norm(deltas, clip_norm=self.dp_clip_norm)
            # 3. Differential Privacy: add Gaussian noise
            deltas = add_gaussian_noise(deltas, clip_norm=self.dp_clip_norm, noise_multiplier=self.dp_noise_multiplier)
            meta["dp_applied"] = True
            meta["original_norm"] = round(original_norm, 4)
            meta["clip_norm"] = self.dp_clip_norm
            meta["noise_multiplier"] = self.dp_noise_multiplier

        # 4. Homomorphic Encryption: TenSEAL CKKS
        if self.he_enabled and self.hospital_context is not None:
            encrypted_chunks, he_meta = encrypt_update(
                deltas, self.hospital_context, num_examples=num_examples
            )
            # Encode serialized ciphertext bytes into uint8 arrays for Flower transmission
            encoded_chunks = [np.frombuffer(c, dtype=np.uint8) for c in encrypted_chunks]
            meta["he_applied"] = True
            meta["he_meta"] = he_meta
            return encoded_chunks, meta

        # If HE is disabled, send updated weights directly (or delta if configured)
        # For standard FedAvg compatibility, if HE is off, send new trained weights
        if not self.dp_enabled:
            return trained_weights, meta

        # If DP is on but HE is off, reconstructed weights = initial + noisy_clipped_delta
        dp_weights = [initial + d for initial, d in zip(initial_weights, deltas)]
        return dp_weights, meta

    def process_inbound(
        self,
        received_parameters: List[np.ndarray],
        initial_weights: List[np.ndarray],
        model: nn.Module,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Inbound pipeline hook: receive -> decrypt -> apply to model.

        Args:
            received_parameters: Parameters received from server.
            initial_weights: Weights before this round.
            model: PyTorch model to update in-place.
            context: Server metadata dictionary.
        """
        if self.he_enabled and self.hospital_context is not None and context and context.get("he_applied"):
            # Serialized chunks are reconstructed from uint8 arrays
            raw_chunks = [arr.tobytes() for arr in received_parameters]
            shapes_and_types = context["shapes_and_types"]
            total_elements = context["total_elements"]
            total_n = context.get("total_n", 1)

            averaged_deltas, _ = decrypt_and_average(
                raw_chunks,
                shapes_and_types,
                total_elements,
                total_n,
                self.hospital_context,
            )
            # Apply delta: new_weights = initial_weights + averaged_delta
            new_weights = [init + delta for init, delta in zip(initial_weights, averaged_deltas)]
            apply_model_parameters(model, new_weights)
        else:
            # Standard FedAvg: received parameters are the aggregated model weights
            apply_model_parameters(model, received_parameters)


__all__ = ["ProtectionPipeline"]
