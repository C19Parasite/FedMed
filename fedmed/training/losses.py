"""Loss functions and wrappers for multi-channel overlapping BraTS tumor segmentation.

BraTS Segmentation Channel Representation:
- Channel 0: Tumor Core (TC)
- Channel 1: Whole Tumor (WT)
- Channel 2: Enhancing Tumor (ET)
Hierarchy: ET inside TC inside WT. There is NO separate background channel.
Because channels are overlapping binary indicators rather than mutually exclusive classes:
- sigmoid=True: applied independently per channel.
- softmax=False: mutually exclusive probability normalization would corrupt overlapping labels.
- to_onehot_y=False: labels are already multi-channel binary tensors [B, 3, D, H, W].
- include_background=True: all 3 channels represent tumor subregions; none is background.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence
import torch
import torch.nn as nn
from monai.losses import DiceCELoss, DiceFocalLoss, DiceLoss

from fedmed.utils.logging import get_logger

logger = get_logger(__name__)


class DeepSupervisionLoss(nn.Module):
    """Wrapper that computes weighted loss across multiple deep supervision heads.

    When deep supervision is enabled, the model returns either:
    - A 6D stacked tensor of shape [B, N, C, D, H, W] where N = deep_supr_num + 1.
    - A sequence (list or tuple) of [B, C, D, H, W] tensors.

    Weights follow a halving progression (1, 1/2, 1/4, ...) normalized to sum to 1.0.
    When input is a standard 5D tensor [B, C, D, H, W], this wrapper is a clean no-op.

    Args:
        base_loss: Underlying criterion (e.g. DiceCELoss).
        weights: Optional custom weights. If None, halving weights summing to 1 are computed.
    """

    def __init__(
        self,
        base_loss: nn.Module | Any,
        weights: Sequence[float] | None = None,
    ) -> None:
        super().__init__()
        self.base_loss = base_loss
        self._custom_weights = list(weights) if weights is not None else None

    def _compute_weights(self, n_stages: int, device: torch.device) -> torch.Tensor:
        if self._custom_weights is not None:
            raw = torch.tensor(self._custom_weights[:n_stages], dtype=torch.float32, device=device)
        else:
            raw = torch.tensor([0.5**i for i in range(n_stages)], dtype=torch.float32, device=device)
        return raw / raw.sum()

    def forward(
        self,
        logits: torch.Tensor | Sequence[torch.Tensor],
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute weighted deep supervision loss or single base loss.

        Args:
            logits: Output from model, either 5D [B, C, D, H, W], 6D [B, N, C, D, H, W],
                or a sequence of tensors.
            target: Ground truth binary masks [B, C, D, H, W].

        Returns:
            Scalar PyTorch loss tensor.
        """
        # Case 1: Sequence of intermediate stage outputs
        if isinstance(logits, (list, tuple)):
            n_stages = len(logits)
            weights = self._compute_weights(n_stages, target.device)
            total_loss = torch.tensor(0.0, device=target.device, dtype=torch.float32)
            for w, pred in zip(weights, logits):
                if pred.shape[2:] != target.shape[2:]:
                    pred = nn.functional.interpolate(
                        pred, size=target.shape[2:], mode="trilinear", align_corners=False
                    )
                total_loss = total_loss + w * self.base_loss(pred, target)
            return total_loss

        # Case 2: 6D stacked tensor from DynUNet deep_supervision in train mode [B, N, C, D, H, W]
        if isinstance(logits, torch.Tensor) and logits.ndim == 6:
            n_stages = logits.shape[1]
            weights = self._compute_weights(n_stages, logits.device)
            total_loss = torch.tensor(0.0, device=logits.device, dtype=torch.float32)
            for i in range(n_stages):
                pred = logits[:, i]
                if pred.shape[2:] != target.shape[2:]:
                    pred = nn.functional.interpolate(
                        pred, size=target.shape[2:], mode="trilinear", align_corners=False
                    )
                total_loss = total_loss + weights[i] * self.base_loss(pred, target)
            return total_loss

        # Case 3: Standard single 5D tensor [B, C, D, H, W] (no-op pass-through)
        return self.base_loss(logits, target)


def build_loss(cfg: Mapping[str, Any]) -> nn.Module:
    """Build segmentation loss function with optional deep supervision wrapper.

    Supported loss names in cfg['loss']['name']:
    - 'dice_ce': Combined Dice and CrossEntropy (BCEWithLogits for multi-label).
    - 'dice_focal': Combined Dice and Focal loss.
    - 'dice': Soft Dice loss alone.

    Config keys:
    - lambda_dice: Trade-off weight for Dice loss (default: 1.0).
    - lambda_ce / lambda_focal: Trade-off weight for auxiliary loss (default: 1.0).
    - gamma: Focusing parameter for focal loss (default: 2.0).
    - smooth_nr / smooth_dr: Smoothing constants to prevent zero/nan division (default: 1e-5).
    - batch: Whether to compute Dice over the batch dimension (default: True).
    - channel_weights: Per-channel weights for [TC, WT, ET] (default: [1.0, 1.0, 1.0]).

    Args:
        cfg: Validated configuration mapping.

    Returns:
        PyTorch loss module callable on (logits, targets).
    """
    loss_cfg = cfg.get("loss", {})
    loss_name = str(loss_cfg.get("name", "dice_ce")).lower()

    # Common parameters
    smooth_nr = float(loss_cfg.get("smooth_nr", 1e-5))
    smooth_dr = float(loss_cfg.get("smooth_dr", 1e-5))
    batch = bool(loss_cfg.get("batch", False))
    lambda_dice = float(loss_cfg.get("lambda_dice", 1.0))

    cw_raw = loss_cfg.get("channel_weights", [1.0, 1.0, 1.0])
    weight_tensor = torch.tensor(cw_raw, dtype=torch.float32) if cw_raw is not None else None

    # Required settings explained:
    # - sigmoid=True: independent probabilities per channel for overlapping tumor regions.
    # - to_onehot_y=False: labels are already 3-channel binary [B, 3, D, H, W].
    # - softmax=False: channels are not mutually exclusive.
    # - include_background=True: all 3 channels are tumor regions (no background channel).
    if loss_name == "dice_ce":
        lambda_ce = float(loss_cfg.get("lambda_ce", 1.0))
        # Note on weight in MONAI DiceCELoss:
        # Passed as 'weight', which applies to both Dice and CE loss components.
        base_loss = DiceCELoss(
            include_background=True,
            to_onehot_y=False,
            sigmoid=True,
            softmax=False,
            squared_pred=True,
            smooth_nr=smooth_nr,
            smooth_dr=smooth_dr,
            batch=batch,
            weight=weight_tensor,
            lambda_dice=lambda_dice,
            lambda_ce=lambda_ce,
        )
    elif loss_name == "dice_focal":
        lambda_focal = float(loss_cfg.get("lambda_focal", 1.0))
        gamma = float(loss_cfg.get("gamma", 2.0))
        base_loss = DiceFocalLoss(
            include_background=True,
            to_onehot_y=False,
            sigmoid=True,
            softmax=False,
            squared_pred=True,
            smooth_nr=smooth_nr,
            smooth_dr=smooth_dr,
            batch=batch,
            gamma=gamma,
            weight=weight_tensor,
            lambda_dice=lambda_dice,
            lambda_focal=lambda_focal,
        )
    elif loss_name == "dice":
        base_loss = DiceLoss(
            include_background=True,
            to_onehot_y=False,
            sigmoid=True,
            softmax=False,
            squared_pred=True,
            smooth_nr=smooth_nr,
            smooth_dr=smooth_dr,
            batch=batch,
            weight=weight_tensor,
        )
    else:
        raise ValueError(
            f"Unsupported loss name '{loss_name}'. Expected 'dice_ce', 'dice_focal', or 'dice'."
        )

    # Wrap in DeepSupervisionLoss to automatically handle multi-stage / deep-supervision outputs
    return DeepSupervisionLoss(base_loss=base_loss)


def get_loss(cfg: Mapping[str, Any]) -> nn.Module:
    """Backwards-compatible alias for existing callers and trainer."""
    if "loss" not in cfg and "training" in cfg and "loss" in cfg["training"]:
        # Legacy config format where loss name was in training.loss
        legacy_cfg = dict(cfg)
        legacy_cfg["loss"] = {"name": cfg["training"]["loss"]}
        return build_loss(legacy_cfg)
    return build_loss(cfg)
