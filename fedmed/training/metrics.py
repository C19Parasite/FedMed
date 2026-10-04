"""Evaluation metrics for BraTS tumor sub-regions (TC, WT, ET)."""

from __future__ import annotations

from typing import Dict
import torch


def compute_dice_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1e-5,
) -> float:
    """Compute binary Dice coefficient between prediction and target tensors.

    Args:
        pred: Binary prediction tensor (values in {0, 1}).
        target: Binary ground truth tensor (values in {0, 1}).
        smooth: Smoothing epsilon to prevent division by zero.

    Returns:
        Scalar Dice score in [0.0, 1.0].
    """
    pred_flat = pred.contiguous().view(-1).float()
    target_flat = target.contiguous().view(-1).float()

    intersection = (pred_flat * target_flat).sum().item()
    cardinality = pred_flat.sum().item() + target_flat.sum().item()

    if cardinality == 0:
        return 1.0 if intersection == 0 else 0.0

    return float((2.0 * intersection + smooth) / (cardinality + smooth))


def evaluate_brats_dice(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute channel-wise Dice scores for Tumor Core, Whole Tumor, and Enhancing Tumor.

    Channel convention:
        - Channel 0: Tumor Core (TC)
        - Channel 1: Whole Tumor (WT)
        - Channel 2: Enhancing Tumor (ET)

    Args:
        logits: Model output logits tensor of shape (B, 3, ...).
        targets: Ground truth binary masks tensor of shape (B, 3, ...).
        threshold: Probability cutoff after sigmoid.

    Returns:
        Dictionary containing 'dice_tc', 'dice_wt', 'dice_et', and 'dice_mean'.
    """
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    tc_dice = compute_dice_score(preds[:, 0], targets[:, 0])
    wt_dice = compute_dice_score(preds[:, 1], targets[:, 1])
    et_dice = compute_dice_score(preds[:, 2], targets[:, 2])

    mean_dice = (tc_dice + wt_dice + et_dice) / 3.0

    return {
        "dice_tc": tc_dice,
        "dice_wt": wt_dice,
        "dice_et": et_dice,
        "dice_mean": mean_dice,
    }
