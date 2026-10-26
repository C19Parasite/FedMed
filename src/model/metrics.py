"""BraTS Dice metrics and evaluation utilities."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from monai.data import DataLoader

from fedmed.training.metrics import SegmentationMetrics, evaluate_brats_dice
from fedmed.utils.logging import get_logger

logger = get_logger(__name__)


def compute_dice_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-5,
) -> Dict[str, float]:
    """Compute channel-wise Dice scores for BraTS compartments (TC, WT, ET).

    Args:
        pred: Predicted probabilities or logits [B, 3, D, H, W] or [3, D, H, W].
        target: Binary ground-truth masks [B, 3, D, H, W] or [3, D, H, W].
        threshold: Binarization cutoff for probabilities.
        smooth: Laplace smoothing constant to prevent zero-division.

    Returns:
        Dictionary containing dice_tc, dice_wt, dice_et, and dice_mean.
    """
    if pred.ndim == 4:
        pred = pred.unsqueeze(0)
    if target.ndim == 4:
        target = target.unsqueeze(0)

    probs = torch.sigmoid(pred) if (pred.min() < 0 or pred.max() > 1.0) else pred
    binary_pred = (probs >= threshold).float()
    target = target.float()

    dice_dict: Dict[str, float] = {}
    channel_names = ["tc", "wt", "et"]

    dice_vals = []
    for c, name in enumerate(channel_names):
        p_c = binary_pred[:, c, ...]
        t_c = target[:, c, ...]
        intersection = (p_c * t_c).sum().item()
        total = p_c.sum().item() + t_c.sum().item()

        if total == 0:
            # Both prediction and target are empty: perfect agreement per BraTS rule
            score = 1.0
        else:
            score = float((2.0 * intersection + smooth) / (total + smooth))
        dice_dict[f"dice_{name}"] = round(score, 4)
        dice_vals.append(score)

    dice_dict["dice_mean"] = round(float(np.mean(dice_vals)), 4)
    return dice_dict


def evaluate_dataset(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device | str = "cpu",
    criterion: Optional[nn.Module] = None,
) -> Tuple[float, Dict[str, float]]:
    """Evaluate a segmentation model over an entire DataLoader.

    Args:
        model: PyTorch segmentation model.
        loader: Evaluation DataLoader.
        device: Evaluation device.
        criterion: Optional loss module.

    Returns:
        Tuple of (mean_loss, metrics_dictionary).
    """
    model.eval()
    dev = torch.device(device)
    model.to(dev)

    total_loss = 0.0
    tc_scores = []
    wt_scores = []
    et_scores = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(dev)
            labels = batch["label"].to(dev)

            outputs = model(images)
            if criterion is not None:
                loss = criterion(outputs, labels)
                total_loss += float(loss.item())

            dice = compute_dice_score(outputs, labels)
            tc_scores.append(dice["dice_tc"])
            wt_scores.append(dice["dice_wt"])
            et_scores.append(dice["dice_et"])

    num_batches = max(1, len(loader))
    mean_loss = total_loss / num_batches if criterion is not None else 0.0

    mean_tc = float(np.mean(tc_scores)) if tc_scores else 0.0
    mean_wt = float(np.mean(wt_scores)) if wt_scores else 0.0
    mean_et = float(np.mean(et_scores)) if et_scores else 0.0
    mean_dice = float(np.mean([mean_tc, mean_wt, mean_et]))

    metrics = {
        "dice_tc": round(mean_tc, 4),
        "dice_wt": round(mean_wt, 4),
        "dice_et": round(mean_et, 4),
        "dice_mean": round(mean_dice, 4),
    }
    return round(mean_loss, 4), metrics


__all__ = ["compute_dice_score", "evaluate_dataset"]
