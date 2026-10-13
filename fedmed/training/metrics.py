"""Evaluation metrics for BraTS tumor sub-regions (TC, WT, ET).

Provides:
- SegmentationMetrics: Stateful accumulator tracking Dice and HD95 across volumes.
- dice_per_channel: Stateless Dice calculation with configurable empty ground-truth policy.
- hd95_per_channel: Stateless 95th-percentile Hausdorff Distance calculation.
- compute_dice_score, evaluate_brats_dice: Backwards-compatible legacy helpers.

Empty ground-truth policy:
- 'brats' (default):
    GT empty and prediction empty  => Dice = 1.0, HD95 = 0.0
    GT empty and prediction not empty => Dice = 0.0, HD95 = max_hd95
    GT not empty and prediction empty => Dice = 0.0, HD95 = max_hd95
- 'skip':
    Exclude volumes with empty GT from that channel's average (reported via n_valid_<channel>).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Sequence
import numpy as np
import torch
from monai.metrics import HausdorffDistanceMetric

from fedmed.utils.logging import get_logger

logger = get_logger(__name__)

# Default maximum HD95 cap in mm: diagonal of a standard 240x240x155 BraTS volume
# sqrt(240^2 + 240^2 + 155^2) = sqrt(57600 + 57600 + 24025) = sqrt(139225) ≈ 373.13 mm
DEFAULT_MAX_HD95 = 373.13


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


def dice_per_channel(
    pred: torch.Tensor,
    target: torch.Tensor,
    empty_policy: str = "brats",
) -> torch.Tensor:
    """Compute channel-wise Dice scores for a batch of volumes.

    Args:
        pred: Binary predictions of shape [B, 3, D, H, W] (values in {0, 1}).
        target: Binary ground truth masks of shape [B, 3, D, H, W] (values in {0, 1}).
        empty_policy: 'brats' (empty/empty -> 1.0, mismatch -> 0.0) or 'skip' (empty GT -> NaN).

    Returns:
        PyTorch tensor of shape [B, 3] on CPU containing per-volume, per-channel Dice scores.
    """
    if pred.ndim != 5 or target.ndim != 5:
        raise ValueError(
            f"Expected 5D inputs [B, 3, D, H, W], got pred {pred.shape} and target {target.shape}"
        )
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs target {target.shape}")

    b_size, n_channels = pred.shape[0], pred.shape[1]
    scores = torch.zeros((b_size, n_channels), dtype=torch.float32, device="cpu")

    pred_cpu = pred.detach().cpu()
    target_cpu = target.detach().cpu()

    for b in range(b_size):
        for c in range(n_channels):
            p = pred_cpu[b, c]
            t = target_cpu[b, c]

            p_sum = p.sum().item()
            t_sum = t.sum().item()

            gt_empty = (t_sum == 0)
            pred_empty = (p_sum == 0)

            if gt_empty:
                if empty_policy == "skip":
                    scores[b, c] = float("nan")
                elif empty_policy == "brats":
                    scores[b, c] = 1.0 if pred_empty else 0.0
                else:
                    raise ValueError(f"Unknown empty_policy '{empty_policy}'")
            else:
                if pred_empty:
                    scores[b, c] = 0.0
                else:
                    intersection = (p * t).sum().item()
                    dice_val = (2.0 * intersection) / (p_sum + t_sum)
                    scores[b, c] = float(dice_val)

    return scores


def hd95_per_channel(
    pred: torch.Tensor,
    target: torch.Tensor,
    spacing: Sequence[float] | None = None,
    max_hd95: float = DEFAULT_MAX_HD95,
    empty_policy: str = "brats",
) -> torch.Tensor:
    """Compute channel-wise 95th-percentile Hausdorff Distance (HD95) for a batch of volumes.

    Args:
        pred: Binary predictions [B, 3, D, H, W] on any device.
        target: Binary ground truth masks [B, 3, D, H, W] on any device.
        spacing: Voxel spacing along spatial dimensions [D, H, W] in mm (default: [1.0, 1.0, 1.0]).
        max_hd95: Distance cap value in mm replacing inf/nan (default: 373.13).
        empty_policy: 'brats' (both empty -> 0.0, mismatch -> max_hd95) or 'skip' (empty GT -> NaN).

    Returns:
        CPU PyTorch tensor of shape [B, 3] with values in [0.0, max_hd95] (or NaN for skipped).
    """
    if pred.ndim != 5 or target.ndim != 5:
        raise ValueError(
            f"Expected 5D inputs [B, 3, D, H, W], got pred {pred.shape} and target {target.shape}"
        )
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs target {target.shape}")

    # Move tensors to CPU for thread-safe and CUDA-safe surface distance computation
    pred_cpu = pred.detach().cpu().float()
    target_cpu = target.detach().cpu().float()

    voxel_spacing = list(spacing) if spacing is not None else [1.0, 1.0, 1.0]
    b_size, n_channels = pred.shape[0], pred.shape[1]
    scores = torch.zeros((b_size, n_channels), dtype=torch.float32, device="cpu")

    hd95_metric = HausdorffDistanceMetric(
        include_background=True,
        percentile=95,
        reduction="none",
    )

    for b in range(b_size):
        for c in range(n_channels):
            p = pred_cpu[b : b + 1, c : c + 1]
            t = target_cpu[b : b + 1, c : c + 1]

            gt_empty = (t.sum().item() == 0)
            pred_empty = (p.sum().item() == 0)

            if gt_empty:
                if empty_policy == "skip":
                    scores[b, c] = float("nan")
                elif empty_policy == "brats":
                    scores[b, c] = 0.0 if pred_empty else max_hd95
                else:
                    raise ValueError(f"Unknown empty_policy '{empty_policy}'")
            elif pred_empty:
                scores[b, c] = max_hd95
            else:
                try:
                    val = hd95_metric(p, t, spacing=voxel_spacing)
                    raw_score = float(val[0, 0].item())
                    if math.isnan(raw_score) or math.isinf(raw_score):
                        scores[b, c] = max_hd95
                    else:
                        scores[b, c] = min(raw_score, max_hd95)
                except Exception:
                    scores[b, c] = max_hd95

    return scores


class SegmentationMetrics:
    """Stateful metric accumulator for multi-channel BraTS segmentation.

    Accumulates Dice and HD95 scores across batches, applying
    sigmoid thresholding and configurable empty ground-truth policies.

    Args:
        cfg: Optional configuration dictionary containing 'metrics' section.
        threshold: Probability cutoff after sigmoid (default: 0.5).
        empty_policy: 'brats' or 'skip' (default: 'brats').
        max_hd95: Upper cap for HD95 in mm (default: 373.13).
        spacing: Voxel spacing in mm (default: [1.0, 1.0, 1.0]).
    """

    def __init__(
        self,
        cfg: Mapping[str, Any] | None = None,
        threshold: float = 0.5,
        empty_policy: str = "brats",
        max_hd95: float = DEFAULT_MAX_HD95,
        spacing: Sequence[float] | None = None,
    ) -> None:
        if cfg is not None:
            m_cfg = cfg.get("metrics", {})
            self.threshold = float(m_cfg.get("threshold", threshold))
            self.empty_policy = str(m_cfg.get("empty_policy", empty_policy)).lower()
            self.max_hd95 = float(m_cfg.get("max_hd95", max_hd95))
            self.spacing = list(m_cfg.get("spacing", spacing if spacing else [1.0, 1.0, 1.0]))
            hd95_sub = m_cfg.get("hd95", {})
            self.hd95_enabled = bool(hd95_sub.get("enabled", False))
            self.hd95_every_n_epochs = int(hd95_sub.get("every_n_epochs", 5))
        else:
            self.threshold = threshold
            self.empty_policy = empty_policy.lower()
            self.max_hd95 = max_hd95
            self.spacing = list(spacing) if spacing else [1.0, 1.0, 1.0]
            self.hd95_enabled = False
            self.hd95_every_n_epochs = 5

        self.reset()

    def reset(self) -> None:
        """Clear all accumulated volume records."""
        self.dice_records: List[torch.Tensor] = []
        self.raw_preds_labels: List[tuple[torch.Tensor, torch.Tensor]] = []
        self.n_volumes = 0

    def update(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        """Process a batch of predictions and labels, updating accumulated records.

        Args:
            logits: Output logits of shape [B, 3, D, H, W].
            labels: Ground truth binary masks of shape [B, 3, D, H, W].
        """
        if logits.ndim != 5 or labels.ndim != 5:
            raise ValueError(
                f"Expected 5D inputs [B, 3, D, H, W], got logits {logits.shape} "
                f"and labels {labels.shape}"
            )
        if logits.shape[1] != 3 or labels.shape[1] != 3:
            raise ValueError(
                f"Expected 3 channels (TC, WT, ET), got logits channels={logits.shape[1]} "
                f"and labels channels={labels.shape[1]}"
            )
        if logits.shape != labels.shape:
            raise ValueError(f"Shape mismatch: logits {logits.shape} vs labels {labels.shape}")

        probs = torch.sigmoid(logits)
        preds = (probs > self.threshold).float()

        dice_batch = dice_per_channel(preds, labels, empty_policy=self.empty_policy)
        self.dice_records.append(dice_batch)

        # Store CPU copies for optional HD95 evaluation during compute()
        self.raw_preds_labels.append((preds.detach().cpu(), labels.detach().cpu()))

        self.n_volumes += logits.shape[0]

    def compute(self, include_hd95: bool = False) -> Dict[str, float]:
        """Aggregate accumulated results across all recorded volumes.

        Args:
            include_hd95: Whether to include 95th-percentile Hausdorff Distance metrics.

        Returns:
            Dictionary containing 'dice_tc', 'dice_wt', 'dice_et', 'dice_mean', 'n_volumes',
            and optionally 'hd95_tc', 'hd95_wt', 'hd95_et', 'hd95_mean'.
        """
        if self.n_volumes == 0 or not self.dice_records:
            res = {
                "dice_tc": 0.0,
                "dice_wt": 0.0,
                "dice_et": 0.0,
                "dice_mean": 0.0,
                "n_volumes": 0,
            }
            if include_hd95:
                res.update({
                    "hd95_tc": self.max_hd95,
                    "hd95_wt": self.max_hd95,
                    "hd95_et": self.max_hd95,
                    "hd95_mean": self.max_hd95,
                })
            return res

        all_dice = torch.cat(self.dice_records, dim=0).numpy()  # [N, 3]

        tc_scores = all_dice[:, 0]
        wt_scores = all_dice[:, 1]
        et_scores = all_dice[:, 2]

        dice_tc = float(np.nanmean(tc_scores)) if not np.isnan(tc_scores).all() else 0.0
        dice_wt = float(np.nanmean(wt_scores)) if not np.isnan(wt_scores).all() else 0.0
        dice_et = float(np.nanmean(et_scores)) if not np.isnan(et_scores).all() else 0.0
        dice_mean = float((dice_tc + dice_wt + dice_et) / 3.0)

        results: Dict[str, float] = {
            "dice_tc": dice_tc,
            "dice_wt": dice_wt,
            "dice_et": dice_et,
            "dice_mean": dice_mean,
            "n_volumes": self.n_volumes,
        }

        if self.empty_policy == "skip":
            results["n_valid_tc"] = int(np.count_nonzero(~np.isnan(tc_scores)))
            results["n_valid_wt"] = int(np.count_nonzero(~np.isnan(wt_scores)))
            results["n_valid_et"] = int(np.count_nonzero(~np.isnan(et_scores)))

        # Optional expensive HD95 computation
        if include_hd95 and self.raw_preds_labels:
            hd95_batches = []
            for p, l in self.raw_preds_labels:
                hd95_batch = hd95_per_channel(
                    p,
                    l,
                    spacing=self.spacing,
                    max_hd95=self.max_hd95,
                    empty_policy=self.empty_policy,
                )
                hd95_batches.append(hd95_batch)

            all_hd95 = torch.cat(hd95_batches, dim=0).numpy()  # [N, 3]

            hd_tc = float(np.nanmean(all_hd95[:, 0])) if not np.isnan(all_hd95[:, 0]).all() else self.max_hd95
            hd_wt = float(np.nanmean(all_hd95[:, 1])) if not np.isnan(all_hd95[:, 1]).all() else self.max_hd95
            hd_et = float(np.nanmean(all_hd95[:, 2])) if not np.isnan(all_hd95[:, 2]).all() else self.max_hd95
            hd_mean = float((hd_tc + hd_wt + hd_et) / 3.0)

            results["hd95_tc"] = hd_tc
            results["hd95_wt"] = hd_wt
            results["hd95_et"] = hd_et
            results["hd95_mean"] = hd_mean

        return results
