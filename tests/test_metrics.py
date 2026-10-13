"""Unit tests for segmentation metrics, empty policies, and stateful accumulation."""

import pytest
import torch

from fedmed.training.metrics import SegmentationMetrics, dice_per_channel


def test_dice_perfect_prediction() -> None:
    """Verify perfect prediction gives Dice 1.0 for all channels."""
    target = torch.zeros((1, 3, 16, 16, 16))
    target[0, :, 4:12, 4:12, 4:12] = 1.0
    pred = target.clone()

    scores = dice_per_channel(pred, target)
    assert scores.shape == (1, 3)
    assert torch.allclose(scores, torch.ones((1, 3)))


def test_dice_all_zero_prediction_vs_non_empty_target() -> None:
    """Verify all-zero prediction vs non-empty target gives Dice 0.0."""
    target = torch.zeros((1, 3, 16, 16, 16))
    target[0, :, 4:12, 4:12, 4:12] = 1.0
    pred = torch.zeros_like(target)

    scores = dice_per_channel(pred, target)
    assert torch.allclose(scores, torch.zeros((1, 3)))


def test_dice_half_overlap_analytical_case() -> None:
    """Verify half-overlap cube case gives the analytically calculated Dice score.

    Cube A: [4:12, 4:12, 4:12] -> size 8x8x8 = 512 voxels
    Cube B: [8:16, 4:12, 4:12] -> size 8x8x8 = 512 voxels
    Intersection: [8:12, 4:12, 4:12] -> size 4x8x8 = 256 voxels
    Dice = 2 * 256 / (512 + 512) = 512 / 1024 = 0.50
    """
    target = torch.zeros((1, 3, 20, 20, 20))
    target[0, :, 4:12, 4:12, 4:12] = 1.0

    pred = torch.zeros((1, 3, 20, 20, 20))
    pred[0, :, 8:16, 4:12, 4:12] = 1.0

    expected_dice = 2.0 * 256.0 / (512.0 + 512.0)
    assert expected_dice == 0.5

    scores = dice_per_channel(pred, target)
    for c in range(3):
        assert pytest.approx(scores[0, c].item(), abs=1e-5) == 0.5


def test_empty_ground_truth_policy() -> None:
    """Verify empty-GT behavior under 'brats' (1.0 if both empty, 0.0 if mismatched) and 'skip' (NaN)."""
    # Volume 0: both empty for channel 0
    # Volume 1: target empty, prediction has false positive
    # Volume 2: target has tumor, prediction empty
    pred = torch.zeros((3, 3, 10, 10, 10))
    target = torch.zeros((3, 3, 10, 10, 10))

    # Vol 1, ch 0: pred non-empty, target empty
    pred[1, 0, 2:5, 2:5, 2:5] = 1.0

    # Vol 2, ch 0: pred empty, target non-empty
    target[2, 0, 2:5, 2:5, 2:5] = 1.0

    # 1. Under 'brats' policy
    scores_brats = dice_per_channel(pred, target, empty_policy="brats")
    assert scores_brats[0, 0].item() == 1.0  # both empty -> 1.0
    assert scores_brats[1, 0].item() == 0.0  # pred non-empty, target empty -> 0.0
    assert scores_brats[2, 0].item() == 0.0  # pred empty, target non-empty -> 0.0

    # 2. Under 'skip' policy
    scores_skip = dice_per_channel(pred, target, empty_policy="skip")
    assert torch.isnan(scores_skip[0, 0])  # target empty -> NaN
    assert torch.isnan(scores_skip[1, 0])  # target empty -> NaN
    assert scores_skip[2, 0].item() == 0.0  # target non-empty -> 0.0


def test_segmentation_metrics_accumulator_batch_equivalence() -> None:
    """Verify accumulating two batches yields identical mean Dice to one combined batch."""
    b1_logits = torch.randn(2, 3, 10, 10, 10)
    b1_labels = (torch.rand(2, 3, 10, 10, 10) > 0.5).float()

    b2_logits = torch.randn(3, 3, 10, 10, 10)
    b2_labels = (torch.rand(3, 3, 10, 10, 10) > 0.5).float()

    # Accumulated evaluation
    metrics_acc = SegmentationMetrics(threshold=0.5, empty_policy="brats")
    metrics_acc.update(b1_logits, b1_labels)
    metrics_acc.update(b2_logits, b2_labels)
    res_acc = metrics_acc.compute()

    # Combined single-batch evaluation
    combined_logits = torch.cat([b1_logits, b2_logits], dim=0)
    combined_labels = torch.cat([b1_labels, b2_labels], dim=0)

    metrics_combined = SegmentationMetrics(threshold=0.5, empty_policy="brats")
    metrics_combined.update(combined_logits, combined_labels)
    res_combined = metrics_combined.compute()

    assert res_acc["n_volumes"] == 5
    assert pytest.approx(res_acc["dice_mean"], abs=1e-5) == res_combined["dice_mean"]
    assert pytest.approx(res_acc["dice_tc"], abs=1e-5) == res_combined["dice_tc"]
    assert pytest.approx(res_acc["dice_wt"], abs=1e-5) == res_combined["dice_wt"]
    assert pytest.approx(res_acc["dice_et"], abs=1e-5) == res_combined["dice_et"]


def test_threshold_sensitivity() -> None:
    """Verify logits thresholding correctly differentiates predictions at threshold 0.3 vs 0.7."""
    # Construct logits whose sigmoid is exactly ~0.5 (logit = 0.0)
    logits = torch.zeros(1, 3, 10, 10, 10)  # sigmoid(0.0) = 0.5
    target = torch.ones(1, 3, 10, 10, 10)

    # At threshold 0.3: sigmoid(0.0) = 0.5 > 0.3 -> predicts 1.0 -> Dice = 1.0
    m_low = SegmentationMetrics(threshold=0.3)
    m_low.update(logits, target)
    res_low = m_low.compute()
    assert res_low["dice_mean"] == 1.0

    # At threshold 0.7: sigmoid(0.0) = 0.5 < 0.7 -> predicts 0.0 -> Dice = 0.0
    m_high = SegmentationMetrics(threshold=0.7)
    m_high.update(logits, target)
    res_high = m_high.compute()
    assert res_high["dice_mean"] == 0.0
