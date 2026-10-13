"""Integration tests connecting data loaders, model, loss, metrics, and sliding-window inference."""

from pathlib import Path
import pytest
import torch
from monai.inferers import sliding_window_inference

from fedmed.data.loaders import get_loaders
from fedmed.models.factory import build_model
from fedmed.training.losses import build_loss
from fedmed.training.metrics import SegmentationMetrics
from fedmed.utils.config import load_config


def test_smoke_loader_end_to_end_training_and_metrics(tmp_path: Path) -> None:
    """Verify an end-to-end cycle: smoke loader -> model -> loss -> backward -> metrics."""
    cfg = load_config("configs/smoke_test.yaml")
    cfg["data"]["root_dir"] = str(tmp_path / "data")
    cfg["data"]["splits_path"] = str(tmp_path / "data" / "splits.json")
    cfg["data"]["metadata_path"] = str(tmp_path / "data" / "metadata.json")
    cfg["data"]["cache_dir"] = str(tmp_path / "data" / "cache")
    cfg["data"]["roi_size"] = [16, 16, 16]
    cfg["data"]["synthetic"]["num_patients"] = 2
    cfg["data"]["synthetic"]["spatial_size"] = [16, 16, 16]
    cfg["data"]["batch_size"] = 1

    # 1. Build loaders from Prompt 2 data pipeline
    loaders = get_loaders(cfg)
    train_loader = loaders["train"]

    # 2. Build model and loss from Prompt 3
    model = build_model(cfg)
    loss_fn = build_loss(cfg)
    metrics = SegmentationMetrics(cfg)

    # 3. Pull one batch
    batch = next(iter(train_loader))
    images = batch["image"]
    labels = batch["label"]

    assert images.ndim == 5
    assert images.shape[1] == 4  # T1, T1ce, T2, FLAIR
    assert labels.ndim == 5
    assert labels.shape[1] == 3  # TC, WT, ET

    # 4. Forward pass
    model.train()
    logits = model(images)
    assert logits.shape == labels.shape

    # 5. Loss computation and backward pass
    loss = loss_fn(logits, labels)
    assert loss.ndim == 0
    assert torch.isfinite(loss).item()

    loss.backward()
    grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    assert grad_norm > 0.0

    # 6. Evaluation through SegmentationMetrics
    metrics.update(logits, labels)
    results = metrics.compute(include_hd95=True)

    assert "dice_tc" in results
    assert "dice_wt" in results
    assert "dice_et" in results
    assert "dice_mean" in results
    assert "hd95_mean" in results
    assert results["n_volumes"] == 1
    assert 0.0 <= results["dice_mean"] <= 1.0
    assert 0.0 <= results["hd95_mean"] <= metrics.max_hd95


def test_sliding_window_inference_compatibility() -> None:
    """Verify monai.inferers.sliding_window_inference evaluates volume larger than model ROI."""
    cfg = load_config("configs/smoke_test.yaml")
    model = build_model(cfg)
    model.eval()

    roi_size = tuple(cfg["data"]["roi_size"])  # (16, 16, 16)
    large_input = torch.randn(1, 4, 32, 32, 32)  # 2x larger in all dimensions

    with torch.no_grad():
        full_pred = sliding_window_inference(
            inputs=large_input,
            roi_size=roi_size,
            sw_batch_size=2,
            predictor=model,
            overlap=0.25,
            mode="gaussian",
        )

    # Output must retain the full input spatial dimensions
    assert full_pred.shape == (1, 3, 32, 32, 32)
    assert torch.isfinite(full_pred).all()
