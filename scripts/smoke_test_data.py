"""Smoke test script for BraTS data pipeline: loads one batch and inspects tensors."""

from __future__ import annotations

import argparse
import sys
import torch

from fedmed.data.loaders import get_loaders
from fedmed.utils.config import load_config
from fedmed.utils.logging import get_logger
from fedmed.utils.seed import set_seed

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for smoke test."""
    parser = argparse.ArgumentParser(description="Smoke test for FedMed data pipeline.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline.yaml",
        help="Path to YAML configuration file.",
    )
    return parser.parse_args()


def run_smoke_test(config_path: str) -> None:
    """Load config, build DataLoaders, fetch a single batch, and inspect tensor attributes.

    Args:
        config_path: Path to YAML configuration file.
    """
    cfg = load_config(config_path)
    seed = int(cfg["seed"])
    set_seed(seed)

    logger.info("Starting data pipeline smoke test with seed=%s", seed)
    loaders = get_loaders(cfg)

    train_loader = loaders["train"]
    val_loader = loaders["val"]

    logger.info("Train loader batches: %s | Val loader batches: %s", len(train_loader), len(val_loader))

    batch = next(iter(train_loader))
    image = batch["image"]
    label = batch["label"]

    print("=================== DATA PIPELINE SMOKE TEST ===================")
    print(f"Image Tensor:")
    print(f"  Shape:       {image.shape}  (Expected: [B, 4, D, H, W])")
    print(f"  Dtype:       {image.dtype}")
    print(f"  Min value:   {image.min().item():.4f}")
    print(f"  Max value:   {image.max().item():.4f}")
    print(f"  Mean value:  {image.mean().item():.4f}")
    print(f"  Std dev:     {image.std().item():.4f}")

    print(f"Label Tensor:")
    print(f"  Shape:       {label.shape}  (Expected: [B, 3, D, H, W] -> TC, WT, ET)")
    print(f"  Dtype:       {label.dtype}")
    print(f"  Unique vals: {torch.unique(label).tolist()}")
    print(f"  TC voxels:   {(label[:, 0] > 0.5).sum().item()}")
    print(f"  WT voxels:   {(label[:, 1] > 0.5).sum().item()}")
    print(f"  ET voxels:   {(label[:, 2] > 0.5).sum().item()}")
    print("================================================================")

    # Assertions
    assert image.ndim == 5, f"Expected 5D tensor [B, C, D, H, W], got {image.ndim}D"
    assert image.shape[1] == 4, f"Expected 4 image channels, got {image.shape[1]}"
    assert label.ndim == 5, f"Expected 5D tensor [B, C, D, H, W], got {label.ndim}D"
    assert label.shape[1] == 3, f"Expected 3 label channels (TC, WT, ET), got {label.shape[1]}"

    logger.info("Data pipeline smoke test PASSED successfully.")


def main() -> None:
    """CLI entry point for smoke test."""
    args = parse_args()
    try:
        run_smoke_test(args.config)
    except Exception as exc:
        logger.error("Data pipeline smoke test failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
