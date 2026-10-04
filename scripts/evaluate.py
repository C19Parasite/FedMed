"""CLI script to evaluate a trained 3D U-Net checkpoint on BraTS splits."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import torch

from fedmed.data.loaders import get_loaders
from fedmed.models.unet3d import get_model
from fedmed.training.metrics import evaluate_brats_dice
from fedmed.utils.config import load_config
from fedmed.utils.logging import get_logger
from fedmed.utils.seed import set_seed

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI options for model evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate 3D U-Net checkpoint on BraTS dataset split."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_metric_model.pt",
        help="Path to saved model checkpoint.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["val", "test", "train"],
        help="Dataset split to evaluate on.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Evaluation device ('cuda' or 'cpu').",
    )
    return parser.parse_args()


def main() -> None:
    """Run evaluation and report segmentation accuracy metrics."""
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]))

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_file():
        logger.error("Checkpoint file not found: %s", ckpt_path.name)
        sys.exit(1)

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    model = get_model(cfg).to(device)
    checkpoint = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    logger.info("Loaded checkpoint from epoch %s", checkpoint.get("epoch", "unknown"))

    loaders = get_loaders(cfg)
    eval_loader = loaders[args.split]

    total_metrics = {"dice_tc": 0.0, "dice_wt": 0.0, "dice_et": 0.0, "dice_mean": 0.0}
    num_batches = 0

    with torch.no_grad():
        for batch in eval_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            logits = model(images)
            batch_metrics = evaluate_brats_dice(logits, labels)

            for k, v in batch_metrics.items():
                total_metrics[k] += v
            num_batches += 1

    count = max(num_batches, 1)
    mean_metrics = {k: v / count for k, v in total_metrics.items()}

    print("\n================ EVALUATION METRIC REPORT ================")
    print(f"Split:               {args.split}")
    print(f"Tumor Core (TC):     {mean_metrics['dice_tc']:.4f}")
    print(f"Whole Tumor (WT):    {mean_metrics['dice_wt']:.4f}")
    print(f"Enhancing Tumor (ET):{mean_metrics['dice_et']:.4f}")
    print(f"Mean BraTS Dice:     {mean_metrics['dice_mean']:.4f}")
    print("==========================================================")


if __name__ == "__main__":
    main()
