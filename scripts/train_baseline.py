"""CLI script to train centralized 3D U-Net baseline on BraTS MRI dataset."""

from __future__ import annotations

import argparse
import sys

from fedmed.data.loaders import get_loaders
from fedmed.models.unet3d import get_model
from fedmed.training.trainer import CentralizedTrainer
from fedmed.utils.config import load_config
from fedmed.utils.logging import get_logger
from fedmed.utils.seed import set_seed

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI options for centralized baseline training."""
    parser = argparse.ArgumentParser(
        description="Train centralized 3D U-Net baseline for brain tumor segmentation."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override total epochs specified in config.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Target execution device (e.g. 'cuda' or 'cpu').",
    )
    return parser.parse_args()


def main() -> None:
    """Execute baseline training run."""
    args = parse_args()
    cfg = load_config(args.config)

    seed = int(cfg["seed"])
    set_seed(seed)

    log_cfg = cfg.get("logging", {})
    logger.info("Initializing centralized training run. Seed=%s", seed)

    loaders = get_loaders(cfg)
    train_loader = loaders["train"]
    val_loader = loaders["val"]

    model = get_model(cfg)
    trainer = CentralizedTrainer(model=model, cfg=cfg, device=args.device)

    epochs = args.epochs if args.epochs is not None else int(cfg["training"]["epochs"])
    results = trainer.fit(train_loader=train_loader, val_loader=val_loader, epochs=epochs)

    print("\n================ CENTRALIZED BASELINE RESULTS ================")
    print(f"Best Mean Dice: {results['best_dice']:.4f}")
    print(f"Checkpoints directory: {cfg['training']['save_dir']}")
    print("==============================================================")


if __name__ == "__main__":
    main()
