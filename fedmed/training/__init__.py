"""Training routines, loss functions, and evaluation metrics."""

from fedmed.training.losses import DeepSupervisionLoss, build_loss, get_loss
from fedmed.training.metrics import (
    SegmentationMetrics,
    compute_dice_score,
    dice_per_channel,
    evaluate_brats_dice,
    hd95_per_channel,
)
from fedmed.training.optim import build_optimizer, build_scheduler, get_lr
from fedmed.training.run import (
    config_hash,
    create_run_dir,
    environment_info,
    save_environment_info,
    save_resolved_config,
)
from fedmed.training.trainer import CentralizedTrainer

__all__ = [
    "CentralizedTrainer",
    "DeepSupervisionLoss",
    "SegmentationMetrics",
    "build_loss",
    "build_optimizer",
    "build_scheduler",
    "compute_dice_score",
    "config_hash",
    "create_run_dir",
    "dice_per_channel",
    "environment_info",
    "evaluate_brats_dice",
    "get_loss",
    "get_lr",
    "hd95_per_channel",
    "save_environment_info",
    "save_resolved_config",
]
