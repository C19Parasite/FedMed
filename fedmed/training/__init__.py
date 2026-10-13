"""Training routines, loss functions, and evaluation metrics."""

from fedmed.training.losses import DeepSupervisionLoss, build_loss, get_loss
from fedmed.training.metrics import compute_dice_score, evaluate_brats_dice
from fedmed.training.trainer import CentralizedTrainer

__all__ = [
    "CentralizedTrainer",
    "DeepSupervisionLoss",
    "build_loss",
    "compute_dice_score",
    "evaluate_brats_dice",
    "get_loss",
]
