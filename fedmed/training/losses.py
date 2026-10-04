"""Loss functions for multi-channel overlapping BraTS tumor segmentation."""

from __future__ import annotations

from typing import Any, Mapping
import torch.nn as nn
from monai.losses import DiceCELoss, DiceFocalLoss, DiceLoss

from fedmed.utils.logging import get_logger

logger = get_logger(__name__)


def get_loss(cfg: Mapping[str, Any]) -> nn.Module:
    """Build segmentation loss function from YAML configuration.

    Because BraTS channels (TC, WT, ET) are overlapping binary masks,
    sigmoid activation is applied independently per channel.

    Args:
        cfg: Validated configuration mapping.

    Returns:
        PyTorch / MONAI loss module.
    """
    loss_name = str(cfg.get("training", {}).get("loss", "dice_ce")).lower()

    if loss_name == "dice_ce":
        logger.info("Using DiceCELoss with sigmoid=True")
        return DiceCELoss(sigmoid=True, squared_pred=True)
    elif loss_name == "dice_focal":
        logger.info("Using DiceFocalLoss with sigmoid=True")
        return DiceFocalLoss(sigmoid=True, squared_pred=True)
    elif loss_name == "dice":
        logger.info("Using DiceLoss with sigmoid=True")
        return DiceLoss(sigmoid=True)
    else:
        raise ValueError(
            f"Unsupported loss function '{loss_name}'. Expected 'dice_ce', 'dice_focal', or 'dice'."
        )
