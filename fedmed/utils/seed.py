"""Reproducible seeding for Python, NumPy, and PyTorch across CPU and CUDA."""

from __future__ import annotations

import os
import random

import numpy as np
import torch

from fedmed.utils.logging import get_logger

logger = get_logger(__name__)


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, PyTorch (CPU and CUDA), and configure deterministic flags.

    Args:
        seed: Integer seed value from configuration.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info("set_seed completed with seed=%s", seed)


# Backward-compatible alias for existing callers
seed_everything = set_seed
