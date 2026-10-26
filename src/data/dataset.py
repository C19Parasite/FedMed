"""Data loading and preprocessing utilities for 3D multi-parametric MRI."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Tuple
from monai.data import DataLoader

from fedmed.data.datasets import discover_cases
from fedmed.data.loaders import get_loaders, _make_dataset
from fedmed.data.transforms import get_eval_transforms, get_train_transforms
from fedmed.data.synthetic import generate_synthetic_dataset
from fedmed.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "discover_cases",
    "get_loaders",
    "get_train_transforms",
    "get_eval_transforms",
    "generate_synthetic_dataset",
]
