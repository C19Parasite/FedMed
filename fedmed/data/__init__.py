"""Datasets, transforms, splits, and DataLoaders for BraTS MRI segmentation."""

from fedmed.data.datasets import (
    build_datalist,
    discover_brats_local,
    discover_cases,
    discover_decathlon,
    discover_synthetic,
)
from fedmed.data.loaders import build_loader, get_loaders
from fedmed.data.metadata import compute_and_save_metadata
from fedmed.data.splits import load_or_create_splits, make_splits
from fedmed.data.synthetic import generate_synthetic_dataset
from fedmed.data.transforms import (
    ConvertToBratsChannelsd,
    get_eval_transforms,
    get_train_transforms,
)

__all__ = [
    "ConvertToBratsChannelsd",
    "build_datalist",
    "build_loader",
    "compute_and_save_metadata",
    "discover_brats_local",
    "discover_cases",
    "discover_decathlon",
    "discover_synthetic",
    "generate_synthetic_dataset",
    "get_eval_transforms",
    "get_loaders",
    "get_train_transforms",
    "load_or_create_splits",
    "make_splits",
]
