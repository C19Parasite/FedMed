"""Train/val/test DataLoaders over discovered cases."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from monai.data import CacheDataset, DataLoader, Dataset, PersistentDataset

from fedmed.data.datasets import discover_cases
from fedmed.data.metadata import compute_and_save_metadata
from fedmed.data.splits import load_or_create_splits
from fedmed.data.transforms import get_eval_transforms, get_train_transforms
from fedmed.utils.logging import get_logger

logger = get_logger(__name__)

_MODES = frozenset({"train", "val", "test"})


def get_loaders(cfg: Mapping[str, Any]) -> dict[str, DataLoader]:
    """Build train, val, and test loaders from ``splits.json``.

    Args:
        cfg: Validated config mapping.

    Returns:
        Dict with keys ``train``, ``val``, ``test``.
    """
    cases = discover_cases(cfg)
    data_cfg = cfg["data"]
    splits = load_or_create_splits(
        patient_ids=[c["patient_id"] for c in cases],
        splits_path=data_cfg["splits_path"],
        seed=int(cfg["seed"]),
        train_ratio=float(data_cfg["train_ratio"]),
        val_ratio=float(data_cfg["val_ratio"]),
        test_ratio=float(data_cfg["test_ratio"]),
    )
    compute_and_save_metadata(
        cases,
        metadata_path=data_cfg["metadata_path"],
        label_maps=data_cfg["label_maps"],
        label_scheme=str(data_cfg["label_scheme"]),
    )
    loaders = {
        mode: build_loader(cfg, splits[mode], mode, cases=cases)
        for mode in ("train", "val", "test")
    }
    logger.info(
        "built loaders n_train=%s n_val=%s n_test=%s",
        len(splits["train"]),
        len(splits["val"]),
        len(splits["test"]),
    )
    return loaders


def build_loader(
    cfg: Mapping[str, Any],
    patient_ids: Sequence[str],
    mode: str,
    cases: Sequence[dict[str, Any]] | None = None,
) -> DataLoader:
    """Build a DataLoader for an explicit patient-ID partition.

    Federated clients can pass their own ID list. Splits.json is not required.

    Args:
        cfg: Validated config mapping.
        patient_ids: IDs to include.
        mode: ``train``, ``val``, or ``test``.
        cases: Optional pre-discovered case records to avoid redundant directory scanning.

    Returns:
        MONAI / PyTorch DataLoader.
    """
    if mode not in _MODES:
        raise ValueError(f"mode must be one of {sorted(_MODES)}, got '{mode}'")
    id_set = set(patient_ids)
    all_cases = cases if cases is not None else discover_cases(cfg)
    matched_cases = [case for case in all_cases if case["patient_id"] in id_set]
    cases = sorted(matched_cases, key=lambda c: c["patient_id"])
    transform = get_train_transforms(cfg) if mode == "train" else get_eval_transforms(cfg)
    dataset = _make_dataset(cfg, cases, transform)
    data_cfg = cfg["data"]
    if mode == "train":
        batch_size = int(data_cfg["batch_size"])
        shuffle = True
    elif mode == "val":
        batch_size = int(data_cfg["val_batch_size"])
        shuffle = False
    else:
        batch_size = int(data_cfg["test_batch_size"])
        shuffle = False
    logger.info("build_loader mode=%s n_patients=%s batch_size=%s", mode, len(cases), batch_size)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=int(data_cfg["num_workers"]),
        pin_memory=bool(data_cfg["pin_memory"]),
    )


def _make_dataset(cfg: Mapping[str, Any], cases: list[dict[str, Any]], transform: Any) -> Dataset:
    data_cfg = cfg["data"]
    dataset_type = data_cfg["dataset_type"]
    if dataset_type == "cache":
        return CacheDataset(
            data=cases,
            transform=transform,
            cache_rate=float(data_cfg["cache_rate"]),
            num_workers=0,
        )
    if dataset_type == "persistent":
        cache_dir = Path(data_cfg["cache_dir"])
        cache_dir.mkdir(parents=True, exist_ok=True)
        return PersistentDataset(data=cases, transform=transform, cache_dir=str(cache_dir))
    if dataset_type == "plain":
        return Dataset(data=cases, transform=transform)
    raise ValueError(f"unknown dataset_type '{dataset_type}'")
