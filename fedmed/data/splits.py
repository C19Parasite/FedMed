"""Deterministic patient-ID splits persisted as JSON (IDs only)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from fedmed.utils.logging import get_logger

logger = get_logger(__name__)

SPLIT_KEYS = ("train", "val", "test")


def load_or_create_splits(
    patient_ids: Sequence[str],
    splits_path: str | Path,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> dict[str, Any]:
    """Reuse ``splits.json`` when IDs match; otherwise regenerate and warn.

    Args:
        patient_ids: All discovered patient IDs.
        splits_path: Destination JSON path (not logged).
        seed: Shuffle seed.
        train_ratio: Fraction for train.
        val_ratio: Fraction for val.
        test_ratio: Fraction for test.

    Returns:
        Mapping with ``seed``, ``train``, ``val``, ``test`` ID lists.
    """
    path = Path(splits_path)
    unique_ids = sorted(set(patient_ids))
    if path.is_file():
        with path.open("r", encoding="utf-8") as handle:
            existing = json.load(handle)
        existing_ids = _collect_ids(existing)
        if existing_ids == set(unique_ids):
            logger.info(
                "reusing splits n_train=%s n_val=%s n_test=%s",
                len(existing.get("train", [])),
                len(existing.get("val", [])),
                len(existing.get("test", [])),
            )
            return existing
        logger.warning(
            "splits ID set mismatch (stored=%s current=%s); regenerating",
            len(existing_ids),
            len(unique_ids),
        )
    splits = make_splits(unique_ids, seed, train_ratio, val_ratio, test_ratio)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(splits, handle, indent=2)
    logger.info(
        "wrote splits n_train=%s n_val=%s n_test=%s seed=%s",
        len(splits["train"]),
        len(splits["val"]),
        len(splits["test"]),
        seed,
    )
    return splits


def make_splits(
    patient_ids: Sequence[str],
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> dict[str, Any]:
    """Shuffle sorted IDs and slice  train/val/test without overlap."""
    del test_ratio  # implied by remainder after train/val counts
    rng = __import__("random").Random(seed)
    ordered = sorted(patient_ids)
    rng.shuffle(ordered)
    n_total = len(ordered)
    n_train = int(round(n_total * train_ratio))
    n_val = int(round(n_total * val_ratio))
    if n_train + n_val > n_total:
        n_val = n_total - n_train
    n_test = n_total - n_train - n_val
    train = ordered[:n_train]
    val = ordered[n_train : n_train + n_val]
    test = ordered[n_train + n_val :]
    _assert_no_overlap(train, val, test)
    logger.info("split sizes train=%s val=%s test=%s", n_train, n_val, n_test)
    return {"seed": seed, "train": train, "val": val, "test": test}


def _collect_ids(splits: Mapping[str, Any]) -> set[str]:
    ids: set[str] = set()
    for key in SPLIT_KEYS:
        ids.update(str(item) for item in splits.get(key, []))
    return ids


def _assert_no_overlap(train: Iterable[str], val: Iterable[str], test: Iterable[str]) -> None:
    sets = {"train": set(train), "val": set(val), "test": set(test)}
    if sets["train"] & sets["val"] or sets["train"] & sets["test"] or sets["val"] & sets["test"]:
        raise ValueError("train/val/test patient IDs overlap")
