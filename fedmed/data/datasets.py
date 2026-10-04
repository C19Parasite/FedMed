"""Discover 4-modality MRI cases from configurable sources and build datalists."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from fedmed.data.synthetic import generate_synthetic_dataset
from fedmed.utils.logging import get_logger

logger = get_logger(__name__)

_T1CE_HINTS = ("t1ce", "t1gd", "t1c")
_T1_HINTS = ("t1",)
_T2_HINTS = ("t2",)
_FLAIR_HINTS = ("flair", "fla")
_SEG_HINTS = ("seg", "label")


def build_datalist(root: str | Path) -> list[dict[str, Any]]:
    """Scan a dataset root directory and return a list of ``{"image": path, "label": path}`` dicts.

    Supports MSD dataset.json, stacked 4D NIfTI subfolders, and multi-file BraTS folders.

    Args:
        root: Root directory containing BraTS/MSD patient folders or dataset.json.

    Returns:
        List of dictionaries with keys "image" and "label".
    """
    root_path = Path(root)
    dataset_json = root_path / "dataset.json"
    if dataset_json.is_file():
        with dataset_json.open("r", encoding="utf-8") as handle:
            spec = json.load(handle)
        datalist = []
        for item in spec.get("training", []):
            datalist.append(
                {
                    "patient_id": Path(str(item["image"])).name.split(".")[0],
                    "image": str(root_path / item["image"]),
                    "label": str(root_path / item["label"]),
                }
            )
        return datalist

    # Check for direct case subdirectories (MSD or BraTS structure)
    msd_cases = _list_msd_case_dirs(root_path)
    if msd_cases:
        return msd_cases

    brats_cases = discover_brats_local(root_path)
    if brats_cases:
        return brats_cases

    return []


def discover_cases(cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return ``{patient_id, image, label}`` dicts for the configured source.

    Paths stay on the dicts for MONAI loaders and are never logged.

    Args:
        cfg: Validated config mapping.
    """
    source = cfg["data"]["source"]
    if source == "synthetic":
        return discover_synthetic(cfg)
    if source == "brats_local":
        return discover_brats_local(Path(cfg["data"]["root_dir"]))
    if source == "decathlon":
        return discover_decathlon(cfg)
    raise ValueError(f"unknown data.source '{source}'")


def discover_synthetic(cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Ensure an MSD-style synthetic dataset exists, then list cases."""
    data = cfg["data"]
    root = Path(data["root_dir"])
    syn = data["synthetic"]
    generate_synthetic_dataset(
        root_dir=root,
        num_patients=int(syn["num_patients"]),
        spatial_size=tuple(syn["spatial_size"]),
        seed=int(cfg["seed"]),
        layout="msd",
        label_scheme="msd" if data["label_scheme"] != "brats" else "brats",
    )
    cases = _list_msd_case_dirs(root)
    logger.info("discovered synthetic cases n=%s", len(cases))
    return cases


def discover_brats_local(root_dir: Path) -> list[dict[str, Any]]:
    """Scan case folders for T1, T1ce, T2, FLAIR, and a segmentation."""
    root = Path(root_dir)
    cases: list[dict[str, Any]] = []
    if not root.is_dir():
        logger.info("brats_local root missing n=0")
        return cases
    for case_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        files = [p for p in case_dir.iterdir() if p.is_file() and _is_nifti(p)]
        t1ce = _pick(files, _T1CE_HINTS, exclude=())
        t1 = _pick(files, _T1_HINTS, exclude=_T1CE_HINTS)
        t2 = _pick(files, _T2_HINTS, exclude=())
        flair = _pick(files, _FLAIR_HINTS, exclude=())
        seg = _pick(files, _SEG_HINTS, exclude=())
        if None in (t1, t1ce, t2, flair, seg):
            continue
        cases.append(
            {
                "patient_id": case_dir.name,
                "image": [str(t1), str(t1ce), str(t2), str(flair)],
                "label": str(seg),
            }
        )
    logger.info("discovered brats_local cases n=%s", len(cases))
    return cases


def discover_decathlon(cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    """List Task01 cases from ``dataset.json``, optionally via DecathlonDataset.

    Download is gated by ``data.download``. Tests never set that flag.
    Patient IDs are basenames without suffixes, not filesystem paths.
    """
    data = cfg["data"]
    root = Path(data["root_dir"])
    task = str(data["task"])
    task_dir = root / task
    if bool(data["download"]):
        from monai.apps import DecathlonDataset

        DecathlonDataset(
            root_dir=str(root),
            task=task,
            section="training",
            download=True,
            seed=int(cfg["seed"]),
        )
    dataset_json = task_dir / "dataset.json"
    if not dataset_json.is_file():
        raise FileNotFoundError(
            f"Decathlon dataset.json missing for task={task}; set data.download true to fetch"
        )
    with dataset_json.open("r", encoding="utf-8") as handle:
        spec = json.load(handle)
    cases: list[dict[str, Any]] = []
    for item in spec.get("training", []):
        image_rel = item["image"]
        label_rel = item["label"]
        patient_id = Path(str(image_rel)).name.split(".")[0]
        cases.append(
            {
                "patient_id": patient_id,
                "image": str(task_dir / image_rel),
                "label": str(task_dir / label_rel),
            }
        )
    logger.info("discovered decathlon cases n=%s task=%s", len(cases), task)
    return cases


def _list_msd_case_dirs(root: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    if not root.is_dir():
        return cases
    for case_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        image = case_dir / "image.nii.gz"
        label = case_dir / "label.nii.gz"
        if image.is_file() and label.is_file():
            cases.append(
                {
                    "patient_id": case_dir.name,
                    "image": str(image),
                    "label": str(label),
                }
            )
    return cases


def _is_nifti(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith(".nii") or name.endswith(".nii.gz")


def _pick(
    files: list[Path],
    hints: tuple[str, ...],
    exclude: tuple[str, ...],
) -> Path | None:
    matches: list[Path] = []
    for path in files:
        stem = path.name.lower()
        if any(token in stem for token in exclude):
            continue
        if any(token in stem for token in hints):
            matches.append(path)
    if not matches:
        return None
    return sorted(matches, key=lambda p: p.name)[0]


# Re-export get_loaders for convenient access from datasets module
from fedmed.data.loaders import get_loaders  # noqa: E402

__all__ = [
    "build_datalist",
    "discover_brats_local",
    "discover_cases",
    "discover_decathlon",
    "discover_synthetic",
    "get_loaders",
]
