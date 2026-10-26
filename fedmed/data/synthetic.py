"""Synthetic 4-modality MRI plus segmentation for tests and smoke runs."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import nibabel as nib
import numpy as np

from fedmed.utils.logging import get_logger

logger = get_logger(__name__)

Layout = Literal["msd", "brats"]


def generate_synthetic_dataset(
    root_dir: str | Path,
    num_patients: int,
    spatial_size: tuple[int, int, int] | list[int],
    seed: int,
    layout: Layout = "msd",
    label_scheme: Literal["msd", "brats"] = "msd",
) -> list[str]:
    """Write a tiny synthetic cohort and return patient IDs.

    Args:
        root_dir: Directory that will contain one folder per patient.
        num_patients: Number of cases to write.
        spatial_size: ``(D, H, W)`` voxel grid.
        seed: RNG seed (also logged).
        layout: ``msd`` writes stacked 4D image NIfTI; ``brats`` writes
            four modality files plus ``seg``.
        label_scheme: Raw label value convention.

    Returns:
        Sorted patient IDs (no paths).
    """
    root = Path(root_dir)
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    ids: list[str] = []
    depth, height, width = (int(x) for x in spatial_size)
    for index in range(num_patients):
        patient_id = f"synthetic_{index:03d}"
        case_dir = root / patient_id
        case_dir.mkdir(parents=True, exist_ok=True)
        image, label = _random_volume(
            rng, depth, height, width, label_scheme=label_scheme
        )
        if layout == "msd":
            img_p = case_dir / "image.nii.gz"
            lbl_p = case_dir / "label.nii.gz"
            if not (img_p.exists() and lbl_p.exists() and img_p.stat().st_size > 100):
                _save_nifti(img_p, image)
                _save_nifti(lbl_p, label)
        else:
            seg_p = case_dir / f"{patient_id}_seg.nii.gz"
            if not (seg_p.exists() and seg_p.stat().st_size > 100):
                for channel, name in enumerate(("t1", "t1ce", "t2", "flair")):
                    _save_nifti(case_dir / f"{patient_id}_{name}.nii.gz", image[channel])
                _save_nifti(seg_p, label)
        ids.append(patient_id)
    logger.info(
        "wrote synthetic dataset n_patients=%s layout=%s seed=%s",
        num_patients,
        layout,
        seed,
    )
    return ids


def _random_volume(
    rng: np.random.Generator,
    depth: int,
    height: int,
    width: int,
    label_scheme: Literal["msd", "brats"],
) -> tuple[np.ndarray, np.ndarray]:
    image = rng.normal(loc=100.0, scale=25.0, size=(4, depth, height, width)).astype(
        np.float32
    )
    label = np.zeros((depth, height, width), dtype=np.uint8)
    # Compact tumor blob so WT/TC/ET are all present.
    d0, h0, w0 = depth // 4, height // 4, width // 4
    d1, h1, w1 = max(d0 + 2, depth // 2), max(h0 + 2, height // 2), max(w0 + 2, width // 2)
    label[d0:d1, h0:h1, w0:w1] = 1
    label[d0 + 1 : d1, h0 + 1 : h1, w0 + 1 : w1] = 2
    label[d0 + 2 : max(d0 + 3, d1 - 1), h0 + 2 : max(h0 + 3, h1 - 1), w0 + 2 : max(w0 + 3, w1 - 1)] = (
        3 if label_scheme == "msd" else 4
    )
    if label_scheme == "msd":
        # 1=ED, 2=ET, 3=NCR (MONAI-compatible defaults).
        pass
    else:
        # Native BraTS: 1=NCR, 2=ED, 4=ET. Remap the outer/mid rings.
        remapped = np.zeros_like(label)
        remapped[label == 1] = 2
        remapped[label == 2] = 1
        remapped[label == 4] = 4
        label = remapped
    image[:, label > 0] += 40.0
    return image, label


def _save_nifti(path: Path, array: np.ndarray) -> None:
    # nibabel expects spatial axes last for 3D; 4D stacked as (H,W,D,C) vs (C,D,H,W).
    # Store channel-first MSD images as (D, H, W, C) on disk (MONAI LoadImaged +
    # EnsureChannelFirstd with channel_dim=-1), or 3D volumes as (D, H, W).
    if array.ndim == 4:
        disk = np.transpose(array, (1, 2, 3, 0))
    else:
        disk = array
    nib.save(nib.Nifti1Image(disk, np.eye(4)), str(path))
