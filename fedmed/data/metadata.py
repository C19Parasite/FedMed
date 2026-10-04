"""Per-patient tumor volume metadata (IDs and numbers only)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import nibabel as nib
import numpy as np

from fedmed.data.transforms import detect_label_scheme
from fedmed.utils.logging import get_logger

logger = get_logger(__name__)


def compute_and_save_metadata(
    cases: Sequence[Mapping[str, Any]],
    metadata_path: str | Path,
    label_maps: Mapping[str, Mapping[str, list[int]]],
    label_scheme: str,
) -> dict[str, dict[str, Any]]:
    """Write ``patient_id -> {tumor_voxels, wt_fraction, shape}``.

    ``tumor_voxels`` is the WT voxel count on the raw label using the resolved
    scheme map. ``wt_fraction`` is WT voxels divided by spatial volume.
    ``shape`` is the raw label spatial shape as a list.

    Args:
        cases: Discovery dicts with ``patient_id`` and ``label`` (path or list).
        metadata_path: JSON destination (path is not logged).
        label_maps: Config ``data.label_maps``.
        label_scheme: ``auto``, ``msd``, or ``brats``.

    Returns:
        The metadata mapping that was written.
    """
    metadata: dict[str, dict[str, Any]] = {}
    for case in cases:
        patient_id = str(case["patient_id"])
        label_ref = case["label"]
        label_path = Path(label_ref[0] if isinstance(label_ref, list) else label_ref)
        label = np.asanyarray(nib.load(str(label_path)).dataobj)
        if label.ndim == 4 and label.shape[-1] == 1:
            label = label[..., 0]
        scheme = label_scheme
        if scheme == "auto":
            scheme = detect_label_scheme(label)
        wt_values = label_maps[scheme]["wt"]
        wt_mask = np.isin(label, wt_values)
        tumor_voxels = int(wt_mask.sum())
        volume = int(np.prod(label.shape))
        wt_fraction = float(tumor_voxels / volume) if volume else 0.0
        metadata[patient_id] = {
            "tumor_voxels": tumor_voxels,
            "wt_fraction": wt_fraction,
            "shape": [int(s) for s in label.shape],
        }
    dest = Path(metadata_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    logger.info("wrote metadata n_patients=%s", len(metadata))
    return metadata
