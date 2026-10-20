"""MONAI dictionary transforms for BraTS-style 4-modality MRI."""

from __future__ import annotations

from typing import Any, Hashable, Mapping

import numpy as np
import torch
from monai.transforms import (
    Compose,
    DeleteItemsd,
    EnsureChannelFirstd,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    Spacingd,
)

from fedmed.utils.logging import get_logger

logger = get_logger(__name__)

IMAGE_KEY = "image"
LABEL_KEY = "label"
CROP_LABEL_KEY = "crop_label"


def detect_label_scheme(label: np.ndarray | torch.Tensor) -> str:
    """Return ``brats`` if any voxel equals 4, otherwise ``msd``.

    Args:
        label: Integer segmentation array or tensor.
    """
    values = label.detach().cpu().numpy() if isinstance(label, torch.Tensor) else np.asarray(label)
    if np.any(values == 4):
        return "brats"
    return "msd"


class ConvertToBratsChannelsd(MapTransform):
    """Convert a single-channel BraTS/MSD label to overlapping TC, WT, ET.

    Channel order is always ``(TC, WT, ET)``.

    Label maps come from YAML ``data.label_maps.<scheme>``. Default MSD maps
    follow MONAI's helper (ET = 2, TC = {2, 3}, WT = {1, 2, 3}).

    The Medical Segmentation Decathlon ``dataset.json`` for Task01 documents
    label 3 as enhancing tumor, while MONAI's
    ``ConvertToMultiChannelBasedOnBratsClassesd`` treats label 2 as ET. FedMed
    uses the MONAI-compatible default as a deliberate, configurable choice to
    verify against the real data.
    """

    def __init__(
        self,
        keys: str | tuple[str, ...] = LABEL_KEY,
        label_maps: Mapping[str, Mapping[str, list[int]]] | None = None,
        scheme: str = "auto",
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys=allow_missing_keys)
        self.label_maps = label_maps or {
            "msd": {"tc": [2, 3], "wt": [1, 2, 3], "et": [2]},
            "brats": {"tc": [1, 4], "wt": [1, 2, 4], "et": [4]},
        }
        self.scheme = scheme

    def __call__(self, data: Mapping[Hashable, Any]) -> dict[Hashable, Any]:
        d = dict(data)
        for key in self.key_iterator(d):
            label = d[key]
            is_tensor = isinstance(label, torch.Tensor)
            array = label.detach().cpu().numpy() if is_tensor else np.asarray(label)
            scheme = self.scheme
            if scheme == "auto":
                scheme = detect_label_scheme(array)
            mapping = self.label_maps[scheme]
            squeezed = array
            if squeezed.ndim == 4 and squeezed.shape[0] == 1:
                squeezed = squeezed[0]
            tc = self._mask(squeezed, mapping["tc"])
            wt = self._mask(squeezed, mapping["wt"])
            et = self._mask(squeezed, mapping["et"])
            stacked = np.stack([tc, wt, et], axis=0).astype(np.float32)
            if is_tensor:
                out_tensor = torch.from_numpy(stacked)
                if hasattr(label, "meta") and getattr(label, "meta", None):
                    from monai.data import MetaTensor

                    out_tensor = MetaTensor(out_tensor, meta=label.meta)
                d[key] = out_tensor
            else:
                d[key] = stacked
        return d

    @staticmethod
    def _mask(label: np.ndarray, values: list[int]) -> np.ndarray:
        mask = np.zeros(label.shape, dtype=np.float32)
        for value in values:
            mask[label == value] = 1.0
        return mask


class CopyWtAsCropLabeld(MapTransform):
    """Copy the WT channel into ``crop_label`` for RandCropByPosNegLabeld."""

    def __init__(self, keys: str | tuple[str, ...] = LABEL_KEY) -> None:
        super().__init__(keys)

    def __call__(self, data: Mapping[Hashable, Any]) -> dict[Hashable, Any]:
        d = dict(data)
        label = d[LABEL_KEY]
        if isinstance(label, torch.Tensor):
            d[CROP_LABEL_KEY] = label[1:2]
        else:
            d[CROP_LABEL_KEY] = np.asarray(label)[1:2]
        return d


def get_train_transforms(cfg: Mapping[str, Any]) -> Compose:
    """Training Compose: load, resample, convert labels, normalize, crop, aug.

    Args:
        cfg: Validated config mapping.
    """
    data = cfg["data"]
    keys = (IMAGE_KEY, LABEL_KEY)
    return Compose(
        [
            LoadImaged(keys=keys),
            EnsureChannelFirstd(keys=keys),
            Orientationd(keys=keys, axcodes=data["orientation"], labels=None),
            Spacingd(
                keys=keys,
                pixdim=tuple(data["spacing"]),
                mode=("bilinear", "nearest"),
            ),
            ConvertToBratsChannelsd(
                keys=LABEL_KEY,
                label_maps=data["label_maps"],
                scheme=data["label_scheme"],
            ),
            NormalizeIntensityd(keys=IMAGE_KEY, nonzero=True, channel_wise=True),
            CopyWtAsCropLabeld(),
            RandCropByPosNegLabeld(
                keys=keys,
                label_key=CROP_LABEL_KEY,
                spatial_size=tuple(data["roi_size"]),
                pos=float(data["pos"]),
                neg=float(data["neg"]),
                num_samples=int(data["num_samples"]),
                image_key=IMAGE_KEY,
            ),
            DeleteItemsd(keys=CROP_LABEL_KEY),
            RandFlipd(keys=keys, spatial_axis=[0, 1, 2], prob=float(data["flip_prob"])),
            RandScaleIntensityd(
                keys=IMAGE_KEY, factors=float(data["scale_intensity_factor"]), prob=1.0
            ),
            RandShiftIntensityd(
                keys=IMAGE_KEY, offsets=float(data["shift_intensity_offset"]), prob=1.0
            ),
        ]
    )


def get_eval_transforms(cfg: Mapping[str, Any]) -> Compose:
    """Deterministic val/test Compose (no random crop or intensity aug).

    Args:
        cfg: Validated config mapping.
    """
    data = cfg["data"]
    keys = (IMAGE_KEY, LABEL_KEY)
    return Compose(
        [
            LoadImaged(keys=keys),
            EnsureChannelFirstd(keys=keys),
            Orientationd(keys=keys, axcodes=data["orientation"], labels=None),
            Spacingd(
                keys=keys,
                pixdim=tuple(data["spacing"]),
                mode=("bilinear", "nearest"),
            ),
            ConvertToBratsChannelsd(
                keys=LABEL_KEY,
                label_maps=data["label_maps"],
                scheme=data["label_scheme"],
            ),
            NormalizeIntensityd(keys=IMAGE_KEY, nonzero=True, channel_wise=True),
        ]
    )
