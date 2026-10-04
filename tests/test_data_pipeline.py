"""Unit tests for BraTS MONAI data pipeline, transforms, shapes, and dataloaders."""

from pathlib import Path
import numpy as np
import pytest
import torch

from fedmed.data.datasets import build_datalist, discover_cases
from fedmed.data.loaders import build_loader, get_loaders
from fedmed.data.splits import load_or_create_splits, make_splits
from fedmed.data.synthetic import generate_synthetic_dataset
from fedmed.data.transforms import (
    ConvertToBratsChannelsd,
    get_eval_transforms,
    get_train_transforms,
)
from fedmed.utils.config import load_config


@pytest.fixture
def test_cfg(tmp_path: Path) -> dict:
    """Fixture providing a minimal fast config with temporary directory."""
    cfg = load_config("configs/baseline.yaml")
    cfg["data"]["root_dir"] = str(tmp_path / "data")
    cfg["data"]["splits_path"] = str(tmp_path / "data" / "splits.json")
    cfg["data"]["metadata_path"] = str(tmp_path / "data" / "metadata.json")
    cfg["data"]["cache_dir"] = str(tmp_path / "data" / "cache")
    cfg["data"]["roi_size"] = [32, 32, 32]
    cfg["data"]["synthetic"]["num_patients"] = 4
    cfg["data"]["synthetic"]["spatial_size"] = [32, 32, 32]
    cfg["data"]["batch_size"] = 2
    cfg["data"]["val_batch_size"] = 1
    return cfg


def test_synthetic_nifti_volume_shapes_and_channels(tmp_path: Path, test_cfg: dict) -> None:
    """Verify transforms produce image [4, D, H, W] and label [3, D, H, W] (TC, WT, ET)."""
    cases = discover_cases(test_cfg)
    assert len(cases) == 4
    first_case = cases[0]

    # Test deterministic evaluation transforms (preserves spatial shape)
    eval_tf = get_eval_transforms(test_cfg)
    eval_out = eval_tf(first_case)
    img_eval = eval_out["image"]
    lbl_eval = eval_out["label"]

    # Image shape: [4, D, H, W], Label shape: [3, D, H, W]
    assert img_eval.shape == (4, 32, 32, 32), f"Expected (4, 32, 32, 32), got {img_eval.shape}"
    assert lbl_eval.shape == (3, 32, 32, 32), f"Expected (3, 32, 32, 32), got {lbl_eval.shape}"
    assert isinstance(img_eval, torch.Tensor)
    assert isinstance(lbl_eval, torch.Tensor)

    # Test training transforms with spatial cropping
    train_tf = get_train_transforms(test_cfg)
    train_out = train_tf(first_case)
    # Train out might be a list if num_samples > 1, or single item
    sample = train_out[0] if isinstance(train_out, list) else train_out
    assert sample["image"].shape == (4, 32, 32, 32)
    assert sample["label"].shape == (3, 32, 32, 32)


def test_convert_to_brats_channels_mapping() -> None:
    """Test ConvertToBratsChannelsd properly maps discrete labels to TC, WT, ET."""
    # MSD schema: 1=ED, 2=ET, 3=NCR. TC={2,3}, WT={1,2,3}, ET={2}
    raw_label = np.zeros((1, 10, 10, 10), dtype=np.uint8)
    raw_label[0, 1:3, 1:3, 1:3] = 1  # Edema
    raw_label[0, 3:5, 3:5, 3:5] = 2  # Enhancing Tumor
    raw_label[0, 5:7, 5:7, 5:7] = 3  # Necrotic Core

    transform = ConvertToBratsChannelsd(keys="label", scheme="msd")
    out = transform({"label": raw_label})
    channels = out["label"]

    assert channels.shape == (3, 10, 10, 10)
    tc, wt, et = channels[0], channels[1], channels[2]

    # TC should include 2 and 3
    assert tc.sum() == (2 * 2 * 2) + (2 * 2 * 2)
    # WT should include 1, 2, and 3
    assert wt.sum() == (2 * 2 * 2) * 3
    # ET should include only 2
    assert et.sum() == (2 * 2 * 2)


def test_build_datalist_scans_correctly(tmp_path: Path) -> None:
    """Verify build_datalist returns list of image and label dicts."""
    p_ids = generate_synthetic_dataset(tmp_path, 3, (16, 16, 16), seed=99, layout="msd")
    datalist = build_datalist(tmp_path)
    assert len(datalist) == 3
    for item in datalist:
        assert "image" in item
        assert "label" in item
        assert Path(item["image"]).is_file()
        assert Path(item["label"]).is_file()


def test_patient_id_splits_no_leakage() -> None:
    """Verify train, val, and test splits have zero overlap and match configured ratios."""
    patient_ids = [f"patient_{i:03d}" for i in range(20)]
    splits = make_splits(
        patient_ids=patient_ids,
        seed=42,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
    )

    train_set = set(splits["train"])
    val_set = set(splits["val"])
    test_set = set(splits["test"])

    assert len(train_set) == 14
    assert len(val_set) == 3
    assert len(test_set) == 3
    # Assert disjoint
    assert len(train_set & val_set) == 0
    assert len(train_set & test_set) == 0
    assert len(val_set & test_set) == 0


def test_dataloaders_batch_shapes(test_cfg: dict) -> None:
    """Verify DataLoaders emit proper 5D batched tensors [B, C, D, H, W]."""
    loaders = get_loaders(test_cfg)
    train_loader = loaders["train"]
    val_loader = loaders["val"]

    batch = next(iter(train_loader))
    assert batch["image"].ndim == 5
    assert batch["image"].shape[1] == 4  # 4 MRI modalities
    assert batch["label"].ndim == 5
    assert batch["label"].shape[1] == 3  # TC, WT, ET

    val_batch = next(iter(val_loader))
    assert val_batch["image"].ndim == 5
    assert val_batch["image"].shape[1] == 4
    assert val_batch["label"].shape[1] == 3
