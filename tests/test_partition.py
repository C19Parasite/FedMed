"""Unit and integration tests for patient-level cross-silo data partitioning."""

from __future__ import annotations

import json
from pathlib import Path
import pytest
from monai.data import DataLoader

from fedmed.data.partition import (
    HospitalLoaders,
    create_or_load_partitions,
    get_global_eval_loaders,
    get_hospital_loaders,
    load_partition_manifest,
    partition_patients,
    print_partition_table,
    save_partition_manifest,
)
from fedmed.data.synthetic import generate_synthetic_dataset
from fedmed.utils.config import load_config


def test_partition_patient_isolation_and_union_completeness() -> None:
    """Assert: no patient ID appears in two hospitals, and union of all hospitals

    plus the global validation set equals the full training set.
    """
    full_training_set = [f"patient_{i:03d}" for i in range(20)]
    num_hospitals = 3

    manifest = partition_patients(
        patient_ids=full_training_set,
        num_hospitals=num_hospitals,
        mode="iid",
        global_val_ratio=0.15,
        seed=42,
    )

    hospitals = manifest["hospitals"]
    h1_ids = set(hospitals["hospital_1"]["patient_ids"])
    h2_ids = set(hospitals["hospital_2"]["patient_ids"])
    h3_ids = set(hospitals["hospital_3"]["patient_ids"])
    global_val_ids = set(manifest["global_val"])

    # 1. Assert no patient ID appears in two hospitals (pairwise disjoint)
    assert len(h1_ids & h2_ids) == 0, "Patient overlap detected between hospital_1 and hospital_2!"
    assert len(h1_ids & h3_ids) == 0, "Patient overlap detected between hospital_1 and hospital_3!"
    assert len(h2_ids & h3_ids) == 0, "Patient overlap detected between hospital_2 and hospital_3!"

    # 2. Assert global validation set belongs to NO hospital
    assert len(global_val_ids & h1_ids) == 0, "Global validation patient leaked into hospital_1!"
    assert len(global_val_ids & h2_ids) == 0, "Global validation patient leaked into hospital_2!"
    assert len(global_val_ids & h3_ids) == 0, "Global validation patient leaked into hospital_3!"

    # 3. Assert the union of all hospitals plus the global validation set equals the full training set
    all_hospitals_union = h1_ids | h2_ids | h3_ids
    assert all_hospitals_union | global_val_ids == set(full_training_set), (
        "Union of hospitals and global validation set does not equal full training set!"
    )


def test_partition_iid_balance() -> None:
    """Verify IID mode partitions training pool into equal/balanced counts."""
    patient_ids = [f"case_{i}" for i in range(30)]
    manifest = partition_patients(
        patient_ids=patient_ids,
        num_hospitals=3,
        mode="iid",
        global_val_ratio=0.10,  # 3 patients for global val -> 27 for training pool
        seed=42,
    )

    assert len(manifest["global_val"]) == 3
    hospitals = manifest["hospitals"]
    # 27 / 3 = 9 patients each
    assert len(hospitals["hospital_1"]["patient_ids"]) == 9
    assert len(hospitals["hospital_2"]["patient_ids"]) == 9
    assert len(hospitals["hospital_3"]["patient_ids"]) == 9


def test_partition_non_iid_proportions() -> None:
    """Verify non-IID mode partitions according to specified proportions (50% / 30% / 20%)."""
    # 20 patients, 0 global val -> 20 in training pool
    patient_ids = [f"case_{i}" for i in range(20)]
    manifest = partition_patients(
        patient_ids=patient_ids,
        num_hospitals=3,
        mode="non_iid",
        proportions=[0.50, 0.30, 0.20],
        global_val_ratio=0.0,
        seed=42,
    )

    hospitals = manifest["hospitals"]
    # Expected: 50% of 20 = 10, 30% of 20 = 6, 20% of 20 = 4
    assert len(hospitals["hospital_1"]["patient_ids"]) == 10
    assert len(hospitals["hospital_2"]["patient_ids"]) == 6
    assert len(hospitals["hospital_3"]["patient_ids"]) == 4


def test_partition_reproducibility_with_fixed_seed() -> None:
    """Verify identical seeds produce identical splits and different seeds differ."""
    patient_ids = [f"p_{i}" for i in range(25)]

    m1 = partition_patients(patient_ids, num_hospitals=3, seed=123)
    m2 = partition_patients(patient_ids, num_hospitals=3, seed=123)
    m3 = partition_patients(patient_ids, num_hospitals=3, seed=999)

    assert m1["global_val"] == m2["global_val"]
    assert m1["hospital_1"] == m2["hospital_1"]
    assert m1["hospital_2"] == m2["hospital_2"]
    assert m1["hospital_3"] == m2["hospital_3"]

    # Different seeds should produce different permutations
    assert (
        m1["hospital_1"] != m3["hospital_1"]
        or m1["global_val"] != m3["global_val"]
    )


def test_partition_manifest_json_ids_only(tmp_path: Path) -> None:
    """Verify manifest saves ONLY patient IDs, containing no file paths or image data."""
    patient_ids = [f"BraTS20_Training_{i:03d}" for i in range(12)]
    manifest = partition_patients(
        patient_ids=patient_ids,
        num_hospitals=3,
        mode="iid",
        seed=42,
    )

    manifest_file = tmp_path / "partitions" / "partition_manifest.json"
    saved_path = save_partition_manifest(manifest, manifest_file)
    assert saved_path.is_file()

    # Read back raw JSON content and check for prohibited keys/substrings
    with saved_path.open("r", encoding="utf-8") as f:
        raw_text = f.read()

    # Assert no image file paths or array data leaked
    assert ".nii" not in raw_text
    assert ".dcm" not in raw_text
    assert "data/" not in raw_text
    assert "image" not in raw_text
    assert "label" not in raw_text

    loaded = load_partition_manifest(saved_path)
    assert loaded["seed"] == 42
    assert "hospital_1" in loaded["hospitals"]
    assert len(loaded["hospitals"]["hospital_1"]["patient_ids"]) > 0
    assert len(loaded["hospital_1"]) > 0


def test_print_partition_table_runs() -> None:
    """Verify print_partition_table formats columns cleanly."""
    patient_ids = [f"p_{i}" for i in range(15)]
    manifest = partition_patients(patient_ids, num_hospitals=3, seed=42)

    table = print_partition_table(manifest)
    assert "Hospital / Set" in table
    assert "hospital_1" in table
    assert "hospital_2" in table
    assert "hospital_3" in table
    assert "Global Validation (Server)" in table


def test_get_hospital_loaders_with_synthetic_data(tmp_path: Path) -> None:
    """Verify get_hospital_loaders builds train and local-val MONAI DataLoaders."""
    # Generate 12 synthetic cases
    p_ids = generate_synthetic_dataset(tmp_path / "data", 12, (16, 16, 16), seed=42, layout="msd")

    cfg = load_config("configs/federated_hospitals.yaml")
    cfg["data"]["root_dir"] = str(tmp_path / "data")
    cfg["data"]["roi_size"] = [16, 16, 16]
    cfg["data"]["batch_size"] = 2
    cfg["data"]["val_batch_size"] = 1
    cfg["data"]["dataset_type"] = "plain"

    manifest_file = tmp_path / "partitions" / "partition_manifest.json"
    cfg["federated"]["manifest_path"] = str(manifest_file)

    manifest = create_or_load_partitions(cfg, patient_ids=p_ids, manifest_path=manifest_file, print_table=False)

    loaders = get_hospital_loaders("hospital_1", cfg, manifest=manifest)
    assert isinstance(loaders, HospitalLoaders)
    assert isinstance(loaders.train, DataLoader)
    assert isinstance(loaders.val, DataLoader)

    # Test indexing & unpacking
    train_loader, val_loader = loaders
    assert train_loader is loaders.train
    assert val_loader is loaders.val
    assert loaders["train"] is loaders.train
    assert loaders["val"] is loaders.val

    # Smoke test loading one batch from train loader
    batch = next(iter(train_loader))
    assert "image" in batch
    assert "label" in batch
    assert batch["image"].shape[1] == 4  # 4 MRI modalities


def test_src_data_partition_reexport() -> None:
    """Verify src.data.partition re-exports work transparently."""
    from src.data.partition import partition_patients as src_partition_patients
    from src.data.partition import HospitalLoaders as SrcHospitalLoaders

    assert src_partition_patients is partition_patients
    assert SrcHospitalLoaders is HospitalLoaders
