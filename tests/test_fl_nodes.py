"""Unit and integration tests for Flower federated learning nodes."""

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import time
import flwr as fl
import pytest
from monai.data import DataLoader

from fedmed.data.loaders import _make_dataset
from fedmed.data.synthetic import generate_synthetic_dataset
from fedmed.data.transforms import get_eval_transforms, get_train_transforms
from fedmed.fl.client import HospitalClient
from fedmed.fl.server import partition_cases_for_hospitals, start_fedmed_server
from fedmed.fl.strategy import FedMedFedAvg, aggregate_eval_metrics
from fedmed.models.unet3d import UNet3D
from fedmed.utils.config import load_config


def test_partition_cases_for_hospitals() -> None:
    """Verify patient cases are evenly partitioned across 3 hospital nodes."""
    cases = [{"patient_id": f"p_{i}"} for i in range(9)]
    partitions = partition_cases_for_hospitals(cases, num_hospitals=3)
    assert len(partitions) == 3
    assert len(partitions[0]) == 3
    assert len(partitions[1]) == 3
    assert len(partitions[2]) == 3

    # Ensure no patient overlap across hospital silos
    s0 = {c["patient_id"] for c in partitions[0]}
    s1 = {c["patient_id"] for c in partitions[1]}
    s2 = {c["patient_id"] for c in partitions[2]}
    assert len(s0 & s1) == 0
    assert len(s1 & s2) == 0


def test_aggregate_eval_metrics() -> None:
    """Verify weighted average metric aggregation across hospital silos."""
    metrics_list = [
        (10, {"dice_mean": 0.80, "dice_tc": 0.70, "dice_wt": 0.85, "dice_et": 0.85}),
        (20, {"dice_mean": 0.90, "dice_tc": 0.85, "dice_wt": 0.95, "dice_et": 0.90}),
    ]
    agg = aggregate_eval_metrics(metrics_list)
    # Total examples = 30; expected mean = (10*0.80 + 20*0.90)/30 = 26/30 ≈ 0.8667
    assert agg["dice_mean"] == pytest.approx(0.8667, rel=1e-3)
    assert "dice_tc" in agg
    assert "dice_wt" in agg
    assert "dice_et" in agg


def test_hospital_client_fit_and_evaluate(tmp_path: Path) -> None:
    """Verify HospitalClient executes fit and evaluate cycles locally."""
    p_ids = generate_synthetic_dataset(tmp_path, 2, (16, 16, 16), seed=77, layout="msd")
    cases = [
        {
            "patient_id": pid,
            "image": str(tmp_path / pid / "image.nii.gz"),
            "label": str(tmp_path / pid / "label.nii.gz"),
        }
        for pid in p_ids
    ]
    cfg = load_config("configs/baseline.yaml")
    cfg["data"]["roi_size"] = [16, 16, 16]

    train_ds = _make_dataset(cfg, cases, get_train_transforms(cfg))
    val_ds = _make_dataset(cfg, cases, get_eval_transforms(cfg))

    train_loader = DataLoader(train_ds, batch_size=2)
    val_loader = DataLoader(val_ds, batch_size=1)

    model = UNet3D(
        in_channels=4,
        out_channels=3,
        channels=(4, 8),
        strides=(2,),
        norm="INSTANCE",
    )

    client = HospitalClient(
        hospital_id="test_hospital",
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        device="cpu",
    )

    params = client.get_parameters(config={})
    assert len(params) > 0
    assert isinstance(params[0], object)

    new_params, num_samples, metrics = client.fit(params, config={"local_epochs": 1})
    assert num_samples > 0
    assert "train_loss" in metrics

    loss, eval_samples, eval_metrics = client.evaluate(new_params, config={})
    assert eval_samples > 0
    assert "dice_mean" in eval_metrics
