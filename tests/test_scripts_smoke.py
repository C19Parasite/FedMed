"""End-to-end smoke and privacy test for train_baseline.py and evaluate.py scripts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys
import pytest
import yaml

from fedmed.training.csv_logger import METRICS_CSV_COLUMNS


def test_scripts_smoke_end_to_end_and_privacy(tmp_path: Path) -> None:
    """Run train_baseline.py --smoke end-to-end, verify outputs, schema, privacy, and evaluate.py."""
    data_dir = tmp_path / "data"
    runs_dir = tmp_path / "runs"
    data_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    # Prepare temporary smoke config pointing to temporary data and run directories
    repo_root = Path(__file__).resolve().parent.parent
    base_smoke_path = repo_root / "configs" / "smoke_test.yaml"
    with base_smoke_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg["data"]["root_dir"] = str(data_dir)
    cfg["data"]["splits_path"] = str(data_dir / "splits.json")
    cfg["data"]["metadata_path"] = str(data_dir / "metadata.json")
    cfg["data"]["cache_dir"] = str(data_dir / "cache")
    cfg["training"]["logging"]["output_dir"] = str(runs_dir)
    cfg["training"]["logging"]["run_name"] = "e2e_smoke"

    test_config_path = tmp_path / "smoke_run_cfg.yaml"
    with test_config_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)

    # 1. Run train_baseline.py via subprocess
    cmd_train = [
        sys.executable,
        str(repo_root / "scripts" / "train_baseline.py"),
        "--config",
        str(test_config_path),
        "--smoke",
    ]
    proc_train = subprocess.run(
        cmd_train,
        cwd=str(repo_root),
        capture_output=True,
        text=True,
    )
    assert proc_train.returncode == 0, f"train_baseline.py failed:\n{proc_train.stderr}\n{proc_train.stdout}"

    # 2. Locate generated run directory
    run_base = runs_dir / "e2e_smoke"
    assert run_base.is_dir()
    run_subdirs = [d for d in run_base.iterdir() if d.is_dir()]
    assert len(run_subdirs) == 1
    run_dir = run_subdirs[0]

    # Verify best.pt and latest.pt
    best_pt = run_dir / "checkpoints" / "best.pt"
    latest_pt = run_dir / "checkpoints" / "latest.pt"
    assert best_pt.is_file(), "best.pt must be created"
    assert latest_pt.is_file(), "latest.pt must be created"

    # Verify metrics.csv
    csv_file = run_dir / "results" / "metrics.csv"
    assert csv_file.is_file(), "metrics.csv must be created"
    with csv_file.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        data_rows = list(reader)

    assert header == METRICS_CSV_COLUMNS
    assert len(data_rows) == 2, f"Expected 2 data rows for 2 epochs, got {len(data_rows)}"

    # Verify baseline_metrics.json schema
    baseline_json_path = run_dir / "results" / "baseline_metrics.json"
    assert baseline_json_path.is_file(), "baseline_metrics.json must be created"
    baseline_record = json.loads(baseline_json_path.read_text(encoding="utf-8"))

    expected_schema_keys = {
        "split",
        "checkpoint_epoch",
        "n_volumes",
        "dice",
        "hd95",
        "empty_policy",
        "threshold",
        "sliding_window",
        "model",
        "training",
        "seed",
        "config_hash",
        "environment",
        "timestamp_utc",
    }
    assert expected_schema_keys.issubset(baseline_record.keys())
    assert baseline_record["split"] == "test"
    assert "mean" in baseline_record["dice"]
    assert "tc" in baseline_record["dice"]
    assert "wt" in baseline_record["dice"]
    assert "et" in baseline_record["dice"]

    # 3. Privacy Rule Check: No tmp data paths or patient IDs in output files or logs
    data_dir_str = str(data_dir).replace("\\", "/")
    all_output_text = (
        baseline_json_path.read_text(encoding="utf-8")
        + csv_file.read_text(encoding="utf-8")
        + (run_dir / "environment.json").read_text(encoding="utf-8")
    )
    # Check that data paths are not leaked
    assert data_dir_str not in all_output_text
    # Check that patient IDs are not leaked
    assert "synthetic_patient_" not in all_output_text
    assert "patient_" not in all_output_text

    # 4. Run evaluate.py on the produced best.pt
    cmd_eval = [
        sys.executable,
        str(repo_root / "scripts" / "evaluate.py"),
        "--config",
        str(test_config_path),
        "--checkpoint",
        str(best_pt),
        "--split",
        "test",
    ]
    proc_eval = subprocess.run(
        cmd_eval,
        cwd=str(repo_root),
        capture_output=True,
        text=True,
    )
    assert proc_eval.returncode == 0, f"evaluate.py failed:\n{proc_eval.stderr}\n{proc_eval.stdout}"

    # Load evaluated metrics from results/test_metrics.json
    eval_metrics_path = repo_root / "results" / "test_metrics.json"
    assert eval_metrics_path.is_file()
    eval_record = json.loads(eval_metrics_path.read_text(encoding="utf-8"))

    # Assert that evaluated metrics match baseline_metrics.json within 1e-6
    for key in ("tc", "wt", "et", "mean"):
        b_val = baseline_record["dice"][key]
        e_val = eval_record["dice"][key]
        assert b_val == pytest.approx(e_val, abs=1e-6)
