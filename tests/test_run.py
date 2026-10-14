"""Unit tests for run directory management, config hash, and environment info."""

from __future__ import annotations

import json
from pathlib import Path
import pytest
import yaml

from fedmed.training.run import (
    config_hash,
    create_run_dir,
    environment_info,
    save_environment_info,
    save_resolved_config,
)
from fedmed.utils.config import load_config


def test_config_hash_consistency() -> None:
    """Verify config_hash is 12 hex characters and deterministic regardless of key order."""
    cfg1 = {"b": 2, "a": 1, "nested": {"y": "hello", "x": [1, 2, 3]}}
    cfg2 = {"a": 1, "nested": {"x": [1, 2, 3], "y": "hello"}, "b": 2}

    h1 = config_hash(cfg1)
    h2 = config_hash(cfg2)

    assert len(h1) == 12
    assert h1 == h2


def test_environment_info_sanitization() -> None:
    """Verify environment_info does not contain usernames, hostnames, or paths."""
    info = environment_info()
    required_keys = {
        "python",
        "torch",
        "monai",
        "numpy",
        "cuda_available",
        "cuda_version",
        "cudnn_version",
        "gpu_name",
        "gpu_count",
        "platform",
    }
    assert required_keys.issubset(info.keys())

    # String representation checks: no filesystem paths or usernames
    serialized = json.dumps(info)
    assert "Users" not in serialized
    assert "home" not in serialized
    assert "C:\\" not in serialized
    assert "/" not in info["python"]


def test_create_run_dir_structure(tmp_path: Path) -> None:
    """Verify create_run_dir builds required subfolders and saves config & environment info."""
    cfg = load_config("configs/smoke_test.yaml")
    cfg["training"]["logging"]["output_dir"] = str(tmp_path / "runs")
    cfg["training"]["logging"]["run_name"] = "test_run"

    run_dir = create_run_dir(cfg)
    assert run_dir.is_dir()
    assert (run_dir / "checkpoints").is_dir()
    assert (run_dir / "tensorboard").is_dir()
    assert (run_dir / "results").is_dir()

    # Config and environment files
    cfg_file = run_dir / "config.yaml"
    env_file = run_dir / "environment.json"
    assert cfg_file.is_file()
    assert env_file.is_file()

    loaded_cfg = yaml.safe_load(cfg_file.read_text(encoding="utf-8"))
    assert loaded_cfg["seed"] == cfg["seed"]

    loaded_env = json.loads(env_file.read_text(encoding="utf-8"))
    assert "torch" in loaded_env


def test_create_run_dir_resume_auto(tmp_path: Path) -> None:
    """Verify resume='auto' reuses the most recent existing run directory."""
    cfg = load_config("configs/smoke_test.yaml")
    cfg["training"]["logging"]["output_dir"] = str(tmp_path / "runs")
    cfg["training"]["logging"]["run_name"] = "resume_run"

    # Create first run
    run1 = create_run_dir(cfg)
    assert run1.is_dir()

    # Now enable resume="auto"
    cfg["training"]["resume"]["enabled"] = True
    cfg["training"]["resume"]["path"] = "auto"

    run2 = create_run_dir(cfg)
    assert run1 == run2
