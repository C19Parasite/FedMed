"""Unit tests for configuration loading and validation."""

from pathlib import Path
import pytest

from fedmed.utils.config import ConfigError, load_config, validate_config


def test_load_baseline_config() -> None:
    """Verify that configs/baseline.yaml loads and passes validation."""
    cfg_path = Path("configs/baseline.yaml")
    assert cfg_path.is_file(), "baseline.yaml must exist"
    cfg = load_config(cfg_path)
    assert cfg["seed"] == 42
    assert "data" in cfg
    assert "model" in cfg
    assert "training" in cfg
    assert "logging" in cfg


def test_config_missing_required_key(tmp_path: Path) -> None:
    """Verify that omitting required keys raises ConfigError."""
    bad_cfg = tmp_path / "bad.yaml"
    bad_cfg.write_text("seed: 42\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(bad_cfg)


def test_config_bad_split_ratios(tmp_path: Path) -> None:
    """Verify that split ratios not summing to 1.0 raise ConfigError."""
    baseline = load_config("configs/baseline.yaml")
    baseline["data"]["train_ratio"] = 0.5
    baseline["data"]["val_ratio"] = 0.2
    baseline["data"]["test_ratio"] = 0.2  # sums to 0.9
    with pytest.raises(ConfigError):
        validate_config(baseline)
