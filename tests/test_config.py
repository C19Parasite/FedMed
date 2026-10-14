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
    assert "loss" in cfg
    assert "metrics" in cfg


def test_load_base_and_smoke_configs() -> None:
    """Verify configs/base.yaml and configs/smoke_test.yaml pass validation."""
    base_cfg = load_config("configs/base.yaml")
    assert base_cfg["model"]["name"] == "monai_unet"
    assert base_cfg["data"]["roi_size"] == [96, 96, 96]

    smoke_cfg = load_config("configs/smoke_test.yaml")
    assert smoke_cfg["model"]["channels"] == [4, 8, 16]
    assert smoke_cfg["data"]["roi_size"] == [16, 16, 16]


def test_config_missing_required_key(tmp_path: Path) -> None:
    """Verify that omitting required keys raises ConfigError."""
    bad_cfg = tmp_path / "bad.yaml"
    bad_cfg.write_text("seed: 42\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(bad_cfg)


def test_config_bad_split_ratios() -> None:
    """Verify that split ratios not summing to 1.0 raise ConfigError."""
    baseline = load_config("configs/baseline.yaml")
    baseline["data"]["train_ratio"] = 0.5
    baseline["data"]["val_ratio"] = 0.2
    baseline["data"]["test_ratio"] = 0.2  # sums to 0.9
    with pytest.raises(ConfigError):
        validate_config(baseline)


def test_config_invalid_model_name() -> None:
    """Verify unknown model name raises ConfigError."""
    cfg = load_config("configs/baseline.yaml")
    cfg["model"]["name"] = "unknown_net"
    with pytest.raises(ConfigError, match="unknown model.name"):
        validate_config(cfg)


def test_config_batchnorm_prohibited() -> None:
    """Verify BatchNorm in config raises ConfigError with informative message."""
    cfg = load_config("configs/baseline.yaml")
    cfg["model"]["norm"] = {"type": "batch"}
    with pytest.raises(ConfigError, match="BatchNorm running statistics diverge"):
        validate_config(cfg)


def test_config_group_norm_divisibility() -> None:
    """Verify num_groups not dividing channels raises ConfigError."""
    cfg = load_config("configs/baseline.yaml")
    cfg["model"]["channels"] = [15, 32, 64]
    cfg["model"]["norm"] = {"type": "group", "num_groups": 8}
    with pytest.raises(ConfigError, match="does not divide all channel widths"):
        validate_config(cfg)


def test_config_roi_divisibility() -> None:
    """Verify ROI size not divisible by product of strides raises ConfigError."""
    cfg = load_config("configs/baseline.yaml")
    cfg["model"]["strides"] = [2, 2, 2, 2]  # product = 16
    cfg["data"]["roi_size"] = [95, 96, 96]  # 95 is not divisible by 16
    with pytest.raises(ConfigError, match="data.roi_size dimensions must be divisible"):
        validate_config(cfg)


def test_config_invalid_loss_name() -> None:
    """Verify unknown loss name raises ConfigError."""
    cfg = load_config("configs/baseline.yaml")
    cfg["loss"]["name"] = "cross_entropy"
    with pytest.raises(ConfigError, match="unknown loss.name"):
        validate_config(cfg)


def test_config_invalid_empty_policy() -> None:
    """Verify unknown empty_policy raises ConfigError."""
    cfg = load_config("configs/baseline.yaml")
    cfg["metrics"]["empty_policy"] = "ignore"
    with pytest.raises(ConfigError, match="unknown metrics.empty_policy"):
        validate_config(cfg)


def test_config_training_validator_rules() -> None:
    """Verify training section validator rules (optimizer, scheduler, warmup, accum, roi, resume)."""
    # 1. Unknown optimizer
    cfg = load_config("configs/baseline.yaml")
    cfg["training"]["optimizer"]["name"] = "rmsprop"
    with pytest.raises(ConfigError, match="unknown training.optimizer.name"):
        validate_config(cfg)

    # 2. Unknown scheduler
    cfg = load_config("configs/baseline.yaml")
    cfg["training"]["scheduler"]["name"] = "step"
    with pytest.raises(ConfigError, match="unknown training.scheduler.name"):
        validate_config(cfg)

    # 3. Warmup epochs >= epochs
    cfg = load_config("configs/baseline.yaml")
    cfg["training"]["epochs"] = 5
    cfg["training"]["scheduler"]["warmup_epochs"] = 5
    with pytest.raises(ConfigError, match="must be less than"):
        validate_config(cfg)

    # 4. grad_accum_steps < 1
    cfg = load_config("configs/baseline.yaml")
    cfg["training"]["grad_accum_steps"] = 0
    with pytest.raises(ConfigError, match="grad_accum_steps must be an integer >= 1"):
        validate_config(cfg)

    # 5. sliding_window.roi_size not divisible by model stride product
    cfg = load_config("configs/baseline.yaml")
    cfg["model"]["strides"] = [2, 2, 2, 2]  # stride product = 16
    cfg["training"]["sliding_window"]["roi_size"] = [95, 96, 96]
    with pytest.raises(ConfigError, match="training.sliding_window.roi_size dimensions must be divisible"):
        validate_config(cfg)

    # 6. resume.path missing when enabled and not 'auto'
    cfg = load_config("configs/baseline.yaml")
    cfg["training"]["resume"]["enabled"] = True
    cfg["training"]["resume"]["path"] = None
    with pytest.raises(ConfigError, match="training.resume.path is required"):
        validate_config(cfg)

    cfg["training"]["resume"]["path"] = "auto"
    validate_config(cfg)  # Should succeed with "auto"

