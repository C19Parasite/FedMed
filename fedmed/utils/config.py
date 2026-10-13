"""YAML config loading with fail-fast validation for FedMed."""

from __future__ import annotations

import math
from functools import reduce
from pathlib import Path
from typing import Any, Mapping

import yaml

ALLOWED_SOURCES = frozenset({"decathlon", "brats_local", "synthetic"})
ALLOWED_DATASET_TYPES = frozenset({"cache", "persistent", "plain"})
ALLOWED_SCHEMES = frozenset({"auto", "msd", "brats"})
ALLOWED_MODES = frozenset({"train", "val", "test"})
ALLOWED_MODEL_NAMES = frozenset({"monai_unet", "segresnet", "dynunet", "unet3d_custom"})
ALLOWED_NORM_TYPES = frozenset({"group", "instance"})
ALLOWED_LOSS_NAMES = frozenset({"dice_ce", "dice_focal", "dice"})
ALLOWED_EMPTY_POLICIES = frozenset({"brats", "skip"})
ALLOWED_INIT_NAMES = frozenset({"kaiming", "xavier", "none"})

_REQUIRED_TOP = ("seed", "data", "model", "training", "logging")
_REQUIRED_DATA = (
    "source",
    "root_dir",
    "task",
    "download",
    "label_scheme",
    "label_maps",
    "splits_path",
    "metadata_path",
    "train_ratio",
    "val_ratio",
    "test_ratio",
    "dataset_type",
    "cache_rate",
    "cache_dir",
    "roi_size",
    "num_samples",
    "pos",
    "neg",
    "spacing",
    "orientation",
    "batch_size",
    "val_batch_size",
    "test_batch_size",
    "num_workers",
    "pin_memory",
    "flip_prob",
    "scale_intensity_factor",
    "shift_intensity_offset",
    "synthetic",
)
_REQUIRED_LABEL_CHANNELS = ("tc", "wt", "et")
_REQUIRED_SYNTHETIC = ("num_patients", "spatial_size")
_REQUIRED_MODEL = ("name", "in_channels", "out_channels", "norm")
_REQUIRED_TRAINING = ("epochs", "lr", "loss", "save_dir")
_REQUIRED_LOGGING = ("level",)


class ConfigError(ValueError):
    """Raised when a YAML config is missing keys or has invalid values."""


def load_config(path: str | Path) -> dict[str, Any]:
    """Load and validate a YAML config file.

    Args:
        path: Path to a YAML file.

    Returns:
        Nested config dictionary.

    Raises:
        ConfigError: If the file is missing, keys are absent, or values are invalid.
    """
    cfg_path = Path(path)
    if not cfg_path.is_file():
        raise ConfigError(f"config file not found: {cfg_path.name}")
    with cfg_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ConfigError("config root must be a mapping")
    validate_config(loaded)
    return loaded


def validate_config(cfg: Mapping[str, Any]) -> None:
    """Fail fast on missing keys, bad ratios, or unknown enums.

    Args:
        cfg: Parsed config mapping.

    Raises:
        ConfigError: On the first validation failure.
    """
    _require_keys(cfg, _REQUIRED_TOP, prefix="")
    _validate_data(cfg)
    _validate_model(cfg)
    _validate_training(cfg)
    _validate_logging(cfg)

    # Optional sections
    if "loss" in cfg:
        _validate_loss(cfg)
    if "metrics" in cfg:
        _validate_metrics(cfg)


def _validate_data(cfg: Mapping[str, Any]) -> None:
    """Validate data section."""
    data = cfg["data"]
    if not isinstance(data, Mapping):
        raise ConfigError("data must be a mapping")
    _require_keys(data, _REQUIRED_DATA, prefix="data.")

    source = data["source"]
    if source not in ALLOWED_SOURCES:
        raise ConfigError(
            f"unknown data.source '{source}'; expected one of {sorted(ALLOWED_SOURCES)}"
        )
    dataset_type = data["dataset_type"]
    if dataset_type not in ALLOWED_DATASET_TYPES:
        raise ConfigError(
            "unknown data.dataset_type "
            f"'{dataset_type}'; expected one of {sorted(ALLOWED_DATASET_TYPES)}"
        )
    scheme = data["label_scheme"]
    if scheme not in ALLOWED_SCHEMES:
        raise ConfigError(
            f"unknown data.label_scheme '{scheme}'; expected one of {sorted(ALLOWED_SCHEMES)}"
        )

    maps = data["label_maps"]
    if not isinstance(maps, Mapping):
        raise ConfigError("data.label_maps must be a mapping")
    for scheme_name in ("msd", "brats"):
        if scheme_name not in maps:
            raise ConfigError(f"missing key 'data.label_maps.{scheme_name}'")
        channel_map = maps[scheme_name]
        if not isinstance(channel_map, Mapping):
            raise ConfigError(f"data.label_maps.{scheme_name} must be a mapping")
        _require_keys(
            channel_map, _REQUIRED_LABEL_CHANNELS, prefix=f"data.label_maps.{scheme_name}."
        )
        for channel in _REQUIRED_LABEL_CHANNELS:
            values = channel_map[channel]
            if not isinstance(values, list) or not values:
                raise ConfigError(
                    f"data.label_maps.{scheme_name}.{channel} must be a non-empty list"
                )

    ratios = (
        float(data["train_ratio"]),
        float(data["val_ratio"]),
        float(data["test_ratio"]),
    )
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ConfigError(
            "data.train_ratio + data.val_ratio + data.test_ratio must sum to 1 "
            f"(got {sum(ratios)})"
        )

    synthetic = data["synthetic"]
    if not isinstance(synthetic, Mapping):
        raise ConfigError("data.synthetic must be a mapping")
    _require_keys(synthetic, _REQUIRED_SYNTHETIC, prefix="data.synthetic.")


def _validate_model(cfg: Mapping[str, Any]) -> None:
    """Validate model section including norm policy and ROI divisibility."""
    model = cfg["model"]
    if not isinstance(model, Mapping):
        raise ConfigError("model must be a mapping")
    _require_keys(model, _REQUIRED_MODEL, prefix="model.")

    # Model name validation
    name = model["name"]
    if name not in ALLOWED_MODEL_NAMES:
        raise ConfigError(
            f"unknown model.name '{name}'; expected one of {sorted(ALLOWED_MODEL_NAMES)}"
        )

    # Normalize norm config: support both string (legacy) and dict formats
    norm = model["norm"]
    if isinstance(norm, str):
        # Legacy format: convert "INSTANCE" -> {"type": "instance"}
        norm_type = norm.lower()
        norm = {"type": norm_type}
    elif isinstance(norm, Mapping):
        if "type" not in norm:
            raise ConfigError("model.norm must have a 'type' key")
        norm_type = str(norm["type"]).lower()
    else:
        raise ConfigError("model.norm must be a string or mapping")

    # BatchNorm is banned in federated learning:
    # - Running statistics (mean/var) drift under non-IID client data
    # - FedAvg averages weights but running stats are not weights
    # - Incompatible with per-sample gradient clipping for differential privacy
    if norm_type == "batch" or "batch" in norm_type:
        raise ConfigError(
            "model.norm.type='batch' is prohibited. "
            "BatchNorm running statistics diverge under non-IID client data, "
            "are not meaningfully averaged by FedAvg, and are incompatible "
            "with per-sample gradient clipping (differential privacy). "
            "Use 'group' or 'instance' instead."
        )

    if norm_type not in ALLOWED_NORM_TYPES:
        raise ConfigError(
            f"unknown model.norm.type '{norm_type}'; "
            f"expected one of {sorted(ALLOWED_NORM_TYPES)}"
        )

    # GroupNorm: validate num_groups
    if norm_type == "group":
        num_groups = norm.get("num_groups")
        if num_groups is None:
            raise ConfigError(
                "model.norm.num_groups is required when norm.type='group'"
            )
        if not isinstance(num_groups, int) or num_groups < 1:
            raise ConfigError(
                f"model.norm.num_groups must be a positive integer, got: {num_groups}"
            )
        # Validate that num_groups divides every channel count
        channels = model.get("channels")
        if channels and isinstance(channels, list):
            bad = [c for c in channels if c % num_groups != 0]
            if bad:
                raise ConfigError(
                    f"model.norm.num_groups={num_groups} does not divide all "
                    f"channel widths. Offending widths: {bad}. "
                    f"All channels must be divisible by num_groups."
                )

    # ROI divisibility: every ROI dimension must be divisible by the
    # product of strides (total downsampling factor)
    data = cfg.get("data", {})
    roi_size = data.get("roi_size")
    strides = model.get("strides")
    if roi_size and strides and isinstance(roi_size, list) and isinstance(strides, list):
        total_downsample = reduce(lambda a, b: a * b, strides, 1)
        bad_dims = [
            (i, d) for i, d in enumerate(roi_size)
            if d % total_downsample != 0
        ]
        if bad_dims:
            raise ConfigError(
                f"data.roi_size dimensions must be divisible by the product of "
                f"model.strides={strides} (total downsampling factor={total_downsample}). "
                f"Offending dimensions: {bad_dims}"
            )

    # Init validation
    init = model.get("init", {})
    if isinstance(init, Mapping):
        init_name = init.get("name", "none")
        if init_name not in ALLOWED_INIT_NAMES:
            raise ConfigError(
                f"unknown model.init.name '{init_name}'; "
                f"expected one of {sorted(ALLOWED_INIT_NAMES)}"
            )


def _validate_training(cfg: Mapping[str, Any]) -> None:
    """Validate training section."""
    training = cfg["training"]
    if not isinstance(training, Mapping):
        raise ConfigError("training must be a mapping")
    _require_keys(training, _REQUIRED_TRAINING, prefix="training.")


def _validate_logging(cfg: Mapping[str, Any]) -> None:
    """Validate logging section."""
    logging_cfg = cfg["logging"]
    if not isinstance(logging_cfg, Mapping):
        raise ConfigError("logging must be a mapping")
    _require_keys(logging_cfg, _REQUIRED_LOGGING, prefix="logging.")


def _validate_loss(cfg: Mapping[str, Any]) -> None:
    """Validate loss section."""
    loss = cfg["loss"]
    if not isinstance(loss, Mapping):
        raise ConfigError("loss must be a mapping")

    loss_name = loss.get("name", "dice_ce")
    if loss_name not in ALLOWED_LOSS_NAMES:
        raise ConfigError(
            f"unknown loss.name '{loss_name}'; "
            f"expected one of {sorted(ALLOWED_LOSS_NAMES)}"
        )

    # Channel weights validation
    cw = loss.get("channel_weights")
    if cw is not None:
        if not isinstance(cw, list) or len(cw) != 3:
            raise ConfigError(
                f"loss.channel_weights must be a list of 3 floats, got: {cw}"
            )
        for i, w in enumerate(cw):
            if not isinstance(w, (int, float)) or w < 0:
                raise ConfigError(
                    f"loss.channel_weights[{i}] must be a non-negative number, got: {w}"
                )


def _validate_metrics(cfg: Mapping[str, Any]) -> None:
    """Validate metrics section."""
    metrics = cfg["metrics"]
    if not isinstance(metrics, Mapping):
        raise ConfigError("metrics must be a mapping")

    # Empty policy
    policy = metrics.get("empty_policy", "brats")
    if policy not in ALLOWED_EMPTY_POLICIES:
        raise ConfigError(
            f"unknown metrics.empty_policy '{policy}'; "
            f"expected one of {sorted(ALLOWED_EMPTY_POLICIES)}"
        )

    # Threshold
    threshold = metrics.get("threshold", 0.5)
    if not isinstance(threshold, (int, float)) or threshold <= 0 or threshold >= 1:
        raise ConfigError(
            f"metrics.threshold must be a float in (0, 1), got: {threshold}"
        )


def _require_keys(mapping: Mapping[str, Any], keys: tuple[str, ...], prefix: str) -> None:
    missing = [f"{prefix}{key}" for key in keys if key not in mapping]
    if missing:
        raise ConfigError(f"missing required config keys: {', '.join(missing)}")
