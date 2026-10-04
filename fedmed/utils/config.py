"""YAML config loading with fail-fast validation for FedMed."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

ALLOWED_SOURCES = frozenset({"decathlon", "brats_local", "synthetic"})
ALLOWED_DATASET_TYPES = frozenset({"cache", "persistent", "plain"})
ALLOWED_SCHEMES = frozenset({"auto", "msd", "brats"})
ALLOWED_MODES = frozenset({"train", "val", "test"})

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
_REQUIRED_MODEL = ("name", "in_channels", "out_channels", "channels", "strides", "norm")
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

    model = cfg["model"]
    if not isinstance(model, Mapping):
        raise ConfigError("model must be a mapping")
    _require_keys(model, _REQUIRED_MODEL, prefix="model.")

    training = cfg["training"]
    if not isinstance(training, Mapping):
        raise ConfigError("training must be a mapping")
    _require_keys(training, _REQUIRED_TRAINING, prefix="training.")

    logging_cfg = cfg["logging"]
    if not isinstance(logging_cfg, Mapping):
        raise ConfigError("logging must be a mapping")
    _require_keys(logging_cfg, _REQUIRED_LOGGING, prefix="logging.")


def _require_keys(mapping: Mapping[str, Any], keys: tuple[str, ...], prefix: str) -> None:
    missing = [f"{prefix}{key}" for key in keys if key not in mapping]
    if missing:
        raise ConfigError(f"missing required config keys: {', '.join(missing)}")
