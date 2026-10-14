"""Run directory management, configuration hashing, and environment logging.

Strict privacy rule: No hostnames, usernames, or filesystem paths in saved files.
No patient IDs or raw images may ever be written to run directories.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
from typing import Any, Mapping

import monai
import numpy as np
import torch
import yaml

from fedmed.utils.logging import get_logger

logger = get_logger(__name__)


def config_hash(cfg: Mapping[str, Any]) -> str:
    """Compute SHA-256 of canonical (sorted-key) JSON of the config.

    Args:
        cfg: Configuration dictionary or mapping.

    Returns:
        First 12 hex characters of the SHA-256 digest.
    """
    canonical_json = json.dumps(cfg, sort_keys=True, default=str)
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()[:12]


def save_resolved_config(cfg: Mapping[str, Any], run_dir: Path | str) -> Path:
    """Save full resolved configuration to <run_dir>/config.yaml.

    Args:
        cfg: Validated configuration mapping.
        run_dir: Destination run directory.

    Returns:
        Path to written config.yaml.
    """
    out_path = Path(run_dir) / "config.yaml"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(dict(cfg), f, sort_keys=False)
    return out_path


def environment_info() -> dict[str, Any]:
    """Capture sanitized environment metadata without hostnames, usernames, or paths.

    Returns:
        Dictionary containing python, torch, monai, numpy, cuda, cudnn, and platform info.
    """
    cuda_avail = torch.cuda.is_available()
    cudnn_ver: str | None = None
    if cuda_avail and torch.backends.cudnn.is_available():
        cudnn_ver = str(torch.backends.cudnn.version())

    gpu_name = torch.cuda.get_device_name(0) if cuda_avail else None
    gpu_count = torch.cuda.device_count() if cuda_avail else 0

    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "monai": monai.__version__,
        "numpy": np.__version__,
        "cuda_available": cuda_avail,
        "cuda_version": torch.version.cuda if cuda_avail else None,
        "cudnn_version": cudnn_ver,
        "gpu_name": gpu_name,
        "gpu_count": gpu_count,
        "platform": platform.system(),
    }


def save_environment_info(run_dir: Path | str) -> Path:
    """Save sanitized environment metadata to <run_dir>/environment.json.

    Args:
        run_dir: Target run directory.

    Returns:
        Path to written environment.json.
    """
    info = environment_info()
    out_path = Path(run_dir) / "environment.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    return out_path


def create_run_dir(cfg: Mapping[str, Any]) -> Path:
    """Create or locate run directory with checkpoints, tensorboard, and results subfolders.

    Format: <output_dir>/<run_name>/<UTC timestamp>/
    If resume is enabled with "auto", reuses the most recent run dir for that run_name.

    Args:
        cfg: Experiment configuration mapping.

    Returns:
        Path to initialized run directory.
    """
    train_cfg = cfg.get("training", {})
    log_cfg = train_cfg.get("logging", {})
    output_dir = Path(log_cfg.get("output_dir", "outputs/runs"))
    run_name = str(log_cfg.get("run_name", "baseline"))

    base_dir = output_dir / run_name
    resume_cfg = train_cfg.get("resume", {})
    is_resume_auto = (
        bool(resume_cfg.get("enabled", False))
        and str(resume_cfg.get("path", "")).strip().lower() == "auto"
    )

    if is_resume_auto and base_dir.is_dir():
        # Find existing run directories sorted alphabetically (which matches UTC timestamp order)
        existing_runs = [d for d in base_dir.iterdir() if d.is_dir()]
        if existing_runs:
            existing_runs.sort(key=lambda d: d.name)
            run_dir = existing_runs[-1]
            (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
            (run_dir / "tensorboard").mkdir(parents=True, exist_ok=True)
            (run_dir / "results").mkdir(parents=True, exist_ok=True)
            logger.info("Resuming run in existing directory: %s", run_dir)
            return run_dir

    # Otherwise create a new timestamped run dir
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "tensorboard").mkdir(parents=True, exist_ok=True)
    (run_dir / "results").mkdir(parents=True, exist_ok=True)

    # Save resolved config and environment info
    save_resolved_config(cfg, run_dir)
    save_environment_info(run_dir)

    logger.info("Created new run directory: %s (hash=%s)", run_dir, config_hash(cfg))
    return run_dir
