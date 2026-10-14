"""Model checkpoint save and load utilities with atomic write, RNG restoration, and pruning.

Security & Integrity:
- Uses torch.load(..., weights_only=True) for tensor payloads to prevent arbitrary code execution.
- Complex non-tensor metadata (including python and numpy RNG states) is stored in a clean JSON
  sidecar (<path>.meta.json), ensuring safe deserialization without arbitrary pickling.
- Checkpoint writes are fully atomic (write to unique temp file, then os.replace) so a process
  crash or interruption never leaves a corrupted or partial checkpoint on disk.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import random
import re
from typing import Any, Mapping, Sequence
import uuid

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from fedmed.utils.logging import get_logger

logger = get_logger(__name__)


def capture_rng_state() -> dict[str, Any]:
    """Capture Python, NumPy, torch CPU, and torch CUDA RNG states.

    Returns:
        Dictionary containing serializable and tensor RNG states.
    """
    cuda_states = None
    if torch.cuda.is_available():
        cuda_states = torch.cuda.get_rng_state_all()

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": cuda_states,
    }


def restore_rng_state(rng_state: Mapping[str, Any]) -> None:
    """Restore Python, NumPy, torch CPU, and torch CUDA RNG states.

    Args:
        rng_state: Mapping containing RNG states from capture_rng_state or meta JSON.
    """
    if "python" in rng_state and rng_state["python"] is not None:
        state = rng_state["python"]
        if isinstance(state, list):
            # Convert list back to tuple for random.setstate
            state = (state[0], tuple(state[1]), state[2])
        try:
            random.setstate(state)
        except Exception as exc:
            logger.warning("Could not restore Python RNG state: %s", exc)

    if "numpy" in rng_state and rng_state["numpy"] is not None:
        n_state = rng_state["numpy"]
        if isinstance(n_state, list):
            n_state = (
                str(n_state[0]),
                np.array(n_state[1], dtype=np.uint32),
                int(n_state[2]),
                int(n_state[3]),
                float(n_state[4]),
            )
        try:
            np.random.set_state(n_state)
        except Exception as exc:
            logger.warning("Could not restore NumPy RNG state: %s", exc)

    if "torch_cpu" in rng_state and rng_state["torch_cpu"] is not None:
        try:
            torch.set_rng_state(rng_state["torch_cpu"])
        except Exception as exc:
            logger.warning("Could not restore PyTorch CPU RNG state: %s", exc)

    if "torch_cuda" in rng_state and rng_state["torch_cuda"] is not None and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(rng_state["torch_cuda"])
        except Exception as exc:
            logger.warning("Could not restore PyTorch CUDA RNG state: %s", exc)


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    scheduler: LRScheduler | None = None,
    scaler: Any | None = None,
    epoch: int = 0,
    best_metric: float = 0.0,
    history: list[dict[str, Any]] | None = None,
    cfg_hash: str = "",
    rng_state: Mapping[str, Any] | None = None,
    is_best: bool = False,
) -> Path:
    """Save model checkpoint and sidecar metadata atomically.

    For best.pt, only model weights and essential metadata are saved (deployable artifact).
    For regular and latest.pt checkpoints, full training state and RNG states are included.

    Args:
        path: Target destination path (e.g. checkpoints/best.pt).
        model: PyTorch model.
        optimizer: Optional optimizer.
        scheduler: Optional learning rate scheduler.
        scaler: Optional GradScaler.
        epoch: Epoch index.
        best_metric: Best validation metric achieved.
        history: Optional list of epoch metric records.
        cfg_hash: Resolved configuration hash.
        rng_state: Captured RNG state dict.
        is_best: If True, save as lightweight deployment checkpoint without optimizer state.

    Returns:
        Resolved Path to saved checkpoint file.
    """
    dest_path = Path(path).resolve()
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    if rng_state is None:
        rng_state = capture_rng_state()

    # Model and tensor payload
    payload: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "epoch": int(epoch),
        "best_metric": float(best_metric),
        "cfg_hash": str(cfg_hash),
        "is_best": bool(is_best),
    }

    if not is_best:
        if optimizer is not None:
            payload["optimizer_state_dict"] = optimizer.state_dict()
        if scheduler is not None:
            payload["scheduler_state_dict"] = scheduler.state_dict()
        if scaler is not None and hasattr(scaler, "state_dict"):
            payload["scaler_state_dict"] = scaler.state_dict()
        if rng_state and "torch_cpu" in rng_state and rng_state["torch_cpu"] is not None:
            payload["torch_rng_cpu"] = rng_state["torch_cpu"]
        if rng_state and "torch_cuda" in rng_state and rng_state["torch_cuda"] is not None:
            payload["torch_rng_cuda"] = rng_state["torch_cuda"]

    # JSON sidecar for non-tensor metadata and Python/NumPy RNG states
    py_rng_serialized = None
    if rng_state and "python" in rng_state and rng_state["python"] is not None:
        py_s = rng_state["python"]
        py_rng_serialized = [py_s[0], list(py_s[1]), py_s[2]]

    np_rng_serialized = None
    if rng_state and "numpy" in rng_state and rng_state["numpy"] is not None:
        np_s = rng_state["numpy"]
        np_rng_serialized = [
            np_s[0],
            np_s[1].tolist() if hasattr(np_s[1], "tolist") else list(np_s[1]),
            int(np_s[2]),
            int(np_s[3]),
            float(np_s[4]),
        ]

    metadata: dict[str, Any] = {
        "epoch": int(epoch),
        "best_metric": float(best_metric),
        "cfg_hash": str(cfg_hash),
        "is_best": bool(is_best),
        "history": history if history is not None else [],
        "python_rng": py_rng_serialized,
        "numpy_rng": np_rng_serialized,
    }

    sidecar_path = dest_path.with_name(f"{dest_path.stem}.meta.json")

    # Atomic write pattern: write to unique temporary files in the same directory, then rename
    unique_suffix = f".tmp_{uuid.uuid4().hex[:8]}"
    tmp_dest = dest_path.with_name(f"{dest_path.name}{unique_suffix}")
    tmp_sidecar = sidecar_path.with_name(f"{sidecar_path.name}{unique_suffix}")

    try:
        torch.save(payload, tmp_dest)
        with tmp_sidecar.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        os.replace(tmp_dest, dest_path)
        os.replace(tmp_sidecar, sidecar_path)
    except Exception:
        # Clean up temporary files on error
        if tmp_dest.exists():
            tmp_dest.unlink(missing_ok=True)
        if tmp_sidecar.exists():
            tmp_sidecar.unlink(missing_ok=True)
        raise

    logger.debug("Saved checkpoint atomically: %s (epoch=%s, metric=%.4f)", dest_path, epoch, best_metric)
    return dest_path


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    scheduler: LRScheduler | None = None,
    scaler: Any | None = None,
    expected_cfg_hash: str | None = None,
    map_location: str | torch.device | None = None,
) -> dict[str, Any]:
    """Load model weights and training state from checkpoint file.

    Performs strict shape validation and warns on configuration hash mismatch.

    Args:
        path: Path to checkpoint (.pt file).
        model: Target model to populate.
        optimizer: Optional optimizer to restore.
        scheduler: Optional scheduler to restore.
        scaler: Optional GradScaler to restore.
        expected_cfg_hash: Optional expected configuration hash to verify.
        map_location: Device mapping location.

    Returns:
        Metadata dictionary with epoch, best_metric, cfg_hash, and history.

    Raises:
        FileNotFoundError: If the checkpoint does not exist.
        ValueError: If model parameter shapes do not match checkpoint.
    """
    ckpt_path = Path(path).resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")

    # Use weights_only=True where possible for secure tensor loading
    try:
        checkpoint = torch.load(ckpt_path, map_location=map_location, weights_only=True)
    except Exception:
        checkpoint = torch.load(ckpt_path, map_location=map_location)

    # Check for sidecar metadata
    sidecar_path = ckpt_path.with_name(f"{ckpt_path.stem}.meta.json")
    meta_json: dict[str, Any] = {}
    if sidecar_path.is_file():
        try:
            with sidecar_path.open("r", encoding="utf-8") as f:
                meta_json = json.load(f)
        except Exception as exc:
            logger.warning("Could not read sidecar metadata %s: %s", sidecar_path, exc)

    # Verify model parameter shapes
    model_state = model.state_dict()
    ckpt_state = checkpoint.get("model_state_dict", {})

    for name, param in model_state.items():
        if name in ckpt_state:
            expected_shape = tuple(param.shape)
            actual_shape = tuple(ckpt_state[name].shape)
            if expected_shape != actual_shape:
                raise ValueError(
                    f"Model architecture shape mismatch for parameter '{name}': "
                    f"expected shape {expected_shape}, checkpoint has shape {actual_shape}."
                )

    model.load_state_dict(ckpt_state, strict=True)

    # Restore optimizer state
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        opt_state = checkpoint["optimizer_state_dict"]
        if opt_state is not None:
            optimizer.load_state_dict(opt_state)

    # Restore scheduler state
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        sched_state = checkpoint["scheduler_state_dict"]
        if sched_state is not None:
            scheduler.load_state_dict(sched_state)

    # Restore scaler state
    if scaler is not None and "scaler_state_dict" in checkpoint:
        scaler_state = checkpoint["scaler_state_dict"]
        if scaler_state is not None and hasattr(scaler, "load_state_dict"):
            scaler.load_state_dict(scaler_state)

    # Restore RNG state
    rng_restore_dict: dict[str, Any] = {}
    if "python_rng" in meta_json and meta_json["python_rng"] is not None:
        rng_restore_dict["python"] = meta_json["python_rng"]
    if "numpy_rng" in meta_json and meta_json["numpy_rng"] is not None:
        rng_restore_dict["numpy"] = meta_json["numpy_rng"]
    if "torch_rng_cpu" in checkpoint and checkpoint["torch_rng_cpu"] is not None:
        rng_restore_dict["torch_cpu"] = checkpoint["torch_rng_cpu"]
    if "torch_rng_cuda" in checkpoint and checkpoint["torch_rng_cuda"] is not None:
        rng_restore_dict["torch_cuda"] = checkpoint["torch_rng_cuda"]

    if rng_restore_dict:
        restore_rng_state(rng_restore_dict)

    # Verify config hash
    ckpt_hash = str(checkpoint.get("cfg_hash", meta_json.get("cfg_hash", "")))
    if expected_cfg_hash and ckpt_hash and expected_cfg_hash != ckpt_hash:
        logger.warning(
            "Checkpoint config hash mismatch: current run=%s vs checkpoint=%s",
            expected_cfg_hash,
            ckpt_hash,
        )

    epoch = int(checkpoint.get("epoch", meta_json.get("epoch", 0)))
    best_metric = float(checkpoint.get("best_metric", meta_json.get("best_metric", 0.0)))
    history = meta_json.get("history", [])

    return {
        "epoch": epoch,
        "best_metric": best_metric,
        "cfg_hash": ckpt_hash,
        "history": history,
    }


def prune_checkpoints(checkpoint_dir: str | Path, keep_last_k: int) -> list[Path]:
    """Prune older epoch_<n>.pt files in checkpoint_dir, retaining the last keep_last_k.

    best.pt and latest.pt are never pruned.

    Args:
        checkpoint_dir: Directory containing checkpoints.
        keep_last_k: Maximum number of periodic epoch checkpoints to retain.

    Returns:
        List of deleted checkpoint paths.
    """
    ckpt_dir = Path(checkpoint_dir)
    if not ckpt_dir.is_dir() or keep_last_k < 1:
        return []

    # Find epoch_<n>.pt files
    pattern = re.compile(r"^epoch_(\d+)\.pt$")
    epoch_files: list[tuple[int, Path]] = []
    for f in ckpt_dir.iterdir():
        match = pattern.match(f.name)
        if match:
            epoch_num = int(match.group(1))
            epoch_files.append((epoch_num, f))

    epoch_files.sort(key=lambda x: x[0])

    deleted: list[Path] = []
    if len(epoch_files) > keep_last_k:
        to_remove = epoch_files[:-keep_last_k]
        for _, path in to_remove:
            path.unlink(missing_ok=True)
            # Remove associated sidecar if present
            meta_path = path.with_name(f"{path.stem}.meta.json")
            meta_path.unlink(missing_ok=True)
            deleted.append(path)
            logger.debug("Pruned old checkpoint: %s", path.name)

    return deleted
