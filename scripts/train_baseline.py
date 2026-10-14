"""Centralized baseline training and evaluation script for 3D brain tumor MRI segmentation.

CLI Options:
  --config PATH: Path to YAML config file (default: configs/baseline.yaml).
  --smoke: Fast smoke test mode using smoke_test settings.
  --resume: Resume training from latest checkpoint.
  --device: Target device ('auto', 'cpu', 'cuda').
  --override: Dotted key=value overrides (e.g. data.batch_size=4).
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Mapping

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import monai.utils
import torch

from fedmed.data.loaders import get_loaders
from fedmed.models.params import log_model_summary
from fedmed.models.unet3d import get_model
from fedmed.training.checkpoint import load_checkpoint
from fedmed.training.losses import build_loss
from fedmed.training.metrics import SegmentationMetrics
from fedmed.training.optim import build_optimizer, build_scheduler
from fedmed.training.run import config_hash, create_run_dir, environment_info
from fedmed.training.trainer import Trainer
from fedmed.utils.config import load_config, validate_config
from fedmed.utils.logging import get_logger
from fedmed.utils.seed import set_seed

logger = get_logger(__name__)


def apply_override(cfg: dict[str, Any], key_path: str, val_str: str) -> None:
    """Apply dotted key-value override into nested dictionary."""
    parts = key_path.split(".")
    curr: dict[str, Any] = cfg
    for p in parts[:-1]:
        if p not in curr or not isinstance(curr[p], dict):
            curr[p] = {}
        curr = curr[p]
    last_key = parts[-1]

    val_lower = val_str.lower()
    if val_lower in ("true", "yes"):
        val: Any = True
    elif val_lower in ("false", "no"):
        val = False
    elif val_lower in ("null", "none"):
        val = None
    else:
        try:
            val = int(val_str)
        except ValueError:
            try:
                val = float(val_str)
            except ValueError:
                val = val_str
    curr[last_key] = val


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train centralized baseline 3D segmentation model on BraTS dataset."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Apply fast smoke test configuration.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from latest checkpoint in run directory.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["auto", "cpu", "cuda"],
        help="Override execution device.",
    )
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="Key=value configuration overrides (e.g. data.batch_size=4).",
    )
    return parser.parse_args()


def main() -> None:
    """Run baseline training and final test evaluation."""
    args = parse_args()

    cfg = load_config(args.config)
    cfg["_config_path"] = args.config

    # Apply dotted key-value overrides
    for ov in args.override:
        if "=" in ov:
            k, v = ov.split("=", 1)
            apply_override(cfg, k.strip(), v.strip())

    # Apply --smoke overrides
    if args.smoke:
        smoke_cfg = load_config("configs/smoke_test.yaml")
        cfg["training"]["epochs"] = smoke_cfg["training"]["epochs"]
        cfg["training"]["max_steps_per_epoch"] = smoke_cfg["training"]["max_steps_per_epoch"]
        cfg["training"]["amp"]["enabled"] = False
        cfg["training"]["early_stopping"]["enabled"] = False
        cfg["metrics"]["hd95"]["enabled"] = False
        cfg["data"]["roi_size"] = smoke_cfg["data"]["roi_size"]
        cfg["training"]["sliding_window"]["roi_size"] = smoke_cfg["training"]["sliding_window"]["roi_size"]
        cfg["data"]["num_workers"] = 0
        if "synthetic" in cfg.get("data", {}):
            cfg["data"]["synthetic"]["spatial_size"] = smoke_cfg["data"]["synthetic"]["spatial_size"]
            cfg["data"]["synthetic"]["num_patients"] = smoke_cfg["data"]["synthetic"]["num_patients"]

    if args.resume:
        cfg["training"]["resume"]["enabled"] = True
        cfg["training"]["resume"]["path"] = "auto"

    if args.device:
        cfg["training"]["device"] = args.device

    # Re-validate config after modifications
    validate_config(cfg)

    # Determinism and seeding
    seed = int(cfg.get("seed", 42))
    set_seed(seed)
    if cfg.get("training", {}).get("deterministic", {}).get("enabled", False):
        monai.utils.set_determinism(seed=seed)

    # Initialize run directory
    run_dir = create_run_dir(cfg)

    # Resolve execution device
    dev_str = cfg.get("training", {}).get("device", "auto")
    if dev_str == "auto":
        target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        target_device = torch.device(dev_str)

    # Build components
    loaders = get_loaders(cfg)
    model = get_model(cfg).to(target_device)
    loss_fn = build_loss(cfg)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    metrics = SegmentationMetrics(cfg)

    # Log architecture summary
    summary = log_model_summary(model, logger)

    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        cfg=cfg,
        device=target_device,
        metrics=metrics,
        run_dir=run_dir,
    )

    # Check for resume
    start_epoch = 0
    if cfg.get("training", {}).get("resume", {}).get("enabled", False):
        res_path_str = str(cfg["training"]["resume"]["path"])
        if res_path_str == "auto":
            ckpt_to_resume = run_dir / "checkpoints" / "latest.pt"
        else:
            ckpt_to_resume = Path(res_path_str)

        if ckpt_to_resume.is_file():
            meta = trainer.load_checkpoint(ckpt_to_resume)
            start_epoch = meta.get("epoch", 0)
            logger.info("Resumed from checkpoint: %s (epoch %d)", ckpt_to_resume, start_epoch)

    # Execute training loop
    fit_results = trainer.fit(
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        start_epoch=start_epoch,
    )

    # Final evaluation on the TEST split with best.pt
    best_ckpt = run_dir / "checkpoints" / "best.pt"
    if not best_ckpt.is_file():
        best_ckpt = run_dir / "checkpoints" / "latest.pt"

    logger.info("Running final evaluation on test split using %s...", best_ckpt.name)
    trainer.load_checkpoint(best_ckpt)
    test_metrics = trainer.validate(loaders["test"], include_hd95=bool(cfg.get("metrics", {}).get("hd95", {}).get("enabled", False)))

    # Assemble baseline_metrics.json (strictly without filesystem paths or patient IDs)
    hd95_data = None
    if "hd95_mean" in test_metrics:
        hd95_data = {
            "tc": float(test_metrics["hd95_tc"]),
            "wt": float(test_metrics["hd95_wt"]),
            "et": float(test_metrics["hd95_et"]),
            "mean": float(test_metrics["hd95_mean"]),
        }

    sw_cfg = cfg.get("training", {}).get("sliding_window", {})
    env_data = environment_info()
    short_env = {
        "python": env_data["python"],
        "torch": env_data["torch"],
        "monai": env_data["monai"],
        "cuda_available": env_data["cuda_available"],
        "platform": env_data["platform"],
    }

    baseline_record = {
        "split": "test",
        "checkpoint_epoch": int(fit_results["best_epoch"]),
        "n_volumes": int(test_metrics["n_volumes"]),
        "dice": {
            "tc": round(float(test_metrics.get("dice_tc", 0.0)), 4),
            "wt": round(float(test_metrics.get("dice_wt", 0.0)), 4),
            "et": round(float(test_metrics.get("dice_et", 0.0)), 4),
            "mean": round(float(test_metrics.get("dice_mean", 0.0)), 4),
        },
        "hd95": hd95_data,
        "empty_policy": str(cfg.get("metrics", {}).get("empty_policy", "brats")),
        "threshold": float(cfg.get("metrics", {}).get("threshold", 0.5)),
        "sliding_window": {
            "roi_size": sw_cfg.get("roi_size", cfg.get("data", {}).get("roi_size", [96, 96, 96])),
            "overlap": float(sw_cfg.get("overlap", 0.25)),
            "sw_batch_size": int(sw_cfg.get("sw_batch_size", 1)),
            "mode": str(sw_cfg.get("mode", "gaussian")),
        },
        "model": {
            "name": str(cfg.get("model", {}).get("name", "monai_unet")),
            "parameters": summary["total_parameters"],
            "size_mb": summary["size_mb"],
            "norm": summary["norm_type"],
        },
        "training": {
            "epochs_run": int(len(fit_results["history"])),
            "best_epoch": int(fit_results["best_epoch"]),
            "optimizer": str(cfg.get("training", {}).get("optimizer", {}).get("name", "adamw")),
            "lr": float(cfg.get("training", {}).get("optimizer", {}).get("lr", 2e-4)),
            "amp": bool(cfg.get("training", {}).get("amp", {}).get("enabled", False)),
        },
        "seed": seed,
        "config_hash": config_hash(cfg),
        "environment": short_env,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    results_dir = run_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    baseline_json_path = results_dir / "baseline_metrics.json"
    with baseline_json_path.open("w", encoding="utf-8") as f:
        json.dump(baseline_record, f, indent=2)

    # Also save to top-level results/baseline_metrics.json for convenience
    top_results = Path("results")
    top_results.mkdir(parents=True, exist_ok=True)
    with (top_results / "baseline_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(baseline_record, f, indent=2)

    # One-screen summary output
    print("\n================ CENTRALIZED BASELINE RESULTS ================")
    print(f"Best Validation Epoch: {fit_results['best_epoch']}")
    print(f"Best Validation Mean Dice: {fit_results['best_metric']:.4f}")
    if fit_results.get("best_metrics"):
        bm = fit_results["best_metrics"]
        print(f"  TC Dice: {bm.get('dice_tc', 0.0):.4f} | WT Dice: {bm.get('dice_wt', 0.0):.4f} | ET Dice: {bm.get('dice_et', 0.0):.4f}")
    print(f"Test Split Mean Dice:   {test_metrics['dice_mean']:.4f}")
    print(f"  TC Dice: {test_metrics.get('dice_tc', 0.0):.4f} | WT Dice: {test_metrics.get('dice_wt', 0.0):.4f} | ET Dice: {test_metrics.get('dice_et', 0.0):.4f}")
    print(f"Run Directory:          {run_dir}")
    print(f"Baseline JSON Saved:    {baseline_json_path}")
    print(f"Resume Command:         python scripts/train_baseline.py --config {args.config} --resume")
    print(f"Evaluate Command:       python scripts/evaluate.py --config {args.config} --checkpoint {best_ckpt} --split test")
    print("==============================================================\n")


if __name__ == "__main__":
    main()
