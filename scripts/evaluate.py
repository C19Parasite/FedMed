"""CLI script to evaluate a trained model checkpoint on validation or test splits."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

from fedmed.data.loaders import get_loaders
from fedmed.models.unet3d import get_model
from fedmed.training.checkpoint import load_checkpoint
from fedmed.training.losses import build_loss
from fedmed.training.metrics import SegmentationMetrics
from fedmed.training.run import config_hash, environment_info
from fedmed.training.trainer import Trainer
from fedmed.utils.config import load_config
from fedmed.utils.logging import get_logger
from fedmed.utils.seed import set_seed

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI options for model evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate 3D segmentation checkpoint on BraTS dataset split."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to saved model checkpoint (.pt file).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["val", "test", "train"],
        help="Dataset split to evaluate on ('val' or 'test').",
    )
    parser.add_argument(
        "--hd95",
        action="store_true",
        help="Compute 95th-percentile Hausdorff Distance (HD95).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["auto", "cpu", "cuda"],
        help="Evaluation execution device.",
    )
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="Key=value configuration overrides (e.g. data.batch_size=4).",
    )
    return parser.parse_args()


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


def main() -> None:
    """Run evaluation and save results/<split>_metrics.json."""
    args = parse_args()

    # Rule: Refuse to evaluate on the train split
    if args.split == "train":
        logger.error("Refusing evaluation on 'train' split. Evaluation is only permitted on 'val' or 'test'.")
        sys.exit(1)

    ckpt_path = Path(args.checkpoint).resolve()
    if not ckpt_path.is_file():
        logger.error("Checkpoint file not found: %s", ckpt_path)
        sys.exit(1)

    cfg = load_config(args.config)
    for ov in args.override:
        if "=" in ov:
            k, v = ov.split("=", 1)
            apply_override(cfg, k.strip(), v.strip())
    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    # Resolve device
    dev_str = args.device or cfg.get("training", {}).get("device", "auto")
    if dev_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(dev_str)

    loaders = get_loaders(cfg)
    if args.split not in loaders:
        logger.error("Requested split '%s' not found in dataset loaders.", args.split)
        sys.exit(1)
    loader = loaders[args.split]

    model = get_model(cfg).to(device)
    loss_fn = build_loss(cfg)
    metrics = SegmentationMetrics(cfg)

    meta = load_checkpoint(ckpt_path, model=model, map_location=device)
    logger.info("Loaded checkpoint from %s (epoch %s)", ckpt_path.name, meta.get("epoch", "unknown"))

    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=None,  # type: ignore
        scheduler=None,
        cfg=cfg,
        device=device,
        metrics=metrics,
    )

    eval_results = trainer.validate(loader, include_hd95=args.hd95)

    # Format output record
    hd95_data = None
    if args.hd95 and "hd95_mean" in eval_results:
        hd95_data = {
            "tc": float(eval_results["hd95_tc"]),
            "wt": float(eval_results["hd95_wt"]),
            "et": float(eval_results["hd95_et"]),
            "mean": float(eval_results["hd95_mean"]),
        }

    env_data = environment_info()
    short_env = {
        "python": env_data["python"],
        "torch": env_data["torch"],
        "monai": env_data["monai"],
        "cuda_available": env_data["cuda_available"],
        "platform": env_data["platform"],
    }

    report = {
        "split": args.split,
        "checkpoint_epoch": int(meta.get("epoch", 0)),
        "n_volumes": int(eval_results["n_volumes"]),
        "dice": {
            "tc": round(float(eval_results.get("dice_tc", 0.0)), 4),
            "wt": round(float(eval_results.get("dice_wt", 0.0)), 4),
            "et": round(float(eval_results.get("dice_et", 0.0)), 4),
            "mean": round(float(eval_results.get("dice_mean", 0.0)), 4),
        },
        "hd95": hd95_data,
        "val_loss": round(float(eval_results.get("val_loss", 0.0)), 4),
        "inference_time_s": round(float(eval_results.get("inference_time_s", 0.0)), 2),
        "empty_policy": str(cfg.get("metrics", {}).get("empty_policy", "brats")),
        "threshold": float(cfg.get("metrics", {}).get("threshold", 0.5)),
        "config_hash": config_hash(cfg),
        "environment": short_env,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    # Save to results/<split>_metrics.json
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    out_json = results_dir / f"{args.split}_metrics.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Print summary table
    print("\n================ EVALUATION METRIC REPORT ================")
    print(f"Split:               {args.split.upper()}")
    print(f"Checkpoint Epoch:    {meta.get('epoch', 'N/A')}")
    print(f"Total Volumes:       {eval_results['n_volumes']}")
    print(f"Tumor Core (TC):     {eval_results['dice_tc']:.4f}")
    print(f"Whole Tumor (WT):    {eval_results['dice_wt']:.4f}")
    print(f"Enhancing Tumor (ET):{eval_results['dice_et']:.4f}")
    print(f"Mean BraTS Dice:     {eval_results['dice_mean']:.4f}")
    if args.hd95 and hd95_data:
        print("--- Hausdorff Distance 95% (mm) ---")
        print(f"  TC HD95:           {hd95_data['tc']:.2f}")
        print(f"  WT HD95:           {hd95_data['wt']:.2f}")
        print(f"  ET HD95:           {hd95_data['et']:.2f}")
        print(f"  Mean HD95:         {hd95_data['mean']:.2f}")
    print(f"Report JSON Saved:   {out_json}")
    print("==========================================================\n")


if __name__ == "__main__":
    main()
