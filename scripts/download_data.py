"""Download or verify the BraTS / MSD dataset according to configuration."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from fedmed.data.datasets import discover_brats_local, discover_decathlon, discover_synthetic
from fedmed.utils.config import load_config
from fedmed.utils.logging import get_logger
from fedmed.utils.seed import set_seed

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for data download script."""
    parser = argparse.ArgumentParser(
        description="Download or verify BraTS / Medical Segmentation Decathlon Task01."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Explicitly trigger DecathlonDataset download regardless of config.",
    )
    return parser.parse_args()


def main() -> None:
    """Execute dataset preparation according to YAML configuration."""
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]))

    data_cfg = cfg["data"]
    source = data_cfg["source"]
    root_dir = Path(data_cfg["root_dir"])

    logger.info("Preparing data pipeline for source=%s", source)

    if source == "decathlon":
        if args.download:
            cfg["data"]["download"] = True
        logger.info(
            "Decathlon mode selected. Task: %s. Download enabled: %s",
            data_cfg.get("task"),
            cfg["data"]["download"],
        )
        try:
            cases = discover_decathlon(cfg)
            logger.info("Successfully verified MSD Decathlon dataset. Total cases: %s", len(cases))
        except FileNotFoundError as exc:
            logger.warning(
                "Decathlon dataset not found locally. To download Task01_BrainTumour automatically:\n"
                "  python scripts/download_data.py --config %s --download\n"
                "or set data.download: true in your YAML configuration.",
                args.config,
            )
            sys.exit(1)

    elif source == "brats_local":
        cases = discover_brats_local(root_dir)
        logger.info("Discovered local BraTS cases: %s", len(cases))
        if not cases:
            logger.warning(
                "No local BraTS cases found in root directory. Expected structure:\n"
                "  <root_dir>/<Patient_ID>/<Patient_ID>_flair.nii.gz\n"
                "  <root_dir>/<Patient_ID>/<Patient_ID>_t1.nii.gz\n"
                "  <root_dir>/<Patient_ID>/<Patient_ID>_t1ce.nii.gz\n"
                "  <root_dir>/<Patient_ID>/<Patient_ID>_t2.nii.gz\n"
                "  <root_dir>/<Patient_ID>/<Patient_ID>_seg.nii.gz\n"
            )

    elif source == "synthetic":
        cases = discover_synthetic(cfg)
        logger.info("Successfully generated/verified synthetic BraTS cases: %s", len(cases))

    else:
        logger.error("Unsupported data source: %s", source)
        sys.exit(1)


if __name__ == "__main__":
    main()
