"""Visualize 4-modality MRI slices and multi-channel tumor segmentation overlay."""

from __future__ import annotations

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch

from fedmed.data.loaders import get_loaders
from fedmed.utils.config import load_config
from fedmed.utils.logging import get_logger
from fedmed.utils.seed import set_seed

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for slice visualization."""
    parser = argparse.ArgumentParser(
        description="Generate and save an MRI multi-modality slice montage with tumor overlay."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Custom output image path (overrides config output.viz_path).",
    )
    return parser.parse_args()


def create_montage(
    image: np.ndarray | torch.Tensor,
    label: np.ndarray | torch.Tensor,
    out_path: str | Path,
) -> None:
    """Create a 5-panel slice montage and save as PNG.

    Panels:
        1. Modality 0 (T1)
        2. Modality 1 (T1ce)
        3. Modality 2 (T2)
        4. Modality 3 (FLAIR)
        5. Composite Overlay (TC=Red, WT=Green, ET=Blue) on FLAIR

    Args:
        image: 4-channel MRI volume of shape (4, D, H, W).
        label: 3-channel tumor mask of shape (3, D, H, W) where channels are (TC, WT, ET).
        out_path: Output PNG file destination.
    """
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    if isinstance(label, torch.Tensor):
        label = label.detach().cpu().numpy()

    # Find the axial slice (depth axis) with the maximum tumor area (channel 1 is WT)
    wt = label[1]
    slice_sums = wt.sum(axis=(1, 2)) if wt.ndim == 3 else np.array([0])
    best_slice_idx = int(np.argmax(slice_sums)) if slice_sums.max() > 0 else image.shape[1] // 2

    # Extract 2D slices: (H, W)
    t1_slice = image[0, best_slice_idx, :, :]
    t1ce_slice = image[1, best_slice_idx, :, :]
    t2_slice = image[2, best_slice_idx, :, :]
    flair_slice = image[3, best_slice_idx, :, :]

    tc_slice = label[0, best_slice_idx, :, :]
    wt_slice = label[1, best_slice_idx, :, :]
    et_slice = label[2, best_slice_idx, :, :]

    # Normalize FLAIR slice for background display [0, 1]
    bg = (flair_slice - flair_slice.min()) / (flair_slice.max() - flair_slice.min() + 1e-8)
    rgb_overlay = np.stack([bg, bg, bg], axis=-1)

    # Blend tumor colors: WT (Green), TC (Red), ET (Yellow/Cyan)
    # TC = Red
    rgb_overlay[tc_slice > 0.5, 0] = 0.9
    rgb_overlay[tc_slice > 0.5, 1] = 0.2
    rgb_overlay[tc_slice > 0.5, 2] = 0.2
    # WT = Green (where TC is not present)
    green_mask = (wt_slice > 0.5) & ~(tc_slice > 0.5)
    rgb_overlay[green_mask, 0] = 0.2
    rgb_overlay[green_mask, 1] = 0.9
    rgb_overlay[green_mask, 2] = 0.2
    # ET = Blue/Yellow accent
    rgb_overlay[et_slice > 0.5, 0] = 0.9
    rgb_overlay[et_slice > 0.5, 1] = 0.9
    rgb_overlay[et_slice > 0.5, 2] = 0.1

    fig, axes = plt.subplots(1, 5, figsize=(20, 4.5))
    titles = [
        "T1 Modality",
        "T1ce Modality",
        "T2 Modality",
        "FLAIR Modality",
        "Tumor Overlay (WT/TC/ET)",
    ]
    slices = [t1_slice, t1ce_slice, t2_slice, flair_slice, rgb_overlay]

    for i, (ax, sl, title) in enumerate(zip(axes, slices, titles)):
        if i == 4:
            ax.imshow(sl)
        else:
            ax.imshow(sl, cmap="gray")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.axis("off")

    plt.tight_layout()
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_file), dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("Saved slice montage successfully.")


def main() -> None:
    """Load sample and write montage."""
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]))

    out_path = args.output or cfg.get("output", {}).get("viz_path", "outputs/slice_montage.png")
    loaders = get_loaders(cfg)
    batch = next(iter(loaders["train"]))

    # Batch image is (B, 4, D, H, W), label is (B, 3, D, H, W)
    sample_img = batch["image"][0]
    sample_lbl = batch["label"][0]

    create_montage(sample_img, sample_lbl, out_path)


if __name__ == "__main__":
    main()
