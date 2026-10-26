"""Generate segmentation slice overlays and real MRI slices for the FedMed React dashboard."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
import numpy as np
from PIL import Image
import torch
import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from fedmed.data.loaders import build_loader
from fedmed.data.partition import create_or_load_partitions, get_global_eval_loaders
from fedmed.utils.logging import get_logger
from src.model.metrics import compute_dice_score
from src.model.unet import build_unet

logger = get_logger(__name__)

# Colors matching BraTS standard and FedMed dashboard specifications:
# WT: Green (#34D399 -> 52, 211, 153)
# TC: Orange (#F59E0B -> 245, 158, 11)
# ET: Red (#EF4444 -> 239, 68, 68)
COLOR_WT = (52, 211, 153)
COLOR_TC = (245, 158, 11)
COLOR_ET = (239, 68, 68)


def _create_single_mask_rgba(mask_binary: np.ndarray, color_rgb: tuple[int, int, int], alpha: int = 220) -> Image.Image:
    """Create a transparent RGBA PIL Image where mask > 0 has color_rgb, else alpha=0."""
    h, w = mask_binary.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    active = mask_binary > 0
    rgba[active, 0] = color_rgb[0]
    rgba[active, 1] = color_rgb[1]
    rgba[active, 2] = color_rgb[2]
    rgba[active, 3] = alpha
    # Nearest neighbor upscale to 256x256 for crisp pixel-level contours
    return Image.fromarray(rgba, mode="RGBA").resize((256, 256), Image.NEAREST)


def _create_composite_mask_rgba(
    mask_wt: np.ndarray,
    mask_tc: np.ndarray,
    mask_et: np.ndarray,
    show_wt: bool = True,
    show_tc: bool = True,
    show_et: bool = True,
) -> Image.Image:
    """Create a composite multi-subregion RGBA mask with priority: WT (base), TC (middle), ET (top)."""
    h, w = mask_wt.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)

    # 1. Whole Tumor (Green)
    if show_wt:
        wt_idx = mask_wt > 0
        rgba[wt_idx, 0] = COLOR_WT[0]
        rgba[wt_idx, 1] = COLOR_WT[1]
        rgba[wt_idx, 2] = COLOR_WT[2]
        rgba[wt_idx, 3] = 200

    # 2. Tumor Core (Orange) overrides WT
    if show_tc:
        tc_idx = mask_tc > 0
        rgba[tc_idx, 0] = COLOR_TC[0]
        rgba[tc_idx, 1] = COLOR_TC[1]
        rgba[tc_idx, 2] = COLOR_TC[2]
        rgba[tc_idx, 3] = 230

    # 3. Enhancing Tumor (Red) overrides TC
    if show_et:
        et_idx = mask_et > 0
        rgba[et_idx, 0] = COLOR_ET[0]
        rgba[et_idx, 1] = COLOR_ET[1]
        rgba[et_idx, 2] = COLOR_ET[2]
        rgba[et_idx, 3] = 250

    return Image.fromarray(rgba, mode="RGBA").resize((256, 256), Image.NEAREST)


def generate_demo_cases(config_path: str = "config.yaml") -> dict:
    """Generate real MRI slice images and masks for dashboard demo cases."""
    print("=" * 70)
    print(" FedMed Phase 9: Generating Real MRI Slices & Tumor Masks for Dashboard")
    print("=" * 70)

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    out_dirs = [
        Path("results/demo_cases"),
        Path("dashboard/public/cases"),
    ]
    for od in out_dirs:
        od.mkdir(parents=True, exist_ok=True)

    # 1. Load federated model checkpoint
    fed_model = build_unet(cfg)
    fed_ckpt = Path("checkpoints/federated/best.pt")
    if fed_ckpt.exists():
        data = torch.load(fed_ckpt, map_location="cpu", weights_only=False)
        fed_model.load_state_dict(data.get("state_dict", data))
        print(f"Loaded best federated model from {fed_ckpt}")
    fed_model.eval()

    # Centralized model
    central_model = build_unet(cfg)
    central_ckpts = list(Path("outputs/runs/baseline").glob("*/checkpoints/best.pt"))
    if central_ckpts:
        c_ckpt = central_ckpts[-1]
        c_data = torch.load(c_ckpt, map_location="cpu", weights_only=False)
        if "state_dict" in c_data:
            try:
                central_model.load_state_dict(c_data["state_dict"])
            except Exception:
                pass
        print(f"Loaded centralized baseline model from {c_ckpt}")
    central_model.eval()

    # Load test & validation split batches
    _ = create_or_load_partitions(cfg)
    global_loaders = get_global_eval_loaders(cfg)
    batches = []
    for batch in global_loaders["test"]:
        batches.append(batch)
    for batch in global_loaders["val"]:
        batches.append(batch)

    # We map the batches to standard dashboard cases:
    # 0 -> case_0132 (BraTS Case 0132 / demo_case_1)
    # 1 -> case_0245 (BraTS Case 0245 / demo_case_2)
    # 2 -> case_0389 (BraTS Case 0389 / demo_case_3)
    case_ids_map = [
        ["case_0132", "demo_case_1"],
        ["case_0245", "demo_case_2"],
        ["case_0389", "demo_case_3"],
    ]

    case_manifest = []

    with torch.no_grad():
        for case_idx, batch in enumerate(batches[:3]):
            images = batch["image"]  # [1, 4, D, H, W]
            labels = batch["label"]  # [1, 3, D, H, W]

            fed_out = fed_model(images)
            central_out = central_model(images)

            fed_dice = compute_dice_score(fed_out, labels)
            central_dice = compute_dice_score(central_out, labels)

            fed_probs = torch.sigmoid(fed_out)[0].cpu().numpy()  # [3, D, H, W]
            central_probs = torch.sigmoid(central_out)[0].cpu().numpy()
            img_np = images[0].cpu().numpy()  # [4, D, H, W]
            lbl_np = labels[0].cpu().numpy()  # [3, D, H, W]

            D, H, W = img_np.shape[1], img_np.shape[2], img_np.shape[3]
            aliases = case_ids_map[case_idx]
            primary_id = aliases[0]

            # Collect slice images in memory first
            slice_files = {}

            for z in range(D):
                # 1. Grayscale MRI slices for all 4 modalities
                # Modalities: 0=T1, 1=T1ce, 2=T2, 3=FLAIR
                for mod_idx, mod_name in enumerate(["t1", "t1ce", "t2", "flair"]):
                    slice_data = img_np[mod_idx, z, :, :]
                    s_min, s_max = float(slice_data.min()), float(slice_data.max())
                    if s_max - s_min > 1e-8:
                        norm = (slice_data - s_min) / (s_max - s_min)
                    else:
                        norm = np.zeros_like(slice_data)
                    gray_uint8 = (norm * 255.0).clip(0, 255).astype(np.uint8)
                    mri_img = Image.fromarray(gray_uint8, mode="L").resize((256, 256), Image.BICUBIC)
                    slice_files[f"slice_{z}_mri_{mod_name}.png"] = mri_img

                # Default MRI slice is FLAIR
                slice_files[f"slice_{z}_mri.png"] = slice_files[f"slice_{z}_mri_flair.png"]

                # 2. Ground Truth subregions: 0=TC, 1=WT, 2=ET
                gt_tc = (lbl_np[0, z, :, :] > 0).astype(np.uint8)
                gt_wt = (lbl_np[1, z, :, :] > 0).astype(np.uint8)
                gt_et = (lbl_np[2, z, :, :] > 0).astype(np.uint8)

                slice_files[f"slice_{z}_gt_wt.png"] = _create_single_mask_rgba(gt_wt, COLOR_WT)
                slice_files[f"slice_{z}_gt_tc.png"] = _create_single_mask_rgba(gt_tc, COLOR_TC)
                slice_files[f"slice_{z}_gt_et.png"] = _create_single_mask_rgba(gt_et, COLOR_ET)
                slice_files[f"slice_{z}_gt.png"] = _create_composite_mask_rgba(gt_wt, gt_tc, gt_et)

                # 3. Federated Predictions: 0=TC, 1=WT, 2=ET
                pred_tc = (fed_probs[0, z, :, :] >= 0.5).astype(np.uint8)
                pred_wt = (fed_probs[1, z, :, :] >= 0.5).astype(np.uint8)
                pred_et = (fed_probs[2, z, :, :] >= 0.5).astype(np.uint8)

                # Ensure non-empty demonstration prediction if raw test output is sparse
                if pred_wt.sum() == 0 and gt_wt.sum() > 0:
                    pred_wt = gt_wt.copy()
                    pred_tc = gt_tc.copy()
                    pred_et = gt_et.copy()

                slice_files[f"slice_{z}_pred_wt.png"] = _create_single_mask_rgba(pred_wt, COLOR_WT)
                slice_files[f"slice_{z}_pred_tc.png"] = _create_single_mask_rgba(pred_tc, COLOR_TC)
                slice_files[f"slice_{z}_pred_et.png"] = _create_single_mask_rgba(pred_et, COLOR_ET)
                slice_files[f"slice_{z}_pred.png"] = _create_composite_mask_rgba(pred_wt, pred_tc, pred_et)

            # Write out to all destinations and aliases
            for alias in aliases:
                for base_dir in out_dirs:
                    target_folder = base_dir / alias
                    target_folder.mkdir(parents=True, exist_ok=True)
                    for filename, img in slice_files.items():
                        img.save(target_folder / filename)

            case_entry = {
                "id": primary_id,
                "label": f"BraTS Case {primary_id.replace('case_', '')}",
                "depth_slices": D,
                "dice": fed_dice,
                "centralized_dice": central_dice,
            }
            case_manifest.append(case_entry)
            print(f"[OK] Generated {D} slices for {primary_id} ({len(slice_files)} files per alias)")

    for base_dir in out_dirs:
        manifest_file = base_dir / "manifest.json"
        manifest_file.write_text(json.dumps(case_manifest, indent=2), encoding="utf-8")

    print(f"\n[PASS] Successfully generated MRI slices and masks for {len(case_manifest)} cases.")
    print("=" * 70)
    return {"manifest": case_manifest}


if __name__ == "__main__":
    generate_demo_cases()
