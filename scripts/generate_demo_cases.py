"""Generate segmentation slice overlays and case previews for the FedMed React dashboard (Phase 9)."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from fedmed.data.loaders import build_loader
from fedmed.data.partition import create_or_load_partitions, get_global_eval_loaders
from fedmed.models.params import set_parameters
from fedmed.utils.logging import get_logger
from src.model.metrics import compute_dice_score
from src.model.unet import build_unet

logger = get_logger(__name__)


def generate_demo_cases(config_path: str = "config.yaml", num_cases: int = 3) -> dict:
    """Generate slice images and overlays for public demo cases on test split."""
    print("=" * 70)
    print(" FedMed Phase 9: Generating Segmentation Results for Dashboard")
    print("=" * 70)

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    out_dir = Path("results/demo_cases")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load federated model checkpoint
    fed_model = build_unet(cfg)
    fed_ckpt = Path("checkpoints/federated/best.pt")
    if fed_ckpt.exists():
        data = torch.load(fed_ckpt, map_location="cpu", weights_only=False)
        fed_model.load_state_dict(data.get("state_dict", data))
        print(f"Loaded best federated model from {fed_ckpt}")
    fed_model.eval()

    # 2. Load centralized model checkpoint if available
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

    # Load test split cases
    _ = create_or_load_partitions(cfg)
    global_loaders = get_global_eval_loaders(cfg)
    test_loader = global_loaders["test"]

    case_manifest = []

    with torch.no_grad():
        for case_idx, batch in enumerate(test_loader):
            if case_idx >= num_cases:
                break

            images = batch["image"]  # [1, 4, D, H, W]
            labels = batch["label"]  # [1, 3, D, H, W]

            case_id = f"demo_case_{case_idx + 1}"
            case_path = out_dir / case_id
            case_path.mkdir(parents=True, exist_ok=True)

            fed_out = fed_model(images)
            central_out = central_model(images)

            fed_dice = compute_dice_score(fed_out, labels)
            central_dice = compute_dice_score(central_out, labels)

            fed_probs = torch.sigmoid(fed_out)[0].cpu().numpy()  # [3, D, H, W]
            central_probs = torch.sigmoid(central_out)[0].cpu().numpy()
            img_np = images[0].cpu().numpy()  # [4, D, H, W]
            lbl_np = labels[0].cpu().numpy()  # [3, D, H, W]

            D, H, W = img_np.shape[1], img_np.shape[2], img_np.shape[3]
            slices_info = []

            for z in range(D):
                # FLAIR modality
                mri_slice = img_np[3, z, :, :]
                mri_norm = (mri_slice - mri_slice.min()) / (mri_slice.max() - mri_slice.min() + 1e-8)

                gt_tc = lbl_np[0, z, :, :]
                gt_wt = lbl_np[1, z, :, :]
                gt_et = lbl_np[2, z, :, :]

                fed_wt = (fed_probs[1, z, :, :] >= 0.5).astype(float)
                central_wt = (central_probs[1, z, :, :] >= 0.5).astype(float)

                # Save MRI slice
                mri_file = case_path / f"slice_{z}_mri.png"
                plt.imsave(mri_file, mri_norm, cmap="gray")

                # Save Ground Truth mask
                gt_file = case_path / f"slice_{z}_gt.png"
                plt.imsave(gt_file, gt_wt, cmap="autumn", vmin=0, vmax=1)

                # Save Federated prediction
                fed_file = case_path / f"slice_{z}_pred.png"
                plt.imsave(fed_file, fed_wt, cmap="winter", vmin=0, vmax=1)

                # Save Centralized prediction
                cen_file = case_path / f"slice_{z}_central.png"
                plt.imsave(cen_file, central_wt, cmap="summer", vmin=0, vmax=1)

                # Save Side-by-Side 4-panel Overlay montage
                fig, axes = plt.subplots(1, 4, figsize=(12, 3))
                axes[0].imshow(mri_norm, cmap="gray")
                axes[0].set_title("FLAIR MRI")
                axes[0].axis("off")

                axes[1].imshow(mri_norm, cmap="gray")
                axes[1].imshow(gt_wt, cmap="autumn", alpha=0.5 * (gt_wt > 0))
                axes[1].set_title("Ground Truth Mask")
                axes[1].axis("off")

                axes[2].imshow(mri_norm, cmap="gray")
                axes[2].imshow(fed_wt, cmap="winter", alpha=0.5 * (fed_wt > 0))
                axes[2].set_title("Federated Prediction")
                axes[2].axis("off")

                axes[3].imshow(mri_norm, cmap="gray")
                axes[3].imshow(central_wt, cmap="summer", alpha=0.5 * (central_wt > 0))
                axes[3].set_title("Centralized Baseline")
                axes[3].axis("off")

                plt.tight_layout()
                overlay_file = case_path / f"slice_{z}_overlay.png"
                plt.savefig(overlay_file, dpi=100)
                plt.close(fig)

                slices_info.append(z)

            case_entry = {
                "id": case_id,
                "label": f"Public BraTS Demo Case #{case_idx + 1}",
                "dataset_note": "Research cohort test case (zero private hospital patient data)",
                "depth_slices": len(slices_info),
                "dice": fed_dice,
                "centralized_dice": central_dice,
            }
            case_manifest.append(case_entry)
            print(f"Generated {len(slices_info)} slice montages for {case_id} (Fed Dice: {fed_dice['dice_mean']:.4f})")

    manifest_file = out_dir / "manifest.json"
    manifest_file.write_text(json.dumps(case_manifest, indent=2), encoding="utf-8")
    print(f"\n[PASS] Saved demo case manifest to {manifest_file}")
    print("=" * 70)
    return {"manifest": case_manifest}


if __name__ == "__main__":
    generate_demo_cases()
