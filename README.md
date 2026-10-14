# FedMed

**FedMed** is a cross-silo federated learning system for 3D brain tumor segmentation on multi-parametric MRI scans (T1, T1ce, T2, FLAIR) built using **PyTorch**, **MONAI**, and **Flower (`flwr`)**.

---

## Repository Structure

```text
FedMed/
├── configs/
│   ├── base.yaml                  # Full default configuration for 3D segmentation
│   ├── baseline.yaml              # Centralized baseline configuration
│   ├── smoke_test.yaml            # Tiny/fast CI & smoke test configuration
│   └── federated_hospitals.yaml   # 3 Mock Hospital Nodes configuration
├── data/                          # Local/cached/synthetic data directories (gitignored)
├── fedmed/
│   ├── data/                      # Datasets, transforms, loaders, splits, synthetic cohorts
│   ├── models/                    # Model factory, custom 3D U-Net, norm policies, params utilities
│   │   ├── factory.py             # build_model factory (monai_unet, segresnet, dynunet, custom)
│   │   ├── unet3d.py              # UNet3DCustom (reference) & UNet3D wrapper
│   │   ├── norm.py                # assert_no_batchnorm, ROI divisibility, GroupNorm validation
│   │   └── params.py              # count/get/set_parameters, vector round-trip, model_size_mb
│   ├── training/                  # Losses, metrics, trainer loop
│   │   ├── losses.py              # build_loss, DeepSupervisionLoss wrapper, Dice/CE/Focal
│   │   ├── metrics.py             # SegmentationMetrics accumulator, dice/hd95_per_channel
│   │   └── trainer.py             # Centralized baseline trainer
│   ├── fl/                        # Flower client, server, aggregation strategy
│   └── utils/                     # Reproducible seeding, privacy logger, config validation
├── scripts/
│   ├── profile_model.py           # Model runtime, parameter count, and GPU headroom profiler
│   ├── download_data.py           # Decathlon / BraTS data download and verification
│   ├── smoke_test_data.py         # Batch tensor loading and range inspection
│   ├── visualize_sample.py        # 5-panel slice montage generator (modality + tumor overlay)
│   ├── train_baseline.py          # Centralized baseline training CLI
│   ├── evaluate.py                # Checkpoint evaluation CLI
│   └── run_federated_nodes.py     # Multi-node Flower hospital runner and orchestrator
├── tests/                         # Comprehensive pytest test suite (CPU-only, fast)
├── requirements.txt               # Pinned core dependencies
├── pyproject.toml                 # Package definition for editable installation
└── README.md
```

---

## Models, Normalization & Architecture Reference

### Supported Architectures (`build_model(cfg)`)

The model factory supports four 3D architectures configured under `model.name`:

1. **`monai_unet`** (`monai.networks.nets.UNet`):
   - Standard 3D residual U-Net with configurable channel progression, strides, and normalization.
2. **`segresnet`** (`monai.networks.nets.SegResNet`):
   - Encoder-decoder ResNet backbone with GroupNorm (8 groups) designed for memory-efficient volumetric segmentation.
3. **`dynunet`** (`monai.networks.nets.DynUNet`):
   - Dynamic U-Net with deep supervision support.
   - **Deep Supervision Output Contract**: In training mode (`.train()`), forward pass returns a 6D stacked tensor `[B, N, C, D, H, W]` where `N = deep_supr_num + 1`. In evaluation mode (`.eval()`), returns a standard 5D tensor `[B, C, D, H, W]`.
4. **`unet3d_custom`** (`fedmed.models.unet3d.UNet3DCustom`):
   - A transparent, readable from-scratch 3D U-Net implementation in `fedmed/models/unet3d.py`.
   - Features modular encoder stages, bottleneck, decoder stages with skip connections via concatenation, optional residual connections, and strict federated normalization enforcement.

### Normalization Policy (Why BatchNorm is Banned)

**Batch Normalization (`BatchNorm1d/2d/3d`) is strictly prohibited in FedMed.**
Only `GroupNorm` and `InstanceNorm` are permitted.

- **Non-IID Drift**: Under non-IID client distributions across hospital silos, local running mean and variance diverge drastically.
- **FedAvg Averaging Failure**: FedAvg averages model weights; batch norm running statistics are not weights, making empirical averaging mathematically ungrounded.
- **Differential Privacy Incompatibility**: Per-sample gradient clipping (required for DP in later sprints) is incompatible with batch statistics that share information across samples.
- **Enforcement**:
  - `build_model(cfg)` and `validate_config(cfg)` reject any config with `norm.type="batch"`.
  - The `assert_no_batchnorm(model)` runtime guard scans all submodules and raises `ValueError` if any BatchNorm layer exists.
  - GroupNorm requires that `num_groups` divides every channel width in the model.
  - `validate_roi_for_model(cfg)` checks that each ROI dimension is divisible by the total downsampling factor (product of strides).

---

## Loss Functions & Deep Supervision

Selected via `loss.name` in configuration:
- **`dice_ce`**: `monai.losses.DiceCELoss` (Dice + Cross-Entropy)
- **`dice_focal`**: `monai.losses.DiceFocalLoss` (Dice + Focal Loss)
- **`dice`**: `monai.losses.DiceLoss`

### Multi-Label Configuration Rationale
BraTS defines 3 overlapping binary tumor channels: **TC (Tumor Core)**, **WT (Whole Tumor)**, and **ET (Enhancing Tumor)**.
- `sigmoid=True`: Independent sigmoid activation per channel (each voxel can belong to multiple subregions).
- `softmax=False`: Softmax forces mutually exclusive probabilities across channels, breaking overlapping subregions.
- `to_onehot_y=False`: Ground truth is already a 3-channel binary tensor `[B, 3, D, H, W]`.
- `include_background=True`: All 3 channels represent active tumor compartments (no background class channel).

### DeepSupervisionLoss Wrapper
When deep supervision is active (e.g. `dynunet` in training mode), `DeepSupervisionLoss` applies halving weights $(1, 1/2, 1/4, \dots)$ normalized to sum to $1.0$ across the stages. If intermediate stages have differing spatial resolutions, trilinear interpolation aligns them to the target. It acts as a transparent no-op when deep supervision is disabled.

---

## Metrics & Empty Ground Truth Policies

### Evaluation Metrics (`SegmentationMetrics`)
Stateful accumulator accumulating per-volume statistics:
- **Dice**: `dice_tc`, `dice_wt`, `dice_et`, and `dice_mean`.
- **HD95** (95th-percentile Hausdorff Distance): `hd95_tc`, `hd95_wt`, `hd95_et`, and `hd95_mean`.
- Thread-safe and CPU-safe: tensors are moved to CPU before surface distance calculations.
- Spacing support: voxel spacing in mm (default `[1.0, 1.0, 1.0]`).
- Cap handling: `inf` and `nan` distances are replaced by `max_hd95` (default `373.13` mm, the diagonal of a standard 240x240x155 BraTS volume: $\sqrt{240^2 + 240^2 + 155^2} \approx 373.13$).

### Empty Ground Truth Policy (`metrics.empty_policy`)
1. **`brats`** (Default):
   - Ground truth empty and prediction empty: Dice = $1.0$, HD95 = $0.0$.
   - Ground truth empty and prediction non-empty: Dice = $0.0$, HD95 = $373.13$.
   - Ground truth non-empty and prediction empty: Dice = $0.0$, HD95 = $373.13$.
2. **`skip`**:
   - Excludes volumes with empty ground truth from that channel's average using NaN-aware means.
   - Reports valid case counts via `n_valid_tc`, `n_valid_wt`, and `n_valid_et`.

---

## Model Profiling CLI (`scripts/profile_model.py`)

Profile parameter counts, model memory footprint, forward+backward execution latency, and GPU headroom:

```bash
# Profile tiny smoke test configuration on CPU:
python scripts/profile_model.py --config configs/smoke_test.yaml --device cpu

# Profile base configuration on CPU:
python scripts/profile_model.py --config configs/base.yaml --device cpu

# Profile on CUDA with mixed precision (AMP) and batch size sweep (1, 2, 4):
python scripts/profile_model.py --config configs/base.yaml --device cuda --amp --sweep
```

The script reports peak VRAM usage via `torch.cuda.max_memory_allocated` and outputs suitability recommendations for 8 GB and 12 GB GPU hardware budgets.

---

## Setup & Testing

### 1. Installation
```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

### 2. Test Suite
Run the full CPU-only test suite:
```bash
pytest tests/ -v
```
All unit and integration tests run deterministically without internet access or GPU dependencies.

---

## Centralized Baseline Training & Evaluation (Week 1)

FedMed includes a complete centralized training and evaluation pipeline producing benchmark segmentation performance on the BraTS multi-sequence dataset.

- **Guide & Runbook**: See [docs/baseline.md](docs/baseline.md) for full configuration reference, directory layout, CSV/JSON schemas, determinism notes, and Google Colab / Kaggle runbooks.
- **Fast Smoke Test**:
  ```bash
  python scripts/train_baseline.py --config configs/smoke_test.yaml --smoke
  ```
- **Full Centralized Training**:
  ```bash
  python scripts/train_baseline.py --config configs/base.yaml --device cuda
  ```
- **Evaluation**:
  ```bash
  python scripts/evaluate.py --config configs/base.yaml --checkpoint <checkpoint_path> --split test --hd95
  ```

