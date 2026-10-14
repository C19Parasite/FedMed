# Centralized Baseline Training and Evaluation Guide

This guide documents the centralized 3D U-Net baseline pipeline for multi-sequence brain tumor MRI segmentation (BraTS). The pipeline produces the benchmark Week 1 baseline against which federated learning models (Prompts 5–7) are compared.

---

## 1. How to Run

### Fast Smoke Test (CPU-Friendly)
To verify your environment, dataset pipelines, model architectures, and metrics logging:
```bash
python scripts/train_baseline.py --config configs/smoke_test.yaml --smoke
```

### Full Centralized Baseline Training
To run full centralized training on the BraTS dataset:
```bash
python scripts/train_baseline.py --config configs/base.yaml --device cuda
```

### Evaluation on Test or Validation Split
To evaluate an existing checkpoint:
```bash
python scripts/evaluate.py --config configs/base.yaml --checkpoint outputs/runs/baseline/<timestamp>/checkpoints/best.pt --split test --hd95
```
> [!NOTE]
> Evaluation is strictly prohibited on the `train` split. The evaluation script will terminate with an error if `--split train` is passed.

---

## 2. Configuration Reference Table (`training:` section)

| Key | Type | Default | Description |
|---|---|---|---|
| `epochs` | `int` | `5` | Total training epochs. |
| `device` | `str` | `"auto"` | Execution device (`"auto"`, `"cpu"`, or `"cuda"`). |
| `optimizer.name` | `str` | `"adamw"` | Optimizer algorithm (`"adamw"`, `"sgd"`, or `"adam"`). |
| `optimizer.lr` | `float` | `0.0002` | Peak learning rate after warmup. |
| `optimizer.weight_decay` | `float` | `0.00001` | Weight decay penalty (biases & norm weights excluded). |
| `optimizer.betas` | `list[float]` | `[0.9, 0.999]` | Adam/AdamW beta parameters. |
| `optimizer.momentum` | `float` | `0.9` | SGD momentum parameter. |
| `optimizer.nesterov` | `bool` | `false` | SGD Nesterov momentum flag. |
| `scheduler.name` | `str` | `"cosine"` | LR scheduler (`"cosine"`, `"poly"`, or `"none"`). |
| `scheduler.warmup_epochs` | `int` | `1` | Number of linear warmup epochs. |
| `scheduler.min_lr` | `float` | `1.0e-6` | Final minimum learning rate. |
| `scheduler.poly_power` | `float` | `1.0` | Polynomial decay power factor. |
| `amp.enabled` | `bool` | `true` | Automatic Mixed Precision (auto-disabled on CPU). |
| `amp.dtype` | `str` | `"float16"` | AMP precision (`"float16"` with GradScaler, or `"bfloat16"`). |
| `grad_clip.enabled` | `bool` | `true` | Gradient norm clipping flag. |
| `grad_clip.max_norm` | `float` | `1.0` | Maximum pre-clip L2 gradient norm threshold. |
| `grad_accum_steps` | `int` | `1` | Number of batches to accumulate gradients over. |
| `val_every_n_epochs` | `int` | `1` | Validation interval (epochs). |
| `sliding_window.roi_size` | `list[int]` | `[96, 96, 96]` | Sliding window patch dimension (must match stride divisibility). |
| `sliding_window.sw_batch_size` | `int` | `1` | Inference batch size for sliding window. |
| `sliding_window.overlap` | `float` | `0.25` | Overlap ratio between adjacent inference windows. |
| `sliding_window.mode` | `str` | `"gaussian"` | Blending mode (`"gaussian"` or `"constant"`). |
| `early_stopping.enabled` | `bool` | `true` | Enable early termination on validation plateaus. |
| `early_stopping.patience` | `int` | `10` | Validation rounds without improvement before stopping. |
| `early_stopping.min_delta` | `float` | `0.0001` | Minimum change to qualify as an improvement. |
| `early_stopping.monitor` | `str` | `"val_dice_mean"` | Metric monitored for improvement. |
| `early_stopping.mode` | `str` | `"max"` | Direction of improvement (`"max"` or `"min"`). |
| `checkpoint.dir` | `str` | `"checkpoints"` | Directory name for saved model weights. |
| `checkpoint.save_every_n_epochs` | `int` | `5` | Interval for saving periodic `epoch_<n>.pt`. |
| `checkpoint.keep_last_k` | `int` | `3` | Number of periodic epoch checkpoints to retain. |
| `resume.enabled` | `bool` | `false` | Resume training from existing checkpoint. |
| `resume.path` | `str \| null` | `null` | `"auto"` or explicit path to `.pt` checkpoint file. |
| `logging.run_name` | `str` | `"baseline"` | Identifier for experiment run directory. |
| `logging.output_dir` | `str` | `"outputs/runs"` | Root output directory. |
| `logging.tensorboard` | `bool` | `true` | Stream metrics and scalar events to TensorBoard. |
| `deterministic.enabled` | `bool` | `true` | Enable reproducible seed settings. |
| `max_steps_per_epoch` | `int \| null` | `null` | Cap steps per epoch (useful for smoke tests and profiling). |

---

## 3. Output Directory Layout

Each execution creates an isolated, timestamped run directory:
```
outputs/runs/baseline/<UTC_TIMESTAMP>/
├── config.yaml              # Resolved canonical config snapshot
├── environment.json         # Sanitized runtime and hardware metadata
├── checkpoints/
│   ├── best.pt              # Small deployable artifact: model weights & metadata only
│   ├── best.meta.json       # Sidecar JSON with epoch, metric, and RNG metadata
│   ├── latest.pt            # Full checkpoint (model, optimizer, scheduler, scaler, RNG)
│   ├── latest.meta.json     # Sidecar JSON metadata
│   ├── epoch_<N>.pt         # Periodic checkpoints (pruned by keep_last_k)
│   └── epoch_<N>.meta.json
├── results/
│   ├── metrics.csv          # Fixed-schema epoch log parsed by React dashboard
│   └── baseline_metrics.json# Benchmark metrics on test split
└── tensorboard/             # TensorBoard event files (train/ and val/ scalars)
```

---

## 4. Metrics CSV Schema (`results/metrics.csv`)

The columns follow a strictly ordered schema required by the Week 3 React dashboard:
```csv
epoch,train_loss,val_loss,val_dice_tc,val_dice_wt,val_dice_et,val_dice_mean,val_hd95_tc,val_hd95_wt,val_hd95_et,val_hd95_mean,lr,grad_norm_mean,epoch_time_s,samples_seen,is_best
```
- **Empty cells**: For epochs without validation or when HD95 is omitted, the respective cells are left empty (`""`), not written as `0` or `NaN`.
- **Atomic writes**: Written row by row and immediately flushed.
- **Resume behavior**: On resume, rows with `epoch >= start_epoch` are automatically truncated to prevent duplicate entries.

---

## 5. Baseline JSON Schema (`results/baseline_metrics.json`)

```json
{
  "split": "test",
  "checkpoint_epoch": 5,
  "n_volumes": 10,
  "dice": {
    "tc": 0.8124,
    "wt": 0.8931,
    "et": 0.7845,
    "mean": 0.8300
  },
  "hd95": {
    "tc": 6.42,
    "wt": 4.15,
    "et": 5.89,
    "mean": 5.4867
  },
  "empty_policy": "brats",
  "threshold": 0.5,
  "sliding_window": {
    "roi_size": [96, 96, 96],
    "overlap": 0.25,
    "sw_batch_size": 1,
    "mode": "gaussian"
  },
  "model": {
    "name": "monai_unet",
    "parameters": 19404,
    "size_mb": 0.07,
    "norm": "GroupNorm"
  },
  "training": {
    "epochs_run": 5,
    "best_epoch": 5,
    "optimizer": "adamw",
    "lr": 0.0002,
    "amp": true
  },
  "seed": 42,
  "config_hash": "e581fa89af8c",
  "environment": {
    "python": "3.12.10",
    "torch": "2.13.0+cpu",
    "monai": "1.6.1",
    "cuda_available": false,
    "platform": "Windows"
  },
  "timestamp_utc": "2026-10-14T18:45:00.000000+00:00"
}
```
> [!IMPORTANT]
> **Privacy Rule**: `baseline_metrics.json` and `environment.json` never contain local file paths, usernames, hostnames, patient IDs, or raw voxel arrays.

---

## 6. Resuming Training

If an experiment run is interrupted (via `Ctrl+C`, spot instance preemption, or network disconnect):
1. State is automatically preserved in `latest.pt`.
2. Resume using the auto-detection flag:
   ```bash
   python scripts/train_baseline.py --config configs/base.yaml --resume
   ```
   Or explicitly specify the checkpoint file:
   ```bash
   python scripts/train_baseline.py --config configs/base.yaml --override training.resume.enabled=true training.resume.path=outputs/runs/baseline/<timestamp>/checkpoints/latest.pt
   ```

---

## 7. Determinism and Bit-Reproducibility

- **CPU Reproducibility**: Exact bit-for-bit reproducibility across runs is guaranteed on CPU by restoring Python `random.getstate()`, NumPy `np.random.get_state()`, and PyTorch `torch.get_rng_state()`.
- **GPU Determinism Notes**:
  - `monai.utils.set_determinism(seed)` and `torch.backends.cudnn.deterministic = True` ensure reproducible initialization and forward passes.
  - Due to atomic floating-point additions in CUDA 3D convolutions (e.g. `atomicAdd` in `cuDNN` backward kernels), small numerical variations ($\sim 10^{-7}$) can accumulate across long runs.
  - Setting `deterministic.cudnn_benchmark = false` prevents non-deterministic heuristic algorithm selection.

---

## 8. Runbook: Google Colab / Kaggle GPU Execution

### Step 1: Environment Setup
```bash
!git clone https://github.com/UdiptaG/FedMed.git
%cd FedMed
!pip install -r requirements.txt
!pip install -e .
```

### Step 2: Download / Mount Data
To download the Decathlon Task01 BraTS dataset:
```bash
python scripts/download_data.py --config configs/base.yaml
```

### Step 3: Run Full Baseline
```bash
python scripts/train_baseline.py --config configs/base.yaml --device cuda
```

### Step 4: Estimated Runtime & GPU Resource Guidance

Extrapolated from `scripts/profile_model.py` benchmarks for 3D UNet with `roi_size = [96, 96, 96]`:
- **NVIDIA T4 / RTX 3070 (8 GB VRAM)**:
  - Recommended config: `batch_size: 1`, `sliding_window.sw_batch_size: 1`, `amp.enabled: true`.
  - Peak GPU memory: $\sim 4.2 \text{ GB}$.
  - Estimated throughput: $\sim 45 \text{ seconds/epoch}$ on 340 training volumes.
  - 100 epochs runtime: $\sim 1.25 \text{ hours}$.
- **NVIDIA V100 / A100 (16 GB - 40 GB VRAM)**:
  - Recommended config: `batch_size: 2` or `4`, `amp.enabled: true`, `num_workers: 4`.
  - Peak GPU memory: $\sim 7.8 \text{ GB}$ (batch size 2).
  - Estimated throughput: $\sim 20 \text{ seconds/epoch}$.
  - 100 epochs runtime: $\sim 35 \text{ minutes}$.

### Step 5: Exporting Results Back to Repo
Only the small JSON report and CSV metrics should be committed back to Git:
```bash
cp outputs/runs/baseline/<TIMESTAMP>/results/metrics.csv results/metrics.csv
cp outputs/runs/baseline/<TIMESTAMP>/results/baseline_metrics.json results/baseline_metrics.json
```
> [!CAUTION]
> Never commit heavy `.pt` checkpoint files, raw `.nii.gz` scans, or TensorBoard event logs to the Git repository.
