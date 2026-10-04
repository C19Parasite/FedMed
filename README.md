# FedMed

**FedMed** is a cross-silo federated learning system for 3D brain tumor segmentation on multi-parametric MRI scans (T1, T1ce, T2, FLAIR) built using **PyTorch**, **MONAI**, and **Flower (`flwr`)**.

---

## Repository Structure

```text
FedMed/
├── configs/
│   ├── baseline.yaml              # Centralized baseline configuration
│   └── federated_hospitals.yaml   # 3 Mock Hospital Nodes configuration
├── data/                          # Local/cached/synthetic data directories (gitignored)
├── fedmed/
│   ├── data/                      # Datasets, transforms, loaders, splits, synthetic cohorts
│   ├── models/                    # 3D U-Net network architecture (InstanceNorm)
│   ├── training/                  # Centralized trainer, DiceCELoss, BraTS metrics
│   ├── fl/                        # Flower client, server, aggregation strategy
│   └── utils/                     # Reproducible seeding, privacy logger, config loader
├── scripts/
│   ├── download_data.py           # Decathlon / BraTS data download and verification
│   ├── smoke_test_data.py         # Batch tensor loading and range inspection
│   ├── visualize_sample.py        # 5-panel slice montage generator (modality + tumor overlay)
│   ├── train_baseline.py          # Centralized baseline training CLI
│   ├── evaluate.py                # Checkpoint evaluation CLI
│   └── run_federated_nodes.py     # Multi-node Flower hospital runner and orchestrator
├── tests/                         # Pytest unit and integration test suite
├── requirements.txt               # Pinned core dependencies
├── pyproject.toml                 # Package definition for editable installation
└── README.md
```

---

## Setup Instructions

### 1. Create a Virtual Environment

FedMed requires Python 3.10 or newer.

```bash
# Using standard venv
python -m venv .venv

# Activate on Windows (PowerShell):
.\.venv\Scripts\Activate.ps1

# Activate on Linux / macOS:
source .venv/bin/activate
```

Alternatively, if using `uv`:
```bash
uv venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install Dependencies

Install the package in editable mode along with development dependencies:

```bash
pip install -r requirements.txt
pip install -e .
```

### 3. Run the Test Suite

Execute pytest across all unit and integration tests:

```bash
pytest
```

---

## Week 1 Workflows & Usage

### 1. Data Pipeline & Smoke Test

Inspect tensor shapes, types, and min/max value ranges from a generated batch:
```bash
python scripts/smoke_test_data.py --config configs/baseline.yaml
```

Generate a 5-panel slice montage (T1, T1ce, T2, FLAIR, and multi-channel tumor overlay):
```bash
python scripts/visualize_sample.py --config configs/baseline.yaml
```

### 2. Centralized 3D U-Net Baseline Training

Train the standard centralized 3D U-Net model with InstanceNorm:
```bash
python scripts/train_baseline.py --config configs/baseline.yaml --epochs 5
```

Evaluate a saved checkpoint on validation or test splits:
```bash
python scripts/evaluate.py --config configs/baseline.yaml --checkpoint checkpoints/best_metric_model.pt --split val
```

### 3. Distributed Flower Node Scaffolding (3 Mock Hospital Silos)

Launch the multi-node federated training orchestration connecting the 3 mock hospital silos:
```bash
python scripts/run_federated_nodes.py --config configs/federated_hospitals.yaml --role orchestrate
```

Alternatively, launch the central aggregation server and individual hospital clients as separate processes:
```bash
# Terminal 1 - Central Server:
python scripts/run_federated_nodes.py --role server --server-address 127.0.0.1:8080

# Terminal 2 - Hospital 1:
python scripts/run_federated_nodes.py --role client --hospital-id hospital_1 --server-address 127.0.0.1:8080

# Terminal 3 - Hospital 2:
python scripts/run_federated_nodes.py --role client --hospital-id hospital_2 --server-address 127.0.0.1:8080

# Terminal 4 - Hospital 3:
python scripts/run_federated_nodes.py --role client --hospital-id hospital_3 --server-address 127.0.0.1:8080
```

---

## Privacy Safeguards (HIPAA / GDPR)

- Raw MRI volumes (`.nii`, `.nii.gz`), DICOM series, and checkpoints are excluded in `.gitignore`.
- Loggers automatically redact absolute and relative filesystem paths, outputting only numerical metrics and anonymized subject identifiers.
- Hospital nodes only exchange aggregated model parameter tensors and summary validation metrics, never raw patient scans.
