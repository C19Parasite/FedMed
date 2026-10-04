# FedMed

**FedMed** is a cross-silo federated learning system for 3D brain tumor segmentation on multi-parametric MRI scans (T1, T1ce, T2, FLAIR) built using **PyTorch**, **MONAI**, and **Flower (`flwr`)**.

---

## Repository Structure

```text
FedMed/
├── configs/            # Experiment YAML configurations
├── data/               # Local/cached/synthetic data directories (gitignored)
├── fedmed/
│   ├── data/           # Datasets, transforms, loaders, splits, synthetic generator
│   ├── models/         # 3D U-Net network architecture
│   ├── training/       # Centralized trainer, Dice loss, segmentation metrics
│   ├── fl/             # Flower client, server, aggregation strategy
│   └── utils/          # Reproducible seeding, privacy logger, config loader
├── scripts/            # CLI utilities (download, train, eval, visualize, fl nodes)
├── tests/              # Pytest unit and integration test suite
├── requirements.txt    # Pinned core dependencies
├── pyproject.toml      # Package definition for editable installation
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

## Privacy Safeguards (HIPAA / GDPR)

- Raw MRI volumes (`.nii`, `.nii.gz`), DICOM series, and checkpoints are excluded in `.gitignore`.
- Loggers automatically redact absolute and relative filesystem paths, outputting only numerical metrics and anonymized subject identifiers.
