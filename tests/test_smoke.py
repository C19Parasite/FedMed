"""Smoke test: package import and deterministic set_seed verification."""

import random
from pathlib import Path
import numpy as np
import torch

import fedmed
from fedmed.utils.logging import get_logger, redact_paths
from fedmed.utils.seed import seed_everything, set_seed


def test_package_import_and_version() -> None:
    """Verify fedmed package is importable and exposes version."""
    assert hasattr(fedmed, "__version__")
    assert isinstance(fedmed.__version__, str)


def test_set_seed_deterministic_torch_and_numpy() -> None:
    """Verify that calling set_seed produces identical pseudo-random outputs."""
    seed = 42

    # First run
    set_seed(seed)
    r1 = random.random()
    np1 = np.random.rand(5)
    t1 = torch.rand(3, 3)

    # Second run with same seed
    set_seed(seed)
    r2 = random.random()
    np2 = np.random.rand(5)
    t2 = torch.rand(3, 3)

    assert r1 == r2
    assert np.allclose(np1, np2)
    assert torch.allclose(t1, t2)

    # Alias check
    seed_everything(seed)
    r3 = random.random()
    assert r1 == r3


def test_privacy_logger_redacts_paths(tmp_path: Path) -> None:
    """Verify that file paths are redacted from log messages and file logging works."""
    test_log_file = tmp_path / "test.log"
    logger = get_logger("test_privacy", log_file=test_log_file)

    secret_path = "D:/FedMed/data/patient_001/BraTS20_001_flair.nii.gz"
    redacted = redact_paths(secret_path)
    assert "<redacted>" in redacted
    assert "BraTS20_001_flair" not in redacted

    logger.info("Processing scan at %s", secret_path)
    assert test_log_file.exists()
    content = test_log_file.read_text(encoding="utf-8")
    assert "BraTS20_001_flair" not in content
    assert "<redacted>" in content
