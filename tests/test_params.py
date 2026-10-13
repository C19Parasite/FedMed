"""Unit tests for parameter serialization, vectorization, and inspection."""

import logging
import numpy as np
import pytest
import torch

from fedmed.models.factory import build_model
from fedmed.models.params import (
    count_parameters,
    get_parameters,
    log_model_summary,
    model_size_mb,
    parameters_to_vector,
    set_parameters,
    vector_to_parameters,
)
from fedmed.utils.seed import seed_everything


@pytest.fixture
def smoke_model_cfg():
    return {
        "model": {
            "name": "unet3d_custom",
            "in_channels": 4,
            "out_channels": 3,
            "channels": [4, 8],
            "strides": [2],
            "norm": {"type": "instance"},
            "init": {"name": "kaiming"},
        }
    }


def test_get_and_set_parameters_round_trip(smoke_model_cfg):
    """Verify get_parameters then set_parameters on a fresh model gives identical outputs."""
    seed_everything(123)
    model1 = build_model(smoke_model_cfg)

    seed_everything(999)
    model2 = build_model(smoke_model_cfg)

    # Initial check: different seeds give different outputs
    x = torch.randn(1, 4, 16, 16, 16)
    with torch.no_grad():
        out1_before = model1(x)
        out2_before = model2(x)
    assert not torch.allclose(out1_before, out2_before)

    # Extract parameters from model1 and load into model2
    params1 = get_parameters(model1)
    set_parameters(model2, params1)

    # Outputs must now match exactly
    with torch.no_grad():
        out1_after = model1(x)
        out2_after = model2(x)
    assert torch.allclose(out1_after, out2_after, atol=1e-5)


def test_set_parameters_validation_errors(smoke_model_cfg):
    """Verify set_parameters raises on length and shape mismatches."""
    model = build_model(smoke_model_cfg)
    params = get_parameters(model)

    # Wrong length
    with pytest.raises(ValueError, match="Parameter count mismatch"):
        set_parameters(model, params[:-1])

    # Wrong shape
    corrupted_params = [p.copy() for p in params]
    corrupted_params[0] = np.zeros((99, 99), dtype=np.float32)
    with pytest.raises(ValueError, match="Shape mismatch"):
        set_parameters(model, corrupted_params)


def test_vector_round_trip_lossless(smoke_model_cfg):
    """Verify parameters_to_vector and vector_to_parameters round-trip is lossless and stable."""
    seed_everything(42)
    model = build_model(smoke_model_cfg)

    vec1 = parameters_to_vector(list(model.parameters()))
    vec2 = parameters_to_vector(list(model.parameters()))

    # Ordering stability
    assert torch.equal(vec1, vec2)

    # Perturb model weights
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()

    # Re-apply vector
    vector_to_parameters(vec1, model)
    vec_restored = parameters_to_vector(list(model.parameters()))

    assert torch.equal(vec1, vec_restored)


def test_count_parameters_and_size_mb(smoke_model_cfg):
    """Verify count_parameters equals manual sum and model_size_mb is positive."""
    model = build_model(smoke_model_cfg)

    trainable_count = count_parameters(model, trainable_only=True)
    all_count = count_parameters(model, trainable_only=False)

    manual_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    manual_all = sum(p.numel() for p in model.parameters())

    assert trainable_count == manual_trainable
    assert all_count == manual_all
    assert trainable_count > 0

    size_mb = model_size_mb(model)
    assert size_mb > 0.0

    # Summary logger
    logger = logging.getLogger("test_logger")
    summary = log_model_summary(model, logger)
    assert summary["total_parameters"] == all_count
    assert summary["size_mb"] == round(size_mb, 4)
    assert "InstanceNorm" in summary["norm_type"]
