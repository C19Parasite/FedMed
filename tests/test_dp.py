"""Unit tests for Differential Privacy components (clipping, noise, RDP accountant)."""

from __future__ import annotations

import math
import numpy as np
import pytest

from src.privacy.dp import RenyiDPAccountant, add_gaussian_noise, clip_update_norm


def test_clipping_bounds_norm_to_c():
    """Verify that L2 clipping strictly bounds tensor norm to C without altering direction."""
    t1 = np.ones((10, 10), dtype=np.float32)  # norm = sqrt(100) = 10.0
    t2 = np.ones((5,), dtype=np.float32)     # norm = sqrt(5) ~ 2.236
    # Total norm = sqrt(100 + 5) = sqrt(105) ~ 10.247

    c = 2.0
    clipped, original_norm = clip_update_norm([t1, t2], clip_norm=c)

    clipped_norm = math.sqrt(float(np.sum(np.square(clipped[0])) + np.sum(np.square(clipped[1]))))
    assert math.isclose(clipped_norm, c, rel_tol=1e-5)
    assert original_norm > c

    # If norm is already below C, it should not be clipped
    c_large = 20.0
    unmodified, norm2 = clip_update_norm([t1, t2], clip_norm=c_large)
    unmod_norm = math.sqrt(float(np.sum(np.square(unmodified[0])) + np.sum(np.square(unmodified[1]))))
    assert math.isclose(unmod_norm, original_norm, rel_tol=1e-5)


def test_gaussian_noise_standard_deviation():
    """Verify that added Gaussian noise statistically matches sigma * C."""
    c = 1.5
    sigma = 2.0
    expected_std = sigma * c  # 3.0

    # Large zero tensor to evaluate noise distribution
    zero_t = [np.zeros((100, 100), dtype=np.float32)]
    noisy = add_gaussian_noise(zero_t, clip_norm=c, noise_multiplier=sigma, seed=42)

    sample_std = float(np.std(noisy[0]))
    sample_mean = float(np.mean(noisy[0]))

    assert math.isclose(sample_mean, 0.0, abs_tol=0.1)
    assert math.isclose(sample_std, expected_std, rel_tol=0.05)


def test_rdp_accountant_monotonicity():
    """Verify that epsilon increases monotonically with rounds and decreases as sigma increases."""
    sigma = 1.0
    accountant = RenyiDPAccountant(noise_multiplier=sigma, delta=1e-5)

    eps_round1 = accountant.compute_epsilon(rounds=1)
    eps_round2 = accountant.compute_epsilon(rounds=2)
    eps_round5 = accountant.compute_epsilon(rounds=5)

    assert 0 < eps_round1 < eps_round2 < eps_round5

    # Higher noise multiplier must yield strictly smaller epsilon (stronger privacy)
    stronger_accountant = RenyiDPAccountant(noise_multiplier=2.0, delta=1e-5)
    eps_stronger_round5 = stronger_accountant.compute_epsilon(rounds=5)

    assert eps_stronger_round5 < eps_round5
