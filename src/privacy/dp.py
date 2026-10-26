"""Client-level Differential Privacy: L2 clipping, Gaussian noise, and Renyi DP accountant."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import numpy as np
from fedmed.utils.logging import get_logger

logger = get_logger(__name__)


def clip_update_norm(
    update_tensors: Sequence[np.ndarray], clip_norm: float
) -> Tuple[List[np.ndarray], float]:
    """Clip client update delta to maximum L2 norm C.

    Mathematical Operation:
        Delta_clipped = Delta * min(1, C / ||Delta||_2)

    Args:
        update_tensors: List of numpy arrays representing model weight deltas.
        clip_norm: Maximum allowed L2 norm (C).

    Returns:
        Tuple of (clipped_tensors, original_l2_norm).
    """
    if clip_norm <= 0:
        raise ValueError(f"clip_norm must be positive, got {clip_norm}")

    total_sumsq = sum(float(np.sum(np.square(t))) for t in update_tensors)
    total_norm = math.sqrt(total_sumsq)

    clip_factor = min(1.0, clip_norm / (total_norm + 1e-12))
    clipped = [t * clip_factor for t in update_tensors]

    return clipped, total_norm


def add_gaussian_noise(
    update_tensors: Sequence[np.ndarray],
    clip_norm: float,
    noise_multiplier: float,
    seed: Optional[int] = None,
) -> List[np.ndarray]:
    """Add spherical Gaussian noise scaled to the clipped L2 sensitivity C.

    Mathematical Operation:
        sigma_noise = noise_multiplier * C
        Delta_noisy = Delta_clipped + N(0, sigma_noise^2)

    Args:
        update_tensors: List of numpy arrays (already clipped to L2 norm C).
        clip_norm: L2 sensitivity bound (C).
        noise_multiplier: Multiplier sigma controlling privacy strength.
        seed: Optional RNG seed for reproducibility.

    Returns:
        List of noisy update tensors.
    """
    if noise_multiplier <= 0:
        return [t.copy() for t in update_tensors]

    rng = np.random.default_rng(seed)
    std = float(noise_multiplier * clip_norm)

    noisy_tensors = []
    for t in update_tensors:
        noise = rng.normal(loc=0.0, scale=std, size=t.shape).astype(t.dtype)
        noisy_tensors.append(t + noise)

    return noisy_tensors


class RenyiDPAccountant:
    """Analytical Renyi Differential Privacy (RDP) accountant for client-level Gaussian mechanism.

    Mathematical Formulation:
    For a Gaussian mechanism with sensitivity C and noise std sigma * C:
      - Per-round RDP at order alpha > 1:
          eps_rdp(alpha) = alpha / (2 * sigma^2)
      - Under linear composition after T rounds with full client participation (q = 1):
          eps_total(alpha) = T * alpha / (2 * sigma^2)
      - Conversion from RDP to (epsilon, delta)-DP for a target delta:
          eps(delta) = min_{alpha > 1} [ eps_total(alpha) + ln(1/delta) / (alpha - 1) ]

    Simplification and Assumptions:
      - This accounts for client-level (hospital-level) differential privacy, where each hospital's
        entire update is treated as one private contribution.
      - Assumes participation rate q = 1.0 (all silos participate each round).
    """

    def __init__(self, noise_multiplier: float, delta: float = 1e-5, orders: Optional[Sequence[float]] = None) -> None:
        if noise_multiplier <= 0:
            raise ValueError("noise_multiplier must be > 0 for differential privacy accounting")
        self.noise_multiplier = float(noise_multiplier)
        self.delta = float(delta)
        if orders is None:
            # Range of Renyi orders covering small to large compositions
            self.orders = [1.1 + 0.1 * i for i in range(50)] + [float(x) for x in range(7, 100)]
        else:
            self.orders = [float(x) for x in orders]

    def compute_epsilon(self, rounds: int) -> float:
        """Compute cumulative privacy loss epsilon at the configured delta after T rounds.

        Args:
            rounds: Number of federated training rounds completed.

        Returns:
            Epsilon privacy loss value.
        """
        if rounds <= 0 or self.noise_multiplier <= 0:
            return 0.0

        sigma = self.noise_multiplier
        log_one_over_delta = math.log(1.0 / self.delta)

        eps_candidates = []
        for alpha in self.orders:
            if alpha <= 1.0:
                continue
            # RDP of Gaussian mechanism for T rounds
            rdp = rounds * alpha / (2.0 * (sigma ** 2))
            # Conversion to standard (eps, delta)-DP
            eps = rdp + log_one_over_delta / (alpha - 1.0)
            eps_candidates.append(eps)

        return float(min(eps_candidates))


__all__ = ["clip_update_norm", "add_gaussian_noise", "RenyiDPAccountant"]
