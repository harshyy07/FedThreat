"""Differential-privacy utilities: update clipping, noise, and an RDP accountant.

Granularity: this framework provides **organization-level** DP for the released global
model: one round's contribution of one org is its clipped update (L2 <= C), and the
Gaussian mechanism is applied to the SUM of cohort updates. It is NOT per-record DP-SGD.

Noise is added by clients *before* masking (distributed DP) so the server never sees an
un-noised individual update. With ``h`` honest clients per aggregate, each adds
N(0, (sigma*C)^2 / h) so the honest part of the sum carries N(0, (sigma*C)^2).
The guarantee only holds if at least ``h`` clients in every aggregate follow the protocol.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

Update = Dict[str, np.ndarray]

# --------------------------------------------------------------------------- clip & noise
def l2_norm(update: Update) -> float:
    return float(math.sqrt(sum(float(np.sum(np.square(v, dtype=np.float64))) for v in update.values())))


def clip_update(update: Update, clip_norm: float) -> Tuple[Update, float]:
    """Scales the whole update so that ||update||_2 <= clip_norm. Returns (clipped, original_norm)."""
    if clip_norm <= 0:
        raise ValueError("clip_norm must be > 0")
    norm = l2_norm(update)
    if not math.isfinite(norm):
        raise ValueError("update contains NaN/Inf")
    scale = min(1.0, clip_norm / (norm + 1e-12))
    return {k: (v * scale).astype(np.float32) for k, v in update.items()}, norm


def per_client_noise_std(noise_multiplier: float, clip_norm: float, honest_min: int = 1) -> float:
    """Std of the Gaussian each client adds so that >= honest_min honest clients sum to sigma*C."""
    if honest_min < 1:
        raise ValueError("honest_min must be >= 1")
    return float(noise_multiplier * clip_norm / math.sqrt(honest_min))


def add_gaussian_noise(update: Update, std: float, rng: Optional[np.random.Generator] = None) -> Update:
    """Adds N(0, std^2) noise. Default RNG is seeded from OS entropy (never a fixed seed).

    Caveat (documented limitation): float64 Gaussian sampling is subject to the floating-point
    attacks of Mironov (2012). Quantized secure aggregation rounds results to 2^-16, which
    removes most of that concern here, but a production system should use a discrete Gaussian.
    """
    if std <= 0:
        return {k: v.copy() for k, v in update.items()}
    rng = rng or np.random.default_rng()  # OS entropy
    return {k: (v + rng.normal(0.0, std, size=v.shape)).astype(np.float32) for k, v in update.items()}


# --------------------------------------------------------------------------- accounting
def _log_comb(n: int, k: int) -> float:
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def _logsumexp(xs: List[float]) -> float:
    m = max(xs)
    return m + math.log(sum(math.exp(x - m) for x in xs))


def rdp_gaussian(sigma: float, alpha: float) -> float:
    """Exact RDP of the Gaussian mechanism (sensitivity 1, noise std sigma): alpha / (2 sigma^2)."""
    return alpha / (2.0 * sigma * sigma)


def rdp_subsampled_gaussian(q: float, sigma: float, alpha: int) -> float:
    """RDP of the Poisson-subsampled Gaussian mechanism for INTEGER alpha >= 2
    (Mironov, Talwar, Zhang 2019, eq. for integer orders): log(A_alpha) / (alpha - 1)."""
    if q >= 1.0:
        return rdp_gaussian(sigma, alpha)
    terms = [
        _log_comb(alpha, k) + (alpha - k) * math.log1p(-q) + k * math.log(q) + (k * k - k) / (2.0 * sigma * sigma)
        for k in range(alpha + 1)
    ]
    return _logsumexp(terms) / (alpha - 1)


def rdp_to_epsilon(rdp: float, alpha: float, delta: float) -> float:
    """Improved RDP -> (eps, delta) conversion (Canonne-Kamath-Steinke 2020 / Balle et al. 2020)."""
    if alpha <= 1:
        raise ValueError("alpha must be > 1")
    return rdp + math.log1p(-1.0 / alpha) - (math.log(delta) + math.log(alpha)) / (alpha - 1)


DEFAULT_ORDERS: Tuple[int, ...] = tuple(list(range(2, 64)) + [64, 80, 96, 128, 192, 256, 512])


class PrivacyAccountant:
    """Cumulative (epsilon, delta) accountant for a sequence of (subsampled) Gaussian mechanisms."""

    def __init__(self, target_delta: float = 1e-5, max_epsilon: float = 10.0,
                 orders: Iterable[int] = DEFAULT_ORDERS):
        if not 0 < target_delta < 1:
            raise ValueError("target_delta must be in (0, 1)")
        self.target_delta = target_delta
        self.max_epsilon = max_epsilon
        self.orders = tuple(orders)
        self._history: Counter = Counter()  # (sigma, q) -> count

    # recording ---------------------------------------------------------------------
    @property
    def rounds_history(self) -> List[Tuple[float, float]]:
        return [key for key, n in sorted(self._history.items()) for _ in range(n)]

    def step(self, noise_multiplier: float, sample_rate: float = 1.0, count: int = 1) -> None:
        if noise_multiplier <= 0:
            raise ValueError("noise_multiplier must be > 0 (no noise => no DP guarantee)")
        if not 0 < sample_rate <= 1:
            raise ValueError("sample_rate must be in (0, 1]")
        self._history[(float(noise_multiplier), float(sample_rate))] += count

    # evaluation --------------------------------------------------------------------
    def _total_rdp(self, alpha: int, extra: Optional[Tuple[float, float]] = None) -> float:
        total = 0.0
        items = list(self._history.items()) + ([(extra, 1)] if extra else [])
        for (sigma, q), n in items:
            total += n * rdp_subsampled_gaussian(q, sigma, alpha)
        return total

    def _epsilon(self, extra: Optional[Tuple[float, float]] = None) -> float:
        if not self._history and extra is None:
            return 0.0
        return min(rdp_to_epsilon(self._total_rdp(a, extra), a, self.target_delta) for a in self.orders)

    def get_epsilon(self) -> float:
        return float(self._epsilon())

    def is_budget_exhausted(self) -> bool:
        return self.get_epsilon() >= self.max_epsilon

    def can_afford(self, noise_multiplier: float, sample_rate: float = 1.0) -> bool:
        """Pre-flight check: would one more step keep epsilon within budget?"""
        return self._epsilon((float(noise_multiplier), float(sample_rate))) <= self.max_epsilon

    def rounds_remaining(self, noise_multiplier: float, sample_rate: float = 1.0, cap: int = 100_000) -> int:
        n, probe = 0, PrivacyAccountant(self.target_delta, self.max_epsilon, self.orders)
        probe._history = Counter(self._history)
        while n < cap and probe.can_afford(noise_multiplier, sample_rate):
            probe.step(noise_multiplier, sample_rate)
            n += 1
        return n
