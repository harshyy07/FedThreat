"""Cohort-based robust secure aggregation.

Secure aggregation hides individual updates, but Byzantine-robust statistics (median, trimmed
mean, Krum) need several vectors to compare. The compromise (Option B in the design doc):

  1. the server randomly partitions clients into cohorts of size >= ``min_cohort_size``;
  2. each cohort runs secure aggregation, so the server learns only per-cohort sums;
  3. a robust aggregator is applied ACROSS the per-cohort means.

Guarantee: one malicious org can contaminate at most ONE cohort. A coordinate-wise median over
``m`` cohorts therefore tolerates ``f < m/2`` malicious orgs, which needs ``m >= 2f + 1`` cohorts,
i.e. at least ``min_cohort_size * (2f + 1)`` participating clients. With fewer than 3 cohorts there
is NO meaningful robustness and this module refuses to pretend otherwise.

Trade-off: smaller cohorts give the server a smaller anonymity set (a cohort of 3 hides one org
among 3), larger cohorts give less robustness. Weighting by ``num_samples`` is deliberately not
used: it is self-reported and unverifiable, so each org counts equally.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from fedthreat.base import Aggregator
from fedthreat.secagg.protocol import simulate_secure_sum
from fedthreat.secagg.quantize import Quantizer

Update = Dict[str, np.ndarray]


class InsufficientCohortsError(RuntimeError):
    """Not enough clients to form the number of cohorts required for robust aggregation."""


def assign_cohorts(
    client_ids: Sequence[int], min_size: int = 3, rng: Optional[random.Random] = None
) -> List[List[int]]:
    """Uniformly random partition into floor(n / min_size) cohorts whose sizes differ by <= 1.

    Uses ``random.SystemRandom`` by default so an adversary cannot predict or steer assignment.
    """
    ids = list(client_ids)
    if min_size < 2:
        raise ValueError("min_size must be >= 2")
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate client ids")
    if len(ids) < min_size:
        raise InsufficientCohortsError(f"{len(ids)} clients < min cohort size {min_size}")
    (rng or random.SystemRandom()).shuffle(ids)
    m = len(ids) // min_size
    cohorts: List[List[int]] = [[] for _ in range(m)]
    for i, cid in enumerate(ids):
        cohorts[i % m].append(cid)
    return [sorted(c) for c in cohorts]


def combine_cohort_means(
    cohort_means: List[Update], robust: Aggregator, global_params: Dict[str, np.ndarray]
) -> Dict[str, np.ndarray]:
    return robust.aggregate(cohort_means, [1.0] * len(cohort_means), global_params)


class CohortRobustAggregator(Aggregator):
    """SIMULATION aggregator: runs the real secure-aggregation protocol code in-process.

    The caller already holds every plaintext update, so this is NOT a security boundary. It lets
    experiments measure accuracy/robustness/overhead of the exact cohort pipeline the server uses
    (see ``fedthreat.server.coordinator`` for the networked, trust-separated version).
    """

    def __init__(
        self,
        robust_aggregator: Aggregator,
        cohort_size: int = 3,
        min_cohorts: int = 3,
        allow_degraded: bool = False,
        rng: Optional[random.Random] = None,
        seed: Optional[int] = None,
    ):
        self.robust_aggregator = robust_aggregator
        self.cohort_size = max(2, cohort_size)
        self.min_cohorts = min_cohorts
        self.allow_degraded = allow_degraded
        self.rng = rng if rng is not None else (random.Random(seed) if seed is not None else None)
        self.last_report: Dict[str, Any] = {}

    def aggregate(
        self,
        client_updates: List[Update],
        client_weights: List[float],
        global_params: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        n = len(client_updates)
        if n == 0:
            return global_params
        cohorts = assign_cohorts(range(n), self.cohort_size, self.rng)
        robust = len(cohorts) >= self.min_cohorts
        if not robust and not self.allow_degraded:
            raise InsufficientCohortsError(
                f"{n} clients form {len(cohorts)} cohort(s); need >= {self.min_cohorts} cohorts "
                f"(>= {self.min_cohorts * self.cohort_size} clients) for Byzantine robustness"
            )

        q = Quantizer(max_terms=max(len(c) for c in cohorts) + 1)
        lim = q.max_abs_value
        means: List[Update] = []
        for cohort in cohorts:
            # An attacker's best move against a bounded encoder is to saturate; model that.
            ups = {i: {k: np.clip(v, -lim, lim) for k, v in client_updates[i].items()} for i in cohort}
            total, survivors = simulate_secure_sum(ups, q)
            means.append({k: (v / len(survivors)).astype(np.float32) for k, v in total.items()})

        self.last_report = {"cohorts": cohorts, "robust": robust}
        if not robust:  # explicit, loudly-flagged degraded mode: plain mean of cohort means
            avg = {k: np.mean([m[k] for m in means], axis=0).astype(np.float32) for k in global_params}
            return {k: global_params[k] + avg[k] for k in global_params}
        return combine_cohort_means(means, self.robust_aggregator, global_params)
