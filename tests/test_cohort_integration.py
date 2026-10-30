import random

import numpy as np
import pytest

from fedthreat.aggregation.cohort import (
    CohortRobustAggregator, InsufficientCohortsError, assign_cohorts,
)
from fedthreat.aggregation.robust import CoordinateWiseMedianAggregator, TrimmedMeanAggregator


def _ups(n, bad=(), honest=1.0, boost=1000.0, d=4):
    return [{"w": np.full(d, boost if i in bad else honest, np.float32)} for i in range(n)]


def test_assignment_is_random_balanced_and_covers_everyone():
    seen = set()
    for s in range(20):
        c = assign_cohorts(range(11), 3, random.Random(s))
        assert sorted(x for g in c for x in g) == list(range(11))
        assert len(c) == 3 and all(len(g) >= 3 for g in c)
        seen.add(tuple(map(tuple, c)))
    assert len(seen) > 5  # not a fixed consecutive split (regression)


def test_too_few_clients_for_robustness_is_refused():
    """Regression: server default (3 clients, 1 cohort) silently gave zero robustness."""
    agg = CohortRobustAggregator(TrimmedMeanAggregator(0.2), cohort_size=3)
    with pytest.raises(InsufficientCohortsError):
        agg.aggregate(_ups(3, bad={2}), [1.0] * 3, {"w": np.zeros(4, np.float32)})


def test_degraded_mode_is_explicit_and_not_robust():
    agg = CohortRobustAggregator(TrimmedMeanAggregator(0.2), cohort_size=3, allow_degraded=True)
    out = agg.aggregate(_ups(3, bad={2}), [1.0] * 3, {"w": np.zeros(4, np.float32)})
    assert agg.last_report["robust"] is False
    assert out["w"][0] > 100  # attacker dominates, and the report says so


@pytest.mark.parametrize("seed", range(5))
def test_single_attacker_neutralised_with_three_cohorts(seed):
    agg = CohortRobustAggregator(CoordinateWiseMedianAggregator(), cohort_size=3, rng=random.Random(seed))
    out = agg.aggregate(_ups(9, bad={4}), [1.0] * 9, {"w": np.zeros(4, np.float32)})
    assert agg.last_report["robust"] is True
    # median of 3 cohort means: two honest (=1.0) and one contaminated
    np.testing.assert_allclose(out["w"], 1.0, atol=1e-3)


def test_trimmed_mean_with_min_trim_also_neutralises():
    agg = CohortRobustAggregator(TrimmedMeanAggregator(beta=0.1, min_trim=1), cohort_size=3, rng=random.Random(1))
    out = agg.aggregate(_ups(9, bad={0}), [1.0] * 9, {"w": np.zeros(4, np.float32)})
    np.testing.assert_allclose(out["w"], 1.0, atol=1e-3)


def test_two_attackers_in_three_cohorts_can_break_median():
    """Documented limit: f must be < number_of_cohorts/2 *after* assignment. Find a seed that
    puts the two attackers in different cohorts => 2 of 3 cohorts contaminated => median breaks."""
    for seed in range(50):
        c = assign_cohorts(range(9), 3, random.Random(seed))
        if any({0, 1} <= set(g) for g in c):
            continue
        agg = CohortRobustAggregator(CoordinateWiseMedianAggregator(), 3, rng=random.Random(seed))
        out = agg.aggregate(_ups(9, bad={0, 1}), [1.0] * 9, {"w": np.zeros(4, np.float32)})
        assert out["w"][0] > 100
        return
    pytest.fail("no separating seed found")
