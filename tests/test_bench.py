import random

import numpy as np
import pytest

from fedthreat.aggregation.cohort import CohortRobustAggregator
from fedthreat.aggregation.robust import CoordinateWiseMedianAggregator
from fedthreat.bench import metrics as M
from fedthreat.bench.fl import RunConfig, run, server_aggregate
from fedthreat.bench.runner import expand, summarise


def test_average_precision_and_recall_at_fpr():
    y = np.array([1, 1, 0, 0, 0, 0]); s = np.array([.9, .8, .1, .2, .3, .05])
    assert M.average_precision(y, s) == pytest.approx(1.0)
    assert M.average_precision(y, -s) < 0.5
    assert M.recall_at_fpr(y, s, 0.0) == 1.0
    assert np.isnan(M.average_precision(np.zeros(3, int), np.ones(3)))


def test_fast_cohort_path_matches_real_secure_aggregation_protocol():
    rng = np.random.default_rng(0)
    ups = [{"w": rng.normal(0, 0.1, (2, 5)).astype(np.float32)} for _ in range(9)]
    cfg = RunConfig(aggregator="cohort_median", clip=10.0, n_orgs=9)
    fast = server_aggregate(cfg, ups, [1.0] * 9, dim=10, rng=random.Random(5))
    proto = CohortRobustAggregator(CoordinateWiseMedianAggregator(), 3, rng=random.Random(5)).aggregate(
        ups, [1.0] * 9, {"w": np.zeros((2, 5), np.float32)})
    np.testing.assert_allclose(fast["w"], proto["w"], atol=5e-4)


def test_grid_expansion_and_summary():
    spec = {"name": "t", "seeds": [0, 1], "base": {"rounds": 2}, "runs": [{"grid": {"alpha": [0.1, 1.0], "aggregator": ["fedavg", "median"]}}]}
    cfgs = expand(spec)
    assert len(cfgs) == 8 and all(c.rounds == 2 and c.experiment == "t" for c in cfgs)
    rows = [{**c.__dict__, "pr_auc": 0.9, "epsilon": float("inf")} for c in cfgs]
    s = summarise(rows)
    assert len(s) == 4 and all(r["n_seeds"] == 2 and r["epsilon_mean"] == float("inf") for r in s)


def test_smoke_run_all_modes_and_attacks():
    for kw in [dict(mode="central"), dict(mode="local"), dict(aggregator="cohort_median"),
               dict(aggregator="krum", attack="lie", num_malicious=2),
               dict(aggregator="secure_sum", sigma=1.0, honest_min=3, attack="backdoor", num_malicious=1)]:
        r = run(RunConfig(rounds=2, n_orgs=9, **kw))
        assert 0.0 <= r["pr_auc"] <= 1.0 and "asr" in r
    assert np.isfinite(run(RunConfig(rounds=2, aggregator="secure_sum", sigma=2.0, honest_min=3))["epsilon"])


def test_cohort_modes_require_clipping():
    with pytest.raises(ValueError):
        server_aggregate(RunConfig(aggregator="cohort_median", clip=0.0), [{"w": np.zeros(2, np.float32)}] * 9, [1.0] * 9, 2, random.Random(0))
