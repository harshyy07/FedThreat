import pytest
import numpy as np

from fedthreat.aggregation.robust import (
    CoordinateWiseMedianAggregator,
    TrimmedMeanAggregator,
    KrumAggregator,
)
from fedthreat.attacks.poisoning import SignFlippingAttacker, ModelReplacementAttacker


def test_coordinate_wise_median_poison_immunity():
    agg = CoordinateWiseMedianAggregator()
    global_p = {"w": np.array([1.0, 1.0], dtype=np.float32)}

    # 4 honest updates around +1.0, 1 Byzantine update at +1000.0
    honest1 = {"w": np.array([0.9, 1.1], dtype=np.float32)}
    honest2 = {"w": np.array([1.0, 1.0], dtype=np.float32)}
    honest3 = {"w": np.array([1.1, 0.9], dtype=np.float32)}
    honest4 = {"w": np.array([1.0, 1.0], dtype=np.float32)}
    malicious = {"w": np.array([1000.0, 1000.0], dtype=np.float32)}

    updates = [honest1, honest2, honest3, honest4, malicious]
    weights = [1.0] * 5

    res = agg.aggregate(updates, weights, global_p)
    # Median along each coord should pick 1.0 -> global_p + 1.0 = [2.0, 2.0]
    np.testing.assert_allclose(res["w"], np.array([2.0, 2.0], dtype=np.float32))


def test_trimmed_mean():
    agg = TrimmedMeanAggregator(beta=0.2)  # trims 1 out of 5 from each end
    global_p = {"w": np.array([0.0], dtype=np.float32)}

    updates = [
        {"w": np.array([-100.0], dtype=np.float32)},  # outlier low
        {"w": np.array([2.0], dtype=np.float32)},
        {"w": np.array([3.0], dtype=np.float32)},
        {"w": np.array([4.0], dtype=np.float32)},
        {"w": np.array([500.0], dtype=np.float32)},   # outlier high
    ]
    weights = [1.0] * 5
    res = agg.aggregate(updates, weights, global_p)
    # Remaining after trim: 2.0, 3.0, 4.0 -> mean = 3.0
    np.testing.assert_allclose(res["w"], np.array([3.0], dtype=np.float32))


def test_krum_discards_outlier():
    agg = KrumAggregator(num_malicious=1, m=1)
    global_p = {"w": np.array([0.0, 0.0], dtype=np.float32)}

    # Honest cluster around [1.0, 1.0]
    honest1 = {"w": np.array([1.0, 1.0], dtype=np.float32)}
    honest2 = {"w": np.array([1.05, 0.95], dtype=np.float32)}
    honest3 = {"w": np.array([0.95, 1.05], dtype=np.float32)}
    honest4 = {"w": np.array([1.0, 1.0], dtype=np.float32)}
    # Byzantine outlier
    byzantine = {"w": np.array([50.0, -50.0], dtype=np.float32)}

    updates = [honest1, honest2, honest3, honest4, byzantine]
    weights = [1.0] * 5

    res = agg.aggregate(updates, weights, global_p)
    # Krum must select one of the honest updates, never the byzantine update
    assert np.all(res["w"] < 5.0)
