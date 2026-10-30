import pytest
import numpy as np

from fedthreat.server.validator import UpdateValidator, UpdateValidationError
from fedthreat.aggregation.cohort import CohortRobustAggregator
from fedthreat.aggregation.robust import CoordinateWiseMedianAggregator, TrimmedMeanAggregator


def test_validator_rejects_nan_inf():
    validator = UpdateValidator(max_l2_norm=10.0)

    # Valid
    validator.validate_update({"w": np.array([1.0, 2.0], dtype=np.float32)})

    # NaN
    with pytest.raises(UpdateValidationError, match="NaN or Inf"):
        validator.validate_update({"w": np.array([1.0, np.nan], dtype=np.float32)})

    # Inf
    with pytest.raises(UpdateValidationError, match="NaN or Inf"):
        validator.validate_update({"w": np.array([np.inf, 2.0], dtype=np.float32)})


def test_validator_rejects_oversized_norm():
    validator = UpdateValidator(max_l2_norm=5.0)

    # Norm is 10.0 > 5.0
    with pytest.raises(UpdateValidationError, match="exceeds server maximum"):
        validator.validate_update({"w": np.array([6.0, 8.0], dtype=np.float32)})


def test_validator_shape_and_spec_hash():
    validator = UpdateValidator(max_l2_norm=10.0, expected_spec_hash="expected_hash_123")

    validator.validate_spec_hash("expected_hash_123")
    with pytest.raises(UpdateValidationError, match="Feature spec mismatch"):
        validator.validate_spec_hash("wrong_hash")

    # Shape mismatch
    with pytest.raises(UpdateValidationError, match="shape mismatch"):
        validator.validate_update({"layer1": np.zeros((3, 3))}, expected_shapes={"layer1": (3, 4)})


def test_cohort_robust_aggregator_multi_cohort():
    """Validates that with 9 clients in 3 cohorts of 3,
    a Byzantine poisoned cohort is properly neutralized by coordinate-wise median!
    """
    median_agg = CoordinateWiseMedianAggregator()
    cohort_agg = CohortRobustAggregator(robust_aggregator=median_agg, cohort_size=3, min_cohorts=3, seed=42)

    global_p = {"w": np.array([0.0, 0.0], dtype=np.float32)}

    # 8 honest updates (+1.0), 1 Byzantine update (+100.0)
    updates = [
        {"w": np.array([1.0, 1.0], dtype=np.float32)},
        {"w": np.array([1.0, 1.0], dtype=np.float32)},
        {"w": np.array([1.0, 1.0], dtype=np.float32)},
        {"w": np.array([1.0, 1.0], dtype=np.float32)},
        {"w": np.array([1.0, 1.0], dtype=np.float32)},
        {"w": np.array([1.0, 1.0], dtype=np.float32)},
        {"w": np.array([1.0, 1.0], dtype=np.float32)},
        {"w": np.array([1.0, 1.0], dtype=np.float32)},
        {"w": np.array([100.0, 100.0], dtype=np.float32)},  # Byzantine client in 1 cohort
    ]
    weights = [1.0] * 9

    res = cohort_agg.aggregate(updates, weights, global_p)

    # Across the two cohorts, honest cohort avg is ~1.0, poisoned cohort avg is ~34.0.
    # The coordinate-wise median selects an honest value close to 1.0 rather than 100!
    assert res["w"][0] < 20.0
