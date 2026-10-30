from pathlib import Path
import pytest
import numpy as np

from fedthreat.features.spec import CanonicalEvent, FeatureSpec
from fedthreat.features.extractor import FeatureExtractor
from fedthreat.features.hashing import simple_hash
from fedthreat.sim.partitioner import generate_synthetic_flow_dataset, partition_non_iid_dirichlet
from fedthreat.metrics import compute_threat_metrics


def test_feature_spec_hash_determinism():
    spec1 = FeatureSpec(numerical_features=["duration_ms", "src_bytes"])
    spec2 = FeatureSpec(numerical_features=["duration_ms", "src_bytes"])
    assert spec1.compute_spec_hash() == spec2.compute_spec_hash()


def test_feature_hashing():
    vec = simple_hash("443", num_bins=16)
    assert len(vec) == 16
    assert np.sum(vec) == 1.0


def test_feature_extractor_output_dim():
    spec = FeatureSpec(
        numerical_features=["duration_ms", "src_bytes"],
        hashed_features=[{"name": "dst_port_hash", "source_field": "dst_port", "num_bins": 8}],
        include_mask_vector=True,
    )
    extractor = FeatureExtractor(spec)
    # 2 numerical + 8 hash + 6 cat = 16. With mask = 32.
    assert extractor.output_dim == 32

    ev = CanonicalEvent(duration_ms=10.0, src_bytes=500.0, dst_port=80, protocol="tcp", action="allow")
    vec, _ = extractor.extract_from_event(ev)
    assert len(vec) == 32


def test_dirichlet_partitioner_held_out():
    events = generate_synthetic_flow_dataset(num_samples=200, seed=123)
    partitions = partition_non_iid_dirichlet(
        events, num_orgs=3, alpha=0.5, held_out_attack="infiltration", held_out_org=0
    )
    assert len(partitions) == 3
    # Check that infiltration only exists in org 0
    for org_id, org_events in partitions.items():
        infil_count = sum(1 for e in org_events if e.label == "infiltration")
        if org_id == 0:
            assert infil_count >= 0
        else:
            assert infil_count == 0


def test_metrics_computation():
    y_true = np.array([0, 0, 0, 1, 1])
    y_scores = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    y_pred = np.array([0, 0, 0, 1, 1])
    res = compute_threat_metrics(y_true, y_scores, y_pred)
    assert res["macro_f1"] == 1.0
    assert res["pr_auc"] == 1.0
