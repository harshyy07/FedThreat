"""The v1 feature spec shared by server and clients (mirrors configs/feature_spec_v1.yaml)."""

from fedthreat.features.spec import FeatureSpec


def default_spec() -> FeatureSpec:
    return FeatureSpec(
        numerical_features=["duration_ms", "src_bytes", "dst_bytes", "total_packets",
                            "bytes_per_second", "packets_per_second", "failed_auth_count"],
        categorical_features=["protocol", "action", "service"],
        hashed_features=[{"name": "dst_port_hash", "source_field": "dst_port", "num_bins": 16},
                         {"name": "src_port_hash", "source_field": "src_port", "num_bins": 16}],
    )
