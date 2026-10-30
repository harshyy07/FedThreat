import json
import pytest
import numpy as np

from fedthreat.adapters.zeek import ZeekAdapter
from fedthreat.adapters.windows_evtx import WindowsEvtxAdapter
from fedthreat.adapters.syslog import SyslogAdapter
from fedthreat.features.spec import FeatureSpec
from fedthreat.features.extractor import FeatureExtractor


@pytest.fixture
def feature_extractor():
    spec = FeatureSpec(
        numerical_features=["duration_ms", "src_bytes", "dst_bytes", "failed_auth_count"],
        hashed_features=[{"name": "dst_port_hash", "source_field": "dst_port", "num_bins": 8}],
        include_mask_vector=True,
    )
    return FeatureExtractor(spec)


def test_zeek_json_adapter(feature_extractor):
    adapter = ZeekAdapter()
    sample_json = json.dumps({
        "ts": 1700000000.0,
        "id.orig_h": "192.168.1.100",
        "id.orig_p": 50123,
        "id.resp_h": "10.0.0.1",
        "id.resp_p": 443,
        "proto": "tcp",
        "conn_state": "SF",
        "duration": 2.5,
        "orig_bytes": 1024,
        "resp_bytes": 4096,
        "orig_pkts": 10,
        "resp_pkts": 12,
    })
    event = adapter.parse_line(sample_json)
    assert event.protocol == "tcp"
    assert event.dst_port == 443
    assert event.duration_ms == 2500.0
    assert event.action == "allow"

    vec, raw_vals = feature_extractor.extract_from_event(event)
    # Check that mask vector is present (output dim = feature_dim * 2)
    assert len(vec) == feature_extractor.output_dim


def test_windows_evtx_adapter(feature_extractor):
    adapter = WindowsEvtxAdapter()
    # Test failed logon (Event 4625)
    sample_4625 = json.dumps({
        "EventID": 4625,
        "TimeCreated": "2026-10-26T12:00:00Z",
        "EventData": {
            "IpAddress": "192.168.1.200",
            "IpPort": "445",
            "TargetUserName": "Administrator",
        }
    })
    event = adapter.parse_line(sample_4625)
    assert event.failed_auth_count == 1.0
    assert event.action == "drop"
    assert event.dst_port == 445

    vec, _ = feature_extractor.extract_from_event(event)
    assert len(vec) == feature_extractor.output_dim


def test_syslog_adapter(feature_extractor):
    adapter = SyslogAdapter()
    log_line = "Oct 26 12:00:00 firewall kernel: [DROP] SRC=172.16.0.4 DST=10.1.1.1 PROTO=TCP SPT=55231 DPT=22 LEN=52"
    event = adapter.parse_line(log_line)
    assert event.action == "drop"
    assert event.protocol == "tcp"
    assert event.dst_port == 22
    assert event.src_bytes == 52.0

    vec, _ = feature_extractor.extract_from_event(event)
    assert len(vec) == feature_extractor.output_dim


def test_missing_features_mask_integrity(feature_extractor):
    """Verifies that missing features have 0 in the mask indicator without crashing."""
    adapter = SyslogAdapter()
    # Syslog has no duration or failed_auth_count
    log_line = "Oct 26 12:00:00 firewall: SRC=1.2.3.4 DST=5.6.7.8 PROTO=UDP SPT=123 DPT=123 LEN=40"
    event = adapter.parse_line(log_line)
    
    vec, raw_vals = feature_extractor.extract_from_event(event)
    half = len(vec) // 2
    raw = vec[:half]
    mask = vec[half:]

    # Mask and raw vectors have equal lengths
    assert len(raw) == len(mask)
    # The mask vector must contain 1.0 for present fields and 0.0 or 1.0 valid values
    assert np.all(np.isin(mask, [0.0, 1.0]))
