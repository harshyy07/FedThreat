"""Synthetic dataset generator and non-IID partitioner mimicking cross-organization threat data."""

from typing import Dict, List, Tuple
import numpy as np
from fedthreat.features.spec import CanonicalEvent


ATTACK_CLASSES = ["benign", "dos", "portscan", "bruteforce", "infiltration"]


def generate_synthetic_flow_dataset(
    num_samples: int = 10000,
    seed: int = 42,
    attack_ratio: float = 0.15,
) -> List[CanonicalEvent]:
    """Generates synthetic security flow events for testing and pipeline validation."""
    rng = np.random.RandomState(seed)
    events: List[CanonicalEvent] = []

    protocols = ["tcp", "udp", "icmp"]
    actions = ["allow", "drop", "block"]

    for _ in range(num_samples):
        is_attack = rng.rand() < attack_ratio
        if is_attack:
            attack_type = rng.choice(ATTACK_CLASSES[1:], p=[0.4, 0.3, 0.2, 0.1])
        else:
            attack_type = "benign"

        # Generate realistic statistical differences
        if attack_type == "dos":
            duration = rng.exponential(scale=20.0)
            src_bytes = float(rng.randint(1000, 50000))
            dst_bytes = float(rng.randint(100, 2000))
            pkts = float(rng.randint(200, 5000))
            dst_p = 80
            proto = "tcp"
            act = "drop" if rng.rand() < 0.4 else "allow"
            failed_auth = 0.0
        elif attack_type == "portscan":
            duration = rng.uniform(0.1, 5.0)
            src_bytes = float(rng.randint(40, 200))
            dst_bytes = 0.0
            pkts = float(rng.randint(1, 5))
            dst_p = int(rng.randint(1, 65535))
            proto = "tcp"
            act = "drop" if rng.rand() < 0.8 else "allow"
            failed_auth = 0.0
        elif attack_type == "bruteforce":
            duration = rng.uniform(10.0, 300.0)
            src_bytes = float(rng.randint(500, 3000))
            dst_bytes = float(rng.randint(500, 3000))
            pkts = float(rng.randint(20, 100))
            dst_p = int(rng.choice([22, 3389, 445]))
            proto = "tcp"
            act = "allow"
            failed_auth = float(rng.randint(5, 50))
        elif attack_type == "infiltration":
            duration = rng.uniform(500.0, 5000.0)
            src_bytes = float(rng.randint(5000, 100000))
            dst_bytes = float(rng.randint(10000, 500000))
            pkts = float(rng.randint(50, 500))
            dst_p = 443
            proto = "tcp"
            act = "allow"
            failed_auth = 0.0
        else:  # Benign
            duration = rng.exponential(scale=50.0)
            src_bytes = float(rng.randint(100, 10000))
            dst_bytes = float(rng.randint(500, 50000))
            pkts = float(rng.randint(5, 100))
            dst_p = int(rng.choice([80, 443, 53, 8080]))
            proto = str(rng.choice(protocols, p=[0.75, 0.20, 0.05]))
            act = "allow"
            failed_auth = 0.0 if rng.rand() < 0.98 else 1.0

        event = CanonicalEvent(
            src_port=int(rng.randint(1024, 65535)),
            dst_port=dst_p,
            protocol=proto,
            action=act,
            duration_ms=float(duration),
            src_bytes=src_bytes,
            dst_bytes=dst_bytes,
            total_packets=pkts,
            failed_auth_count=failed_auth,
            label=attack_type,
        )
        events.append(event)

    return events


def partition_non_iid_dirichlet(
    events: List[CanonicalEvent],
    num_orgs: int = 5,
    alpha: float = 0.5,
    seed: int = 42,
    held_out_attack: str | None = "infiltration",
    held_out_org: int = 0,
) -> Dict[int, List[CanonicalEvent]]:
    """Partitions events across organizations using a Dirichlet distribution for label skew,
    optionally holding out a rare attack family so only one organization experiences it.
    """
    rng = np.random.RandomState(seed)
    org_events: Dict[int, List[CanonicalEvent]] = {i: [] for i in range(num_orgs)}

    # Group by attack label
    label_to_events: Dict[str, List[CanonicalEvent]] = {}
    for ev in events:
        lbl = ev.label or "benign"
        if lbl not in label_to_events:
            label_to_events[lbl] = []
        label_to_events[lbl].append(ev)

    for lbl, ev_list in label_to_events.items():
        rng.shuffle(ev_list)
        n_ev = len(ev_list)

        # Handle held-out attack family: all given to held_out_org
        if held_out_attack and lbl == held_out_attack:
            org_events[held_out_org].extend(ev_list)
            continue

        # Sample Dirichlet proportions
        proportions = rng.dirichlet(np.repeat(alpha, num_orgs))
        proportions = proportions / proportions.sum()
        splits = (np.cumsum(proportions) * n_ev).astype(int)[:-1]

        subsets = np.split(ev_list, splits)
        for org_idx, subset in enumerate(subsets):
            org_events[org_idx].extend(subset.tolist())

    return org_events
