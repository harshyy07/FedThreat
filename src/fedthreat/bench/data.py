"""Benchmark data: non-IID organisations built from the synthetic flow generator.

NOTE: the generator is synthetic. Numbers from it validate the *machinery* (privacy/robustness/
heterogeneity trade-offs), not real-world detection quality. To use CIC-IDS2017 etc., write an
adapter producing the same ``OrgData`` structure (see ``build_orgs``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from fedthreat.features.default import default_spec
from fedthreat.features.extractor import FeatureExtractor
from fedthreat.sim.partitioner import generate_synthetic_flow_dataset, partition_non_iid_dirichlet

HELD_OUT = "infiltration"
TRIGGER_COL = 3  # raw feature column used as a backdoor trigger (arbitrary, documented)
TRIGGER_VAL = 1.0e4


@dataclass
class OrgData:
    X: np.ndarray
    y: np.ndarray           # binary: 1 = attack
    fam: np.ndarray         # attack family string ("benign" for benign)


@dataclass
class Benchmark:
    orgs: Dict[int, OrgData]
    test: OrgData
    held_out_org: int
    dim: int


def _to_org(events, ext) -> OrgData:
    fam = np.array([e.label or "benign" for e in events])
    return OrgData(ext.extract_batch(events), (fam != "benign").astype(np.int64), fam)


def build_orgs(n_orgs: int, alpha: float, seed: int, n_events: int = 12000, n_test: int = 20000,
               held_out_org: int = 0) -> Benchmark:
    ext = FeatureExtractor(default_spec())
    train = generate_synthetic_flow_dataset(n_events, seed=seed)
    parts = partition_non_iid_dirichlet(train, n_orgs, alpha=alpha, seed=seed, held_out_attack=HELD_OUT,
                                        held_out_org=held_out_org)
    orgs = {i: _to_org(p, ext) for i, p in parts.items() if len(p) > 0}
    test = _to_org(generate_synthetic_flow_dataset(n_test, seed=seed + 10_000), ext)
    return Benchmark(orgs, test, held_out_org, ext.output_dim)
