"""Org container entrypoint: trains locally and participates in rounds over mutual TLS.

Env: ORG_ID, SERVER_URL, CERT_DIR (default /certs), NUM_ORGS, ROUNDS (default 5).
NOTE: uses a synthetic local dataset as a stand-in for the org's real log pipeline.
"""

import os

import numpy as np
from cryptography.hazmat.primitives import serialization

from fedthreat.client.agent import OrgAgent
from fedthreat.client.numpy_trainer import train_delta
from fedthreat.client.transport import mtls_client
from fedthreat.features.default import default_spec
from fedthreat.features.extractor import FeatureExtractor
from fedthreat.server.app import load_registry
from fedthreat.sim.partitioner import generate_synthetic_flow_dataset, partition_non_iid_dirichlet


def main() -> None:
    org = int(os.environ["ORG_ID"]); n = int(os.environ.get("NUM_ORGS", "9"))
    cdir = os.environ.get("CERT_DIR", "/certs")
    spec = default_spec()
    ext = FeatureExtractor(spec)
    parts = partition_non_iid_dirichlet(generate_synthetic_flow_dataset(6000, seed=7), n, alpha=1.0, seed=7, held_out_attack=None)
    ev = parts[org]
    X = ext.extract_batch(ev); y = np.array([0 if e.label == "benign" else 1 for e in ev])
    ident = serialization.load_pem_private_key(open(f"{cdir}/identity.key", "rb").read(), None)
    http = mtls_client(os.environ["SERVER_URL"], f"{cdir}/ca.crt", f"{cdir}/client.crt", f"{cdir}/client.key")
    agent = OrgAgent(org, ident, load_registry(f"{cdir}/enrolled_orgs.json"), http, spec.compute_spec_hash())
    for r in range(int(os.environ.get("ROUNDS", "5"))):
        res = agent.run_round(lambda p: train_delta(p, X, y, epochs=3, seed=org * 1000 + r))
        print(f"[org {org}] round {r}: {res.get('status')}", flush=True)
        if res.get("status") == "halted":
            break


if __name__ == "__main__":
    main()
