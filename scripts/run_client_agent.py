"""Organization client runner contacting the aggregation server over mTLS."""

import os
import sys
import time
from pathlib import Path
import httpx
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from fedthreat.features.spec import FeatureSpec
from fedthreat.features.extractor import FeatureExtractor
from fedthreat.models.mlp import ThreatMLP
from fedthreat.client.trainer import ClientTrainer
from fedthreat.sim.partitioner import generate_synthetic_flow_dataset, partition_non_iid_dirichlet


def run_org_client():
    org_id = os.environ.get("ORG_ID", "1")
    server_url = os.environ.get("SERVER_URL", "https://server:8443")
    ssl_cert_file = os.environ.get("SSL_CERT_FILE", "/certs/client_1.crt")
    ssl_key_file = os.environ.get("SSL_KEY_FILE", "/certs/client_1.key")
    ssl_ca_file = os.environ.get("SSL_CA_FILE", "/certs/ca.crt")

    print(f"[+] Starting Organization Client Agent {org_id} connecting to {server_url}...")

    # Configure mTLS client with client certificate and private key
    ssl_context = httpx.create_ssl_context(verify=ssl_ca_file)
    ssl_context.load_cert_chain(certfile=ssl_cert_file, keyfile=ssl_key_file)

    client_http = httpx.Client(base_url=server_url, verify=ssl_context, timeout=30.0)

    # 1. Fetch current global model and spec hash
    print("[+] Requesting current global model from aggregation server...")
    res = client_http.get("/model/current")
    if res.status_code != 200:
        print(f"[!] Server returned status {res.status_code}: {res.text}")
        sys.exit(1)

    state = res.json()
    round_number = state["round_number"]
    spec_hash = state["spec_hash"]
    global_weights = {k: np.array(v, dtype=np.float32) for k, v in state["weights"].items()}

    # 2. Local Dataset & Training
    spec = FeatureSpec(numerical_features=["duration_ms", "src_bytes", "dst_bytes", "failed_auth_count"])
    extractor = FeatureExtractor(spec)
    events = generate_synthetic_flow_dataset(num_samples=2000, seed=int(org_id) * 100)
    X = extractor.extract_batch(events)
    y = np.array([0 if e.label == "benign" else 1 for e in events], dtype=np.int64)

    # Normalize
    mean = np.mean(X, axis=0, keepdims=True)
    std = np.std(X, axis=0, keepdims=True) + 1e-7
    X = (X - mean) / std

    ds = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y))
    loader = DataLoader(ds, batch_size=32, shuffle=True)

    model = ThreatMLP(input_dim=extractor.output_dim, hidden_dims=[128, 64, 32], num_classes=2)
    trainer = ClientTrainer(client_id=int(org_id), model=model, train_loader=loader, val_loader=loader)

    print(f"[+] Training locally for Round {round_number} (with L2 clip C=1.0)...")
    delta, num_samples, loss = trainer.train_round(
        global_params=global_weights,
        epochs=2,
        clip_norm=1.0,
        noise_multiplier=0.0,
    )

    # 3. Submit parameter delta
    print("[+] Submitting authenticated parameter update delta to server...")
    payload = {
        "client_id": str(org_id),
        "round_number": round_number,
        "num_samples": num_samples,
        "spec_hash": spec_hash,
        "delta": {k: v.tolist() for k, v in delta.items()},
    }
    sub_res = client_http.post("/round/submit", json=payload)
    print(f"[+] Server response: {sub_res.status_code} - {sub_res.json()}")


if __name__ == "__main__":
    run_org_client()
