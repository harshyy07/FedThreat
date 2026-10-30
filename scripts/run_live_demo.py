"""Live demo of the REAL stack: TLS with REQUIRED client certs, signed requests, secure aggregation
across cohorts, norm-bounded cohort sums, and a median over cohorts. 9 orgs, 1 of them malicious.

Run:  python scripts/run_live_demo.py          (no Docker needed)
"""

import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import httpx
import numpy as np
from cryptography.hazmat.primitives import serialization

sys.path.insert(0, str(Path(__file__).parent))
from generate_certs import generate  # noqa: E402

from fedthreat.client.agent import OrgAgent
from fedthreat.client.transport import mtls_client
from fedthreat.client.numpy_trainer import init_linear_params, predict_proba, train_delta
from fedthreat.features.default import default_spec
from fedthreat.features.extractor import FeatureExtractor
from fedthreat.server.app import load_registry
from fedthreat.sim.partitioner import generate_synthetic_flow_dataset, partition_non_iid_dirichlet

N, EVIL, ROUNDS = 9, 4, 4


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def accuracy(params, X, y):
    return float((predict_proba(params, X).argmax(1) == y).mean())


def main(defended: bool = True) -> dict:
    tmp = Path(tempfile.mkdtemp(prefix="fedthreat_demo_"))
    generate(tmp, N)
    spec = default_spec(); ext = FeatureExtractor(spec)
    np.savez(tmp / "server" / "initial_params.npz", **init_linear_params(ext.output_dim))
    events = generate_synthetic_flow_dataset(9000, seed=7)
    parts = partition_non_iid_dirichlet(events, N, alpha=1.0, seed=7, held_out_attack=None)
    data = {i: (ext.extract_batch(p), np.array([0 if e.label == "benign" else 1 for e in p])) for i, p in parts.items()}
    Xall = np.vstack([data[i][0] for i in data]); yall = np.concatenate([data[i][1] for i in data])

    port = free_port()
    env = dict(os.environ, FEDTHREAT_SPEC_HASH=spec.compute_spec_hash(), FEDTHREAT_REGISTRY=str(tmp / "server/enrolled_orgs.json"),
               FEDTHREAT_INITIAL_PARAMS=str(tmp / "server/initial_params.npz"), FEDTHREAT_MIN_COHORTS="3" if defended else "1",
               FEDTHREAT_MIN_COHORT_SIZE="3" if defended else str(N),
               FEDTHREAT_ROBUST="median" if defended else "trimmed_mean", FEDTHREAT_MAX_EPSILON="10")
    srv = subprocess.Popen([sys.executable, "-m", "uvicorn", "fedthreat.server.app:app", "--port", str(port), "--log-level", "warning",
                            "--ssl-keyfile", str(tmp / "server/server.key"), "--ssl-certfile", str(tmp / "server/server.crt"),
                            "--ssl-ca-certs", str(tmp / "server/ca.crt"), "--ssl-cert-reqs", "2"], env=env)
    base = f"https://localhost:{port}"
    try:
        for _ in range(100):
            try:
                mtls_client(base, str(tmp / "ca.crt"), str(tmp / "org0/client.crt"), str(tmp / "org0/client.key")).get("/health")
                break
            except Exception:
                time.sleep(0.2)
        # 1) a client WITHOUT a CA-signed certificate must be refused at the TLS layer
        try:
            httpx.get(base + "/health", verify=str(tmp / "ca.crt"))
            print("[!] unauthenticated TLS client was ACCEPTED"); raise SystemExit(1)
        except httpx.HTTPError as e:
            print(f"[ok] connection without client certificate refused ({type(e).__name__})")

        reg = load_registry(tmp / "server/enrolled_orgs.json")
        def make(i):
            d = tmp / f"org{i}"
            http = mtls_client(base, str(d / "ca.crt"), str(d / "client.crt"), str(d / "client.key"))
            ident = serialization.load_pem_private_key((d / "identity.key").read_bytes(), None)
            return OrgAgent(i, ident, reg, http, spec.compute_spec_hash(), unsafe_skip_clip=(i == EVIL))
        agents = {i: make(i) for i in range(N)}
        print(f"[+] {N} orgs enrolled; org {EVIL} is MALICIOUS (skips clipping, sends sign-flipped x1000 update)"
              f"; defenses {'ON' if defended else 'OFF (1 cohort)'}")
        for r in range(ROUNDS):
            out = {}
            def work(i, r=r):
                X, y = data[i]
                def tr(p):
                    d = train_delta(p, X, y, epochs=3, seed=i * 100 + r)
                    return {k: -1000.0 * v for k, v in d.items()} if i == EVIL else d
                out[i] = agents[i].run_round(tr, poll_s=0.05, timeout_s=120)
            ts = [threading.Thread(target=work, args=(i,)) for i in range(N)]
            [t.start() for t in ts]; [t.join() for t in ts]
            from fedthreat.server import wire
            m = {k: wire.dec_array(v, "<f4") for k, v in agents[0].call("/v1/model").items()}
            res = out[0]
            print(f"  round {r + 1}: {res.get('status'):10s} cohorts_used={res.get('cohorts_used')} "
                  f"clipped={list((res.get('clipped') or {}).keys())}  global acc={accuracy(m, Xall, yall):.3f}")
        return {"acc": accuracy(m, Xall, yall)}
    finally:
        srv.terminate()
        try:
            srv.wait(timeout=5)
        except subprocess.TimeoutExpired:
            srv.kill()


if __name__ == "__main__":
    print("=== DEFENDED ==="); a = main(True)
    print("\n=== DEFENSES OFF (single cohort: plain secure sum, no robustness) ==="); b = main(False)
    print(f"\nfinal accuracy  defended={a['acc']:.3f}   undefended={b['acc']:.3f}")
