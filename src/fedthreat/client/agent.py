"""Organization-side agent: trains locally, clips/noises, runs the secure-aggregation client.

The agent is written as a generator (``steps``) that yields whenever it must wait for the
server, so it can be driven by a blocking loop in production (``run_round``) or stepped in
lockstep by tests/simulations without threads. Raw logs and features never leave this process;
the only things sent are public keys, ciphertexts, masked integers and revealed key shares.
"""

from __future__ import annotations

import secrets
import time
from typing import Any, Callable, Dict, Generator, Mapping, Optional

import numpy as np
from cryptography.hazmat.primitives.asymmetric import ed25519

from fedthreat.client.dp import add_gaussian_noise, clip_update
from fedthreat.secagg.protocol import ProtocolError, SecAggClient
from fedthreat.secagg.quantize import Quantizer
from fedthreat.server import wire

Params = Dict[str, np.ndarray]
TrainFn = Callable[[Params], Params]


class AgentError(RuntimeError):
    pass


class OrgAgent:
    def __init__(
        self,
        org_id: int,
        identity_key: ed25519.Ed25519PrivateKey,
        registry: Mapping[int, ed25519.Ed25519PublicKey],
        transport: Any,                       # httpx.Client or starlette TestClient (needs .post)
        spec_hash: str,
        clock: Callable[[], float] = time.time,
        unsafe_skip_clip: bool = False,       # for attack simulation ONLY: behaves like a malicious org
    ):
        self.org_id, self.identity_key, self.registry = org_id, identity_key, dict(registry)
        self.http, self.spec_hash, self.clock = transport, spec_hash, clock
        self.unsafe_skip_clip = unsafe_skip_clip
        self.last_update_norm: Optional[float] = None

    # ------------------------------------------------------------------ transport
    def call(self, path: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        payload = payload or {}
        ts, nonce = int(self.clock() * 1000), secrets.token_hex(16)
        sig = self.identity_key.sign(wire.signing_input(path, self.org_id, ts, nonce, payload)).hex()
        r = self.http.post(path, json={"client_id": self.org_id, "ts_ms": ts, "nonce": nonce, "payload": payload, "sig": sig})
        if r.status_code != 200:
            raise AgentError(f"{path} -> HTTP {r.status_code}: {r.text[:200]}")
        return r.json()

    # ------------------------------------------------------------------ one round
    def steps(self, train_fn: TrainFn) -> Generator[None, None, Dict[str, Any]]:
        st = self.call("/v1/status")
        cfg = st["round_config"]
        if cfg["spec_hash"] != self.spec_hash:
            raise AgentError("server feature spec differs from local spec; refusing to train")

        # 1) local training on a model version that is fixed until the current round completes
        raw = self.call("/v1/model")
        global_params = {k: wire.dec_array(v, "<f4") for k, v in raw.items()}
        delta = {k: v.astype(np.float32) for k, v in train_fn(global_params).items()}

        # 2) DP post-processing BEFORE anything leaves this process
        quant = Quantizer(cfg["quantizer_scale"], cfg["max_terms"])
        if self.unsafe_skip_clip:
            lim = quant.max_abs_value
            delta = {k: np.clip(v, -lim, lim) for k, v in delta.items()}
        else:
            delta, self.last_update_norm = clip_update(delta, cfg["clip_norm"])
            if cfg["client_noise_std"] > 0:
                delta = add_gaussian_noise(delta, cfg["client_noise_std"])

        # 3) wait for the next ADVERTISE window
        while True:
            st = self.call("/v1/status")
            if st["stage"] == "advertise":
                break
            if st["stage"].startswith("halted"):
                return {"status": "halted", "reason": "privacy budget exhausted"}
            yield
        round_id = bytes.fromhex(st["round_id"])
        my_round = st["round"]
        sc = SecAggClient(self.org_id, self.identity_key, round_id, quant)
        ad = sc.advertise()
        self.call("/v1/advertise", {"spec_hash": self.spec_hash, "c_pub": wire.b64e(ad.c_pub),
                                    "s_pub": wire.b64e(ad.s_pub), "sig": wire.b64e(ad.signature)})

        # 4) cohort keys -> share keys
        while True:
            st = self.call("/v1/status")
            if st["stage"] != "advertise":
                break
            yield
        if st["round_id"] != round_id.hex():
            return {"status": "aborted", **st["last_result"]}
        if not st.get("participating"):
            return {"status": "excluded"}
        ads = {int(k): wire.dec_ad(v) for k, v in self.call("/v1/cohort").items()}
        sc.receive_advertisements(ads, self.registry)
        self.call("/v1/shares/post", {"msgs": {str(p): wire.b64e(b) for p, b in sc.make_share_messages().items()}})

        # 5) masked input
        while True:
            st = self.call("/v1/status")
            if st["round_id"] != round_id.hex():
                return {"status": "aborted", **st["last_result"]}
            if st["stage"] in ("masked", "reveal"):
                break
            yield
        if not st.get("participating"):
            return {"status": "excluded"}
        shares = {int(k): wire.b64d(v) for k, v in self.call("/v1/shares/get").items()}
        sc.receive_share_messages(shares)
        masked = sc.masked_input(delta)
        self.call("/v1/masked", {"tensors": {k: wire.enc_array(v, "<u4") for k, v in masked.items()}})

        # 6) reveal shares for the survivor/dropout split the server reports
        while True:
            st = self.call("/v1/status")
            if st["round_id"] != round_id.hex():
                return {"status": "aborted", **st["last_result"]}
            if st["stage"] == "reveal":
                break
            yield
        sv = self.call("/v1/survivors")
        try:
            rev = sc.reveal(sv["survivors"], sv["dropped"])
        except ProtocolError as exc:
            raise AgentError(f"refusing to reveal: {exc}")
        self.call("/v1/reveal", wire.enc_reveal(rev))

        # 7) wait for the aggregate to land
        while True:
            st = self.call("/v1/status")
            if st["last_result"].get("round") == my_round:
                return dict(st["last_result"])
            yield

    def run_round(self, train_fn: TrainFn, poll_s: float = 0.5, timeout_s: float = 600.0) -> Dict[str, Any]:
        gen = self.steps(train_fn)
        t0 = time.time()
        try:
            while True:
                next(gen)
                if time.time() - t0 > timeout_s:
                    raise AgentError("timed out waiting for server")
                time.sleep(poll_s)
        except StopIteration as done:
            return done.value
