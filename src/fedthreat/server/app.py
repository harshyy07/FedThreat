"""FastAPI aggregation server.

Authentication model (two independent layers):
  1. Transport: TLS with REQUIRED client certificates issued by the consortium CA
     (run uvicorn with ``--ssl-cert-reqs 2``; see docker/server.Dockerfile). Only enrolled
     members can even open a connection.
  2. Application: every request carries an Ed25519 signature by the org's enrolled identity key
     over (path, client_id, timestamp, nonce, payload). The server derives identity from the
     SIGNATURE, never from a self-asserted field, rejects stale timestamps and replayed nonces.
The same identity key signs secure-aggregation key advertisements, so a malicious server cannot
inject or swap keys for an org.
"""

from __future__ import annotations

import os
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Dict

import numpy as np
from cryptography.hazmat.primitives.asymmetric import ed25519
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from fedthreat.secagg import keys as K
from fedthreat.server import wire
from fedthreat.server.coordinator import CoordinatorError, RoundCoordinator, ServerConfig


class Envelope(BaseModel):
    client_id: int
    ts_ms: int
    nonce: str = Field(min_length=16, max_length=64)
    payload: Dict[str, Any] = Field(default_factory=dict)
    sig: str


def load_registry(path: str | Path) -> Dict[int, ed25519.Ed25519PublicKey]:
    import json
    raw = json.loads(Path(path).read_text())
    return {int(k): K.ed25519_public_from_bytes(bytes.fromhex(v)) for k, v in raw.items()}


def create_app(coordinator: RoundCoordinator, clock: Callable[[], float] = time.time, max_skew_s: float = 120.0) -> FastAPI:
    app = FastAPI(title="FedThreat Aggregation Server", version="0.2.0")
    seen: "OrderedDict[tuple, float]" = OrderedDict()
    seen_lock = threading.Lock()

    def authenticate(path: str, env: Envelope) -> int:
        pub = coordinator.enrolled.get(env.client_id)
        if pub is None:
            raise HTTPException(401, "unknown client")
        now = clock()
        if abs(now - env.ts_ms / 1000.0) > max_skew_s:
            raise HTTPException(401, "stale or future timestamp")
        try:
            sig = bytes.fromhex(env.sig)
        except ValueError:
            raise HTTPException(401, "bad signature encoding")
        if not K.verify(pub, sig, wire.signing_input(path, env.client_id, env.ts_ms, env.nonce, env.payload)):
            raise HTTPException(401, "bad signature")
        key = (env.client_id, env.nonce)
        with seen_lock:
            if key in seen:
                raise HTTPException(401, "replayed nonce")
            seen[key] = now
            while seen and next(iter(seen.values())) < now - 2 * max_skew_s:
                seen.popitem(last=False)
        return env.client_id

    def guarded(path: str, env: Envelope, fn: Callable[[int, Dict[str, Any]], Any]) -> Any:
        cid = authenticate(path, env)
        try:
            return fn(cid, env.payload)
        except CoordinatorError as exc:
            raise HTTPException(409, str(exc))
        except (KeyError, ValueError, TypeError) as exc:
            raise HTTPException(400, f"malformed payload: {exc}")

    @app.get("/health")
    def health() -> Dict[str, Any]:
        return coordinator.health()

    @app.post("/v1/status")
    def status(env: Envelope) -> Any:
        def f(cid: int, p: Dict[str, Any]) -> Any:
            out = coordinator.status(cid, clock())
            out["round_config"] = coordinator.round_config()
            return out
        return guarded("/v1/status", env, f)

    @app.post("/v1/model")
    def model(env: Envelope) -> Any:
        def f(cid: int, p: Dict[str, Any]) -> Any:
            return {k: wire.enc_array(v, "<f4") for k, v in coordinator.get_model().items()}
        return guarded("/v1/model", env, f)

    @app.post("/v1/advertise")
    def advertise(env: Envelope) -> Any:
        def f(cid: int, p: Dict[str, Any]) -> Any:
            ad = wire.dec_ad({"client_id": cid, "c_pub": p["c_pub"], "s_pub": p["s_pub"], "sig": p["sig"]})
            coordinator.advertise(cid, str(p["spec_hash"]), ad, clock())
            return {"ok": True}
        return guarded("/v1/advertise", env, f)

    @app.post("/v1/cohort")
    def cohort(env: Envelope) -> Any:
        def f(cid: int, p: Dict[str, Any]) -> Any:
            return {str(k): wire.enc_ad(v) for k, v in coordinator.cohort_ads(cid, clock()).items()}
        return guarded("/v1/cohort", env, f)

    @app.post("/v1/shares/post")
    def shares_post(env: Envelope) -> Any:
        def f(cid: int, p: Dict[str, Any]) -> Any:
            coordinator.post_shares(cid, {int(k): wire.b64d(v) for k, v in p["msgs"].items()}, clock())
            return {"ok": True}
        return guarded("/v1/shares/post", env, f)

    @app.post("/v1/shares/get")
    def shares_get(env: Envelope) -> Any:
        def f(cid: int, p: Dict[str, Any]) -> Any:
            return {str(k): wire.b64e(v) for k, v in coordinator.get_shares(cid, clock()).items()}
        return guarded("/v1/shares/get", env, f)

    @app.post("/v1/masked")
    def masked(env: Envelope) -> Any:
        def f(cid: int, p: Dict[str, Any]) -> Any:
            shapes = coordinator.shapes
            tensors = {k: wire.dec_array(v, "<u4", shapes[k]).astype(np.uint32) for k, v in p["tensors"].items() if k in shapes}
            if len(tensors) != len(p["tensors"]):
                raise ValueError("unknown tensor name")
            coordinator.post_masked(cid, tensors, clock())
            return {"ok": True}
        return guarded("/v1/masked", env, f)

    @app.post("/v1/survivors")
    def survivors(env: Envelope) -> Any:
        return guarded("/v1/survivors", env, lambda cid, p: coordinator.survivors(cid, clock()))

    @app.post("/v1/reveal")
    def reveal(env: Envelope) -> Any:
        def f(cid: int, p: Dict[str, Any]) -> Any:
            coordinator.post_reveal(cid, wire.dec_reveal(cid, p), clock())
            return {"ok": True}
        return guarded("/v1/reveal", env, f)

    @app.post("/v1/audit")
    def audit(env: Envelope) -> Any:
        return guarded("/v1/audit", env, lambda cid, p: coordinator.audit[-int(p.get("last", 50)):])

    return app


def build_app_from_env() -> FastAPI:
    """Entrypoint for ``uvicorn fedthreat.server.app:app`` (configured via environment)."""
    from fedthreat.models.params import init_mlp_params

    env = os.environ.get
    cfg = ServerConfig(
        spec_hash=env("FEDTHREAT_SPEC_HASH", "dev"),
        min_cohort_size=int(env("FEDTHREAT_MIN_COHORT_SIZE", "3")),
        min_cohorts=int(env("FEDTHREAT_MIN_COHORTS", "3")),
        clip_norm=float(env("FEDTHREAT_CLIP_NORM", "1.0")),
        noise_multiplier=float(env("FEDTHREAT_NOISE_MULTIPLIER", "0.0")),
        max_epsilon=float(env("FEDTHREAT_MAX_EPSILON", "10.0")),
        robust=env("FEDTHREAT_ROBUST", "median"),
    )
    registry = load_registry(env("FEDTHREAT_REGISTRY", "/certs/enrolled_orgs.json"))
    npz = env("FEDTHREAT_INITIAL_PARAMS")
    params = dict(np.load(npz)) if npz else init_mlp_params(int(env("FEDTHREAT_INPUT_DIM", "16")), seed=0)
    return create_app(RoundCoordinator(cfg, params, registry))


def __getattr__(name: str) -> Any:  # lazy: `uvicorn fedthreat.server.app:app` builds from env on first access
    if name == "app":
        app = build_app_from_env()
        globals()["app"] = app
        return app
    raise AttributeError(name)
