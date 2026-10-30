"""End-to-end tests: signed HTTP API + coordinator + org agents (lockstep, fake clock)."""
import json

import numpy as np
import pytest
from cryptography.hazmat.primitives.asymmetric import ed25519
from fastapi.testclient import TestClient

from fedthreat.client.agent import AgentError, OrgAgent
from fedthreat.server import wire
from fedthreat.server.app import create_app
from fedthreat.server.coordinator import RoundCoordinator, ServerConfig

D = 30
SPEC = "spec-hash-test"


class Clock:
    def __init__(self): self.t = 1_700_000_000.0
    def __call__(self): return self.t
    def advance(self, s): self.t += s


def make_env(n=9, **cfg):
    clock = Clock()
    keys = {i: ed25519.Ed25519PrivateKey.generate() for i in range(n)}
    reg = {i: k.public_key() for i, k in keys.items()}
    params = {"w": np.zeros(D, np.float32), "b": np.zeros((2, 3), np.float32)}
    config = ServerConfig(spec_hash=SPEC, **cfg)
    coord = RoundCoordinator(config, params, reg)
    app = create_app(coord, clock=clock)
    http = TestClient(app)
    return clock, keys, reg, coord, http


def agents(keys, reg, http, clock, spec=SPEC, evil=()):
    return {i: OrgAgent(i, k, reg, http, spec, clock, unsafe_skip_clip=i in evil) for i, k in keys.items()}


def const_train(val_w=0.01, val_b=0.01):
    return lambda p: {"w": np.full(D, val_w, np.float32), "b": np.full((2, 3), val_b, np.float32)}


def drive(gens, coord, clock, drop_when=None, max_passes=200):
    """Round-robin the agents' generators; jump the fake clock when nothing makes progress."""
    results, active, last = {}, dict(gens), None
    for _ in range(max_passes):
        for cid, g in list(active.items()):
            if drop_when and drop_when(cid, coord):
                del active[cid]
                continue
            try:
                next(g)
            except StopIteration as done:
                results[cid] = done.value
                del active[cid]
        if not active:
            return results
        sig = (coord.stage, coord.round_number, tuple(sorted((i, len(s.masked), len(s.reveals), len(s.share_msgs))
               for i, s in (coord.round.servers.items() if coord.round else []))), len(coord.round.ads) if coord.round else 0)
        if sig == last:
            clock.advance(61)
        last = sig
    raise AssertionError("did not converge")


def run_round(keys, reg, http, clock, coord, train=None, evil=(), drop_when=None, only=None):
    ag = agents(keys, reg, http, clock, evil=evil)
    ids = only if only is not None else list(ag)
    gens = {i: ag[i].steps(train(i) if callable(train) and train.__code__.co_argcount == 1 and False else (train or const_train())) for i in ids}
    return drive(gens, coord, clock, drop_when)


def test_honest_round_updates_model_exactly():
    clock, keys, reg, coord, http = make_env(9)
    res = run_round(keys, reg, http, clock, coord)
    assert all(r["status"] == "aggregated" and r["cohorts_used"] == 3 for r in res.values())
    np.testing.assert_allclose(coord.params["w"], 0.01, atol=2e-5)
    np.testing.assert_allclose(coord.params["b"], 0.01, atol=2e-5)
    assert coord.health()["epsilon"] is None and coord.health()["dp_enabled"] is False  # no fake epsilon


def _victims(c):
    """One client from each of two different cohorts (two in one 3-cohort is legitimately unrecoverable)."""
    return {c.round.cohorts[0][0], c.round.cohorts[1][0]}


def test_dropout_BEFORE_masked_input_is_recovered_end_to_end():
    """Exercises recovery of the dropped clients' pairwise masks via Shamir-reconstructed keys."""
    clock, keys, reg, coord, http = make_env(9)
    state = {}

    def drop(cid, c):
        if c.round is None or not c.round.servers:
            return False
        state.setdefault("v", _victims(c))
        return cid in state["v"] and c.stage.value == "masked" and any(cid in s.share_msgs and cid not in s.masked for s in c.round.servers.values())

    res = run_round(keys, reg, http, clock, coord, drop_when=drop)
    assert all(r["status"] == "aggregated" for r in res.values())
    assert any(len(s) for s in [state["v"]])
    np.testing.assert_allclose(coord.params["w"], 0.01, atol=2e-5)
    surv = [e for e in coord.audit if e["event"] == "round_aggregated"][0]["survivors"]
    assert not state["v"] & {x for v in surv.values() for x in v}  # victims really were excluded


def test_dropout_AFTER_masked_input_is_recovered_end_to_end():
    """Victim's masked input counts, but it never reveals: the other t members must suffice."""
    clock, keys, reg, coord, http = make_env(9)
    state = {}

    def drop(cid, c):
        if c.round is None or not c.round.servers:
            return False
        state.setdefault("v", _victims(c))
        return cid in state["v"] and any(cid in s.masked for s in c.round.servers.values())

    res = run_round(keys, reg, http, clock, coord, drop_when=drop)
    assert all(r["status"] == "aggregated" for r in res.values())
    np.testing.assert_allclose(coord.params["w"], 0.01, atol=2e-5)


def test_model_replacement_attacker_is_neutralised():
    """Attacker skips clipping and sends a 10^4x boosted update. Cohort-norm clipping + median over cohorts."""
    for evil in (0, 4, 8):
        clock, keys, reg, coord, http = make_env(9)
        ag = agents(keys, reg, http, clock, evil={evil})
        gens = {i: ag[i].steps((lambda p: {k: np.full_like(v, 5000.0) for k, v in const_train()(p).items()})
                                 if i == evil else const_train()) for i in ag}
        res = drive(gens, coord, clock)
        assert all(r["status"] == "aggregated" for r in res.values())
        np.testing.assert_allclose(coord.params["w"], 0.01, atol=2e-5), f"attacker {evil} moved the model"
        assert len(res[0]["clipped"]) == 1  # its cohort sum was clipped to the bound


def test_too_few_clients_aborts_and_leaves_model_unchanged():
    clock, keys, reg, coord, http = make_env(9)
    res = run_round(keys, reg, http, clock, coord, only=[0, 1, 2, 3, 4, 5])  # 6 < 9 => only 2 cohorts
    assert all(r["status"] == "aborted" for r in res.values())
    assert np.all(coord.params["w"] == 0)


def test_privacy_budget_halts_training():
    clock, keys, reg, coord, http = make_env(9, noise_multiplier=1.0, honest_min=2, max_epsilon=7.0)
    n = coord.accountant.rounds_remaining(1.0)
    assert 1 <= n <= 5
    for r in range(n):
        res = run_round(keys, reg, http, clock, coord)
        assert all(x["status"] == "aggregated" for x in res.values()), res
        assert coord.health()["epsilon"] <= 7.0
    res = run_round(keys, reg, http, clock, coord)
    assert all(x["status"] == "halted" for x in res.values())
    assert coord.health()["stage"] == "halted_privacy_budget"


# ---------------------------------------------------------------- authentication
def _env(path, cid, key, payload=None, ts=None, nonce="n" * 16, clock=None):
    payload = payload or {}
    ts = ts if ts is not None else int(clock() * 1000)
    sig = key.sign(wire.signing_input(path, cid, ts, nonce, payload)).hex()
    return {"client_id": cid, "ts_ms": ts, "nonce": nonce, "payload": payload, "sig": sig}


def test_auth_rejections():
    clock, keys, reg, coord, http = make_env(9)
    ok = http.post("/v1/status", json=_env("/v1/status", 0, keys[0], clock=clock))
    assert ok.status_code == 200
    # replay of the exact same envelope
    assert http.post("/v1/status", json=_env("/v1/status", 0, keys[0], clock=clock)).status_code == 401
    # wrong key (impersonation of org 1 by holder of org 0's key)
    assert http.post("/v1/status", json=_env("/v1/status", 1, keys[0], nonce="m" * 16, clock=clock)).status_code == 401
    # signature for a different path cannot be reused
    assert http.post("/v1/model", json=_env("/v1/status", 0, keys[0], nonce="p" * 16, clock=clock)).status_code == 401
    # stale timestamp
    assert http.post("/v1/status", json=_env("/v1/status", 0, keys[0], ts=int((clock() - 1000) * 1000), nonce="q" * 16)).status_code == 401
    # unknown org
    rogue = ed25519.Ed25519PrivateKey.generate()
    assert http.post("/v1/status", json=_env("/v1/status", 99, rogue, nonce="r" * 16, clock=clock)).status_code == 401
    # tampered payload
    e = _env("/v1/advertise", 0, keys[0], {"spec_hash": SPEC}, nonce="s" * 16, clock=clock)
    e["payload"]["spec_hash"] = "other"
    assert http.post("/v1/advertise", json=e).status_code == 401


def test_spec_hash_mismatch_is_enforced_both_sides():
    clock, keys, reg, coord, http = make_env(9)
    bad = OrgAgent(0, keys[0], reg, http, "different-spec", clock)
    with pytest.raises(AgentError):
        next(bad.steps(const_train()))  # client refuses a server with a different feature spec
    # a client that lies about its spec hash is rejected by the server
    good = OrgAgent(1, keys[1], reg, http, SPEC, clock)
    http.post("/v1/status", json=_env("/v1/status", 1, keys[1], clock=clock))  # starts the round
    liar = OrgAgent(1, keys[1], reg, http, SPEC, clock)
    with pytest.raises(AgentError):
        liar.call("/v1/advertise", {"spec_hash": "evil", "c_pub": wire.b64e(b"\0" * 32), "s_pub": wire.b64e(b"\0" * 32), "sig": wire.b64e(b"\0" * 64)})


def test_duplicate_advertise_cannot_inflate_cohort():
    clock, keys, reg, coord, http = make_env(9)
    ag = OrgAgent(0, keys[0], reg, http, SPEC, clock)
    g = ag.steps(const_train())
    next(g) if False else None
    run = ag.call("/v1/status")
    from fedthreat.secagg.protocol import SecAggClient
    from fedthreat.secagg.quantize import Quantizer
    sc = SecAggClient(0, keys[0], bytes.fromhex(run["round_id"]), Quantizer())
    ad = sc.advertise()
    body = {"spec_hash": SPEC, "c_pub": wire.b64e(ad.c_pub), "s_pub": wire.b64e(ad.s_pub), "sig": wire.b64e(ad.signature)}
    ag.call("/v1/advertise", body)
    with pytest.raises(AgentError):
        ag.call("/v1/advertise", body)
    assert len(coord.round.ads) == 1


def test_no_plaintext_update_appears_on_the_wire():
    clock, keys, reg, coord, http = make_env(9)
    seen = []
    orig = http.post
    http.post = lambda path, json=None, **kw: (seen.append((path, json)), orig(path, json=json, **kw))[1]
    secret = np.full(D, 0.01, np.float32)
    run_round(keys, reg, http, clock, coord)
    blob = json.dumps(seen)
    assert wire.b64e(secret.tobytes()) not in blob
    q = coord.quantizer.encode(secret).astype("<u4")
    assert wire.b64e(q.tobytes()) not in blob  # not even the quantized plaintext
