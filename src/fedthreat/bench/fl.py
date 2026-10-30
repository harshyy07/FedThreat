"""Numpy federated-learning experiment engine used by the benchmark runner.

Reuses the production pieces: ``assign_cohorts`` / ``combine_cohort_means`` / robust aggregators,
``clip_update`` / ``per_client_noise_std`` and the RDP ``PrivacyAccountant``. The cohort path
replicates ``server.coordinator._finalize`` (sum -> norm bound -> mean -> robust combine); the
cryptographic masking is skipped here because secure aggregation is exact (see tests/test_secagg.py),
which keeps sweeps fast. ``tests/test_bench.py`` checks equivalence with the in-process protocol.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from fedthreat.aggregation.cohort import assign_cohorts, combine_cohort_means
from fedthreat.aggregation.robust import CoordinateWiseMedianAggregator, KrumAggregator, TrimmedMeanAggregator
from fedthreat.bench import metrics as M
from fedthreat.bench.data import TRIGGER_COL, TRIGGER_VAL, HELD_OUT, Benchmark, OrgData, build_orgs
from fedthreat.client.dp import PrivacyAccountant, add_gaussian_noise, clip_update, l2_norm, per_client_noise_std
from fedthreat.client.numpy_trainer import init_linear_params, predict_proba, train_delta

Params = Dict[str, np.ndarray]
SECURE_MODES = {"secure_sum", "cohort_median", "cohort_trimmed"}
PLAIN_MODES = {"fedavg", "fedavg_clip", "median", "trimmed_mean", "krum"}


@dataclass
class RunConfig:
    experiment: str = "adhoc"
    mode: str = "federated"            # federated | local | central
    aggregator: str = "fedavg"
    n_orgs: int = 9
    alpha: float = 1.0                 # Dirichlet label skew (0.1 severe .. 100 ~IID)
    rounds: int = 25
    local_epochs: int = 2
    lr: float = 0.1
    clip: float = 1.0                  # 0 disables client clipping
    sigma: float = 0.0                 # DP noise multiplier (0 = off)
    honest_min: int = 2
    cohort_size: int = 3
    attack: str = "none"               # none|sign_flip|gaussian|label_flip|backdoor|lie
    num_malicious: int = 0
    attack_scale: float = 20.0
    lie_z: float = 1.0
    norm_slack: float = 1.2
    seed: int = 0
    target_delta: float = 1e-5


# ------------------------------------------------------------------ helpers
def _flat(u: Params) -> np.ndarray:
    return np.concatenate([v.ravel() for v in u.values()])


def _unflat(v: np.ndarray, like: Params) -> Params:
    out, i = {}, 0
    for k, a in like.items():
        out[k] = v[i:i + a.size].reshape(a.shape).astype(np.float32); i += a.size
    return out


def _scale(u: Params, s: float) -> Params:
    return {k: (v * s).astype(np.float32) for k, v in u.items()}


def _zeros(like: Params) -> Params:
    return {k: np.zeros_like(v) for k, v in like.items()}


def _mean(us: List[Params], w: Optional[List[float]] = None) -> Params:
    w = np.asarray(w if w is not None else [1.0] * len(us), np.float64); w = w / w.sum()
    return {k: sum(wi * u[k] for wi, u in zip(w, us)).astype(np.float32) for k in us[0]}


def _robust(name: str, n: int):
    if name == "median":
        return CoordinateWiseMedianAggregator()
    if name == "trimmed_mean":
        return TrimmedMeanAggregator(0.2, min_trim=1)
    if name == "krum":
        return KrumAggregator(num_malicious=max(1, int(0.2 * n)), m=1)
    raise ValueError(name)


# ------------------------------------------------------------------ server-side aggregation
def server_aggregate(cfg: RunConfig, updates: List[Params], weights: List[float], dim: int,
                     rng: random.Random) -> Params:
    like = updates[0]
    zero = _zeros(like)
    a = cfg.aggregator
    if a == "fedavg":
        return _mean(updates, weights)
    if a == "fedavg_clip":
        return _mean([clip_update(u, cfg.clip)[0] for u in updates], weights)
    if a in ("median", "trimmed_mean", "krum"):
        return _robust(a, len(updates)).aggregate(updates, [1.0] * len(updates), zero)
    if a in SECURE_MODES:
        if cfg.clip <= 0:
            raise ValueError("cohort modes need clip > 0 (sensitivity / norm bound)")
        size = len(updates) if a == "secure_sum" else cfg.cohort_size
        cohorts = assign_cohorts(range(len(updates)), size, rng)
        means = []
        for c in cohorts:
            s = len(c)
            total = {k: sum(updates[i][k] for i in c) for k in like}
            bound = s * cfg.clip
            if cfg.sigma > 0:
                std_sum = per_client_noise_std(cfg.sigma, cfg.clip, cfg.honest_min) * math.sqrt(s)
                bound += cfg.norm_slack * std_sum * math.sqrt(dim)
            bound *= 1.001
            nrm = l2_norm(total)
            if nrm > bound:
                total = _scale(total, bound / nrm)
            means.append(_scale(total, 1.0 / s))
        if a == "secure_sum":
            return means[0]
        robust = CoordinateWiseMedianAggregator() if a == "cohort_median" else TrimmedMeanAggregator(0.2, min_trim=1)
        return combine_cohort_means(means, robust, zero)
    raise ValueError(f"unknown aggregator {a}")


# ------------------------------------------------------------------ evaluation
def evaluate(params: Params, bench: Benchmark) -> Dict[str, float]:
    t = bench.test
    s = predict_proba(params, t.X)[:, 1]
    held = t.fam == HELD_OUT
    out = {
        "pr_auc": M.average_precision(t.y, s),
        "recall_1fpr": M.recall_at_fpr(t.y, s, 0.01),
        "recall_0.1fpr": M.recall_at_fpr(t.y, s, 0.001),
        "heldout_recall_1fpr": M.recall_at_fpr(t.y, s, 0.01, pos_mask=held),
    }
    atk = t.y == 1
    Xt = t.X[atk].copy(); Xt[:, TRIGGER_COL] = TRIGGER_VAL
    out["miss_clean"] = float((predict_proba(params, t.X[atk])[:, 1] < 0.5).mean())
    out["asr"] = float((predict_proba(params, Xt)[:, 1] < 0.5).mean())
    return out


# ------------------------------------------------------------------ experiment runners
def _train(params: Params, d: OrgData, cfg: RunConfig, seed: int, epochs: Optional[int] = None,
           y: Optional[np.ndarray] = None, X: Optional[np.ndarray] = None) -> Params:
    return train_delta(params, d.X if X is None else X, d.y if y is None else y,
                       epochs=cfg.local_epochs if epochs is None else epochs, lr=cfg.lr, seed=seed)


def _poison_backdoor(d: OrgData, rng: np.random.Generator):
    X, y = d.X.copy(), d.y.copy()
    idx = np.where(y == 1)[0]
    pick = rng.choice(idx, size=max(1, len(idx) // 2), replace=False) if len(idx) else idx
    X[pick, TRIGGER_COL] = TRIGGER_VAL
    y[pick] = 0
    return X, y


def run_federated(cfg: RunConfig, bench: Benchmark) -> Dict[str, Any]:
    rng_np, rng_py = np.random.default_rng(cfg.seed), random.Random(cfg.seed)
    ids = sorted(bench.orgs)
    n = len(ids)
    bad = set(rng_np.permutation(ids)[:cfg.num_malicious].tolist()) if cfg.attack != "none" else set()
    secure = cfg.aggregator in SECURE_MODES
    noise_std = 0.0
    if cfg.sigma > 0:
        noise_std = per_client_noise_std(cfg.sigma, cfg.clip, cfg.honest_min if secure else 1)
    params = init_linear_params(bench.dim)
    acct = PrivacyAccountant(cfg.target_delta, 1e9)
    for r in range(cfg.rounds):
        # honest computation for everyone (malicious orgs need it as a reference / base)
        raw = {i: _train(params, bench.orgs[i], cfg, seed=cfg.seed * 7919 + r * 131 + i) for i in ids}
        proc = {}
        for i in ids:
            u = raw[i]
            if cfg.clip > 0:
                u, _ = clip_update(u, cfg.clip)
            proc[i] = add_gaussian_noise(u, noise_std, rng_np) if noise_std > 0 else u
        honest_ids = [i for i in ids if i not in bad]
        sent = {}
        for i in ids:
            if i not in bad:
                sent[i] = proc[i]; continue
            if cfg.attack == "sign_flip":
                sent[i] = _scale(raw[i], -cfg.attack_scale)
            elif cfg.attack == "gaussian":
                sent[i] = _unflat(rng_np.normal(0, cfg.attack_scale, size=_flat(raw[i]).size), raw[i])
            elif cfg.attack == "label_flip":
                d = bench.orgs[i]
                u = _train(params, d, cfg, seed=cfg.seed * 31 + r + i, y=1 - d.y)
                sent[i] = clip_update(u, cfg.clip)[0] if cfg.clip > 0 else u  # stays stealthy: obeys clipping
            elif cfg.attack == "backdoor":
                Xp, yp = _poison_backdoor(bench.orgs[i], rng_np)
                u = _train(params, bench.orgs[i], cfg, seed=cfg.seed * 37 + r + i, X=Xp, y=yp)
                sent[i] = _scale(u, cfg.attack_scale)  # model-replacement style boost, no clipping
            elif cfg.attack == "lie":  # "A Little Is Enough": mean - z*std of honest (already-processed) updates
                H = np.stack([_flat(proc[j]) for j in honest_ids])
                sent[i] = _unflat(H.mean(0) - cfg.lie_z * H.std(0), raw[i])
            else:
                raise ValueError(cfg.attack)
        order = ids
        updates = [sent[i] for i in order]
        weights = [float(len(bench.orgs[i].y)) for i in order]
        delta = server_aggregate(cfg, updates, weights, dim=int(_flat(updates[0]).size), rng=rng_py)
        params = {k: (params[k] + delta[k]).astype(np.float32) for k in params}
        if cfg.sigma > 0:
            acct.step(cfg.sigma, 1.0)
    res = evaluate(params, bench)
    res["epsilon"] = acct.get_epsilon() if cfg.sigma > 0 else float("inf")
    res["n_malicious"] = len(bad)
    return res


def run_local(cfg: RunConfig, bench: Benchmark) -> Dict[str, Any]:
    """Each org trains alone (same total epochs). Reports mean over all orgs and over orgs that never saw the held-out family."""
    rows, nonowner = [], []
    for i, d in bench.orgs.items():
        p = init_linear_params(bench.dim)
        delta = _train(p, d, cfg, seed=cfg.seed + i, epochs=cfg.rounds * cfg.local_epochs)
        p = {k: p[k] + delta[k] for k in p}
        e = evaluate(p, bench); rows.append(e)
        if i != bench.held_out_org:
            nonowner.append(e)
    out = {k: float(np.nanmean([r[k] for r in rows])) for k in rows[0]}
    out.update({f"nonowner_{k}": float(np.nanmean([r[k] for r in nonowner])) for k in rows[0]})
    out["epsilon"] = float("inf")
    return out


def run_central(cfg: RunConfig, bench: Benchmark) -> Dict[str, Any]:
    X = np.vstack([d.X for d in bench.orgs.values()]); y = np.concatenate([d.y for d in bench.orgs.values()])
    pooled = OrgData(X, y, np.array([]))
    p = init_linear_params(bench.dim)
    delta = _train(p, pooled, cfg, seed=cfg.seed, epochs=cfg.rounds * cfg.local_epochs)
    out = evaluate({k: p[k] + delta[k] for k in p}, bench)
    out["epsilon"] = float("inf")
    return out


_CACHE: Dict[tuple, Benchmark] = {}


def run(cfg: RunConfig) -> Dict[str, Any]:
    key = (cfg.n_orgs, cfg.alpha, cfg.seed)
    if key not in _CACHE:
        _CACHE.clear(); _CACHE[key] = build_orgs(cfg.n_orgs, cfg.alpha, cfg.seed)
    bench = _CACHE[key]
    t0 = time.time()
    fn = {"federated": run_federated, "local": run_local, "central": run_central}[cfg.mode]
    res = fn(cfg, bench)
    return {**asdict(cfg), **res, "seconds": round(time.time() - t0, 2)}
