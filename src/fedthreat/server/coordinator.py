"""Round coordinator: the aggregation server's state machine (no web framework, no torch).

One training round:  ADVERTISE -> SHARE -> MASKED -> REVEAL -> (aggregate) -> IDLE
Every stage has a deadline; late or missing clients simply drop out (secure aggregation
recovers). A round is ABORTED (model unchanged) rather than aggregated in a way that would
silently lose the robustness or privacy guarantees we advertise.
"""

from __future__ import annotations

import hashlib
import json
import math
import secrets
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Set

import numpy as np
from cryptography.hazmat.primitives.asymmetric import ed25519

from fedthreat.aggregation.cohort import InsufficientCohortsError, assign_cohorts, combine_cohort_means
from fedthreat.aggregation.robust import CoordinateWiseMedianAggregator, KrumAggregator, TrimmedMeanAggregator
from fedthreat.base import Aggregator
from fedthreat.client.dp import PrivacyAccountant, l2_norm, per_client_noise_std
from fedthreat.secagg.protocol import Advertisement, ProtocolError, Reveal, SecAggServer
from fedthreat.secagg.quantize import Quantizer

Params = Dict[str, np.ndarray]


class Stage(str, Enum):
    IDLE = "idle"
    ADVERTISE = "advertise"
    SHARE = "share"
    MASKED = "masked"
    REVEAL = "reveal"
    HALTED = "halted_privacy_budget"


class CoordinatorError(Exception):
    """Request is invalid for the current state (maps to HTTP 4xx)."""


@dataclass
class ServerConfig:
    spec_hash: str
    min_cohort_size: int = 3
    min_cohorts: int = 3                 # >=3 cohorts: median over cohorts tolerates 1 malicious org
    clip_norm: float = 1.0
    noise_multiplier: float = 0.0        # 0 => DP DISABLED (reported honestly as epsilon = null)
    honest_min: int = 2                  # honest clients per cohort assumed for noise calibration
    target_delta: float = 1e-5
    max_epsilon: float = 10.0
    robust: str = "median"               # median | trimmed_mean | krum
    stage_timeout_s: float = 60.0
    norm_slack: float = 1.2              # multiplier on the expected noise norm in the cohort norm screen
    server_lr: float = 1.0
    quantizer_scale: float = float(1 << 16)
    max_cohort_size: int = 64


def make_robust(name: str, n_cohorts: int) -> Aggregator:
    if name == "median":
        return CoordinateWiseMedianAggregator()
    if name == "trimmed_mean":
        return TrimmedMeanAggregator(beta=0.2, min_trim=1)
    if name == "krum":
        return KrumAggregator(num_malicious=max(1, (n_cohorts - 3) // 2), m=1)
    raise ValueError(f"unknown robust aggregator {name!r}")


@dataclass
class _Round:
    round_id: bytes
    number: int
    stage: Stage
    deadline: float
    ads: Dict[int, Advertisement] = field(default_factory=dict)
    cohorts: List[List[int]] = field(default_factory=list)
    cohort_of: Dict[int, int] = field(default_factory=dict)
    servers: Dict[int, SecAggServer] = field(default_factory=dict)  # cohort idx -> server (live cohorts only)
    failed_cohorts: Dict[int, str] = field(default_factory=dict)


class RoundCoordinator:
    def __init__(
        self,
        config: ServerConfig,
        initial_params: Params,
        enrolled: Mapping[int, ed25519.Ed25519PublicKey],
    ):
        self.cfg = config
        self.params: Params = {k: v.astype(np.float32).copy() for k, v in initial_params.items()}
        self.shapes = {k: v.shape for k, v in self.params.items()}
        self.dim = int(sum(v.size for v in self.params.values()))
        self.enrolled = dict(enrolled)
        self.accountant = PrivacyAccountant(config.target_delta, config.max_epsilon)
        self.round_number = 0
        self.round: Optional[_Round] = None
        self.stage = Stage.IDLE
        self.audit: List[Dict[str, Any]] = []
        self.last_result: Dict[str, Any] = {}
        self._lock = threading.RLock()
        self._max_terms = config.max_cohort_size + 1
        self.quantizer = Quantizer(scale=config.quantizer_scale, max_terms=self._max_terms)
        if config.noise_multiplier > 0 and config.honest_min > config.min_cohort_size:
            raise ValueError("honest_min cannot exceed the minimum cohort size")
        # worst-case encodable magnitude check: clip + noise must fit the ring
        if config.clip_norm > self.quantizer.max_abs_value:
            raise ValueError("clip_norm too large for the fixed-point ring")
        if config.noise_multiplier > 0:
            std = per_client_noise_std(config.noise_multiplier, config.clip_norm, config.honest_min)
            if 6.0 * std + config.clip_norm > self.quantizer.max_abs_value:
                raise ValueError("per-client DP noise (6 sigma) would overflow the fixed-point ring; lower sigma or clip_norm")

    # ---------------------------------------------------------------- helpers
    @property
    def dp_enabled(self) -> bool:
        return self.cfg.noise_multiplier > 0

    def round_config(self) -> Dict[str, Any]:
        return {
            "clip_norm": self.cfg.clip_norm,
            "noise_multiplier": self.cfg.noise_multiplier,
            "honest_min": self.cfg.honest_min,
            "client_noise_std": per_client_noise_std(self.cfg.noise_multiplier, self.cfg.clip_norm, self.cfg.honest_min)
            if self.dp_enabled else 0.0,
            "quantizer_scale": self.cfg.quantizer_scale,
            "max_terms": self._max_terms,
            "spec_hash": self.cfg.spec_hash,
        }

    def _log(self, event: str, **info: Any) -> None:
        prev = self.audit[-1]["hash"] if self.audit else "0" * 64
        entry = {"seq": len(self.audit), "t": time.time(), "round": self.round_number, "event": event, **info, "prev": prev}
        entry["hash"] = hashlib.sha256(json.dumps(entry, sort_keys=True, default=str).encode()).hexdigest()
        self.audit.append(entry)

    def _abort(self, reason: str) -> None:
        self._log("round_aborted", reason=reason)
        self.last_result = {"round": self.round_number, "status": "aborted", "reason": reason}
        self.round, self.stage = None, Stage.IDLE

    def _require(self, client_id: int, stage: Stage) -> _Round:
        if client_id not in self.enrolled:
            raise CoordinatorError("client not enrolled")
        if self.round is None or self.stage != stage:
            raise CoordinatorError(f"not accepting this message now (stage={self.stage.value})")
        return self.round

    # ---------------------------------------------------------------- lifecycle
    def tick(self, now: Optional[float] = None) -> None:
        now = time.time() if now is None else now
        with self._lock:
            if self.stage == Stage.IDLE:
                self._maybe_start(now)
                return
            r = self.round
            if r is None or self.stage == Stage.HALTED:
                return
            if self.stage == Stage.ADVERTISE:
                if len(r.ads) >= len(self.enrolled) or now >= r.deadline:
                    self._close_advertise(now)
            elif self.stage == Stage.SHARE:
                live = [s for s in r.servers.values()]
                if all(len(s.share_msgs) == len(s.ads) for s in live) or now >= r.deadline:
                    self._close_share(now)
            elif self.stage == Stage.MASKED:
                if all(len(s.masked) == len(s.active) for s in r.servers.values()) or now >= r.deadline:
                    self._close_masked(now)
            elif self.stage == Stage.REVEAL:
                if all(len(s.reveals) == len(s.masked) for s in r.servers.values()) or now >= r.deadline:
                    self._finalize(now)

    def _maybe_start(self, now: float) -> None:
        if self.dp_enabled and not self.accountant.can_afford(self.cfg.noise_multiplier, 1.0):
            self.stage = Stage.HALTED
            self._log("halted", reason="privacy budget exhausted", epsilon=self.accountant.get_epsilon())
            return
        self.round_number += 1
        self.round = _Round(secrets.token_bytes(16), self.round_number, Stage.ADVERTISE, now + self.cfg.stage_timeout_s)
        self.stage = Stage.ADVERTISE
        self._log("round_started", round_id=self.round.round_id.hex())

    def _close_advertise(self, now: float) -> None:
        r = self.round
        assert r is not None
        need = self.cfg.min_cohort_size * self.cfg.min_cohorts
        if len(r.ads) < need:
            return self._abort(f"only {len(r.ads)} clients advertised; need >= {need} for {self.cfg.min_cohorts} cohorts")
        try:
            cohorts = assign_cohorts(sorted(r.ads), self.cfg.min_cohort_size)
        except InsufficientCohortsError as exc:
            return self._abort(str(exc))
        if len(cohorts) < self.cfg.min_cohorts or max(len(c) for c in cohorts) > self.cfg.max_cohort_size:
            return self._abort("cannot form valid cohorts")
        r.cohorts = cohorts
        for idx, members in enumerate(cohorts):
            srv = SecAggServer(r.round_id, members, self.enrolled, self.quantizer, self.shapes)
            for m in members:
                srv.add_advertisement(r.ads[m])
                r.cohort_of[m] = idx
            r.servers[idx] = srv
        self.stage, r.stage, r.deadline = Stage.SHARE, Stage.SHARE, now + self.cfg.stage_timeout_s
        self._log("cohorts_assigned", sizes=[len(c) for c in cohorts])

    def _drop_failed(self, why: str, bad: List[int]) -> None:
        r = self.round
        assert r is not None
        for idx in bad:
            r.failed_cohorts[idx] = why
            r.servers.pop(idx, None)

    def _check_enough_cohorts(self) -> bool:
        r = self.round
        assert r is not None
        if len(r.servers) < self.cfg.min_cohorts:
            self._abort(f"only {len(r.servers)} cohorts survived (< {self.cfg.min_cohorts}); robustness would be void")
            return False
        return True

    def _close_share(self, now: float) -> None:
        r = self.round
        assert r is not None
        self._drop_failed("too few clients shared keys", [i for i, s in r.servers.items() if len(s.active) < s.threshold])
        if not self._check_enough_cohorts():
            return
        self.stage, r.stage, r.deadline = Stage.MASKED, Stage.MASKED, now + self.cfg.stage_timeout_s

    def _close_masked(self, now: float) -> None:
        r = self.round
        assert r is not None
        self._drop_failed("too few masked inputs", [i for i, s in r.servers.items() if len(s.masked) < s.threshold])
        if not self._check_enough_cohorts():
            return
        self.stage, r.stage, r.deadline = Stage.REVEAL, Stage.REVEAL, now + self.cfg.stage_timeout_s

    def _finalize(self, now: float) -> None:
        r = self.round
        assert r is not None
        means: List[Params] = []
        used: List[int] = []
        rejected: Dict[int, str] = {}
        clipped: Dict[int, str] = {}
        released_any = False
        for idx, srv in list(r.servers.items()):
            if len(srv.reveals) < srv.threshold:
                rejected[idx] = "too few reveals"
                continue
            try:
                total = srv.finalize()
            except ProtocolError as exc:
                rejected[idx] = f"unmask failed: {exc}"
                continue
            released_any = True
            s = len(srv.survivors)
            bound = s * self.cfg.clip_norm
            if self.dp_enabled:
                std_sum = per_client_noise_std(self.cfg.noise_multiplier, self.cfg.clip_norm, self.cfg.honest_min) * math.sqrt(s)
                bound += self.cfg.norm_slack * std_sum * math.sqrt(self.dim)
            bound *= 1.0 + 1e-3
            norm = l2_norm(total)
            if not math.isfinite(norm):
                rejected[idx] = "non-finite cohort sum"
                continue
            if norm > bound:
                # Clip rather than reject: dropping the cohort would let ONE attacker push the number of
                # valid cohorts below min_cohorts and abort every round (a denial of service).
                total = {k: (v * (bound / norm)).astype(np.float32) for k, v in total.items()}
                clipped[idx] = f"norm {norm:.3g} clipped to {bound:.3g}"
            means.append({k: (v / s).astype(np.float32) for k, v in total.items()})
            used.append(idx)
        # privacy accounting: spend as soon as ANY noisy cohort sum was reconstructed (conservative)
        if released_any and self.dp_enabled:
            self.accountant.step(self.cfg.noise_multiplier, 1.0)
        if len(means) < self.cfg.min_cohorts:
            self._log("cohort_rejections", rejected={str(k): v for k, v in rejected.items()})
            return self._abort(f"{len(means)} valid cohorts < {self.cfg.min_cohorts}; model NOT updated")
        agg = make_robust(self.cfg.robust, len(means))
        zero = {k: np.zeros_like(v) for k, v in self.params.items()}
        combined = combine_cohort_means(means, agg, zero)
        for k in self.params:
            self.params[k] = (self.params[k] + self.cfg.server_lr * combined[k]).astype(np.float32)
        self.last_result = {
            "round": self.round_number, "status": "aggregated", "cohorts_used": len(means),
            "rejected": {str(k): v for k, v in rejected.items()},
            "clipped": {str(k): v for k, v in clipped.items()},
            "epsilon": self.accountant.get_epsilon() if self.dp_enabled else None,
        }
        self._log("round_aggregated", cohorts_used=len(means), rejected=self.last_result["rejected"],
                  survivors={str(i): r.servers[i].survivors for i in used},
                  epsilon=self.last_result["epsilon"])
        self.round, self.stage = None, Stage.IDLE

    # ---------------------------------------------------------------- client-facing operations
    def status(self, client_id: int, now: Optional[float] = None) -> Dict[str, Any]:
        with self._lock:
            self.tick(now)
            if client_id not in self.enrolled:
                raise CoordinatorError("client not enrolled")
            r = self.round
            out: Dict[str, Any] = {
                "stage": self.stage.value, "round": self.round_number,
                "round_id": r.round_id.hex() if r else None,
                "in_cohort": bool(r and client_id in r.cohort_of and r.cohort_of[client_id] in r.servers),
                "last_result": self.last_result,
            }
            if r and self.stage != Stage.ADVERTISE:
                out["participating"] = client_id in r.cohort_of and r.cohort_of[client_id] in r.servers
            return out

    def get_model(self) -> Params:
        with self._lock:
            return {k: v.copy() for k, v in self.params.items()}

    def advertise(self, client_id: int, spec_hash: str, ad: Advertisement, now: Optional[float] = None) -> None:
        with self._lock:
            self.tick(now)
            r = self._require(client_id, Stage.ADVERTISE)
            if spec_hash != self.cfg.spec_hash:
                raise CoordinatorError("feature spec hash mismatch")
            if client_id in r.ads or ad.client_id != client_id:
                raise CoordinatorError("duplicate or mismatched advertisement")
            # verify signature now so garbage never enters cohorts
            from fedthreat.secagg import keys as K
            msg = Advertisement.signed_bytes(r.round_id, client_id, ad.c_pub, ad.s_pub)
            if len(ad.c_pub) != 32 or len(ad.s_pub) != 32 or not K.verify(self.enrolled[client_id], ad.signature, msg):
                raise CoordinatorError("invalid advertisement")
            r.ads[client_id] = ad
            self.tick(now)

    def _cohort_srv(self, client_id: int) -> SecAggServer:
        r = self.round
        assert r is not None
        idx = r.cohort_of.get(client_id)
        if idx is None or idx not in r.servers:
            raise CoordinatorError("you are not in a live cohort this round")
        return r.servers[idx]

    def cohort_ads(self, client_id: int, now: Optional[float] = None) -> Dict[int, Advertisement]:
        with self._lock:
            self.tick(now)
            if self.round is None or self.stage not in (Stage.SHARE, Stage.MASKED, Stage.REVEAL):
                raise CoordinatorError(f"cohorts not ready (stage={self.stage.value})")
            if client_id not in self.enrolled:
                raise CoordinatorError("client not enrolled")
            return dict(self._cohort_srv(client_id).ads)

    def post_shares(self, client_id: int, msgs: Mapping[int, bytes], now: Optional[float] = None) -> None:
        with self._lock:
            self.tick(now)
            self._require(client_id, Stage.SHARE)
            try:
                self._cohort_srv(client_id).add_shares(client_id, msgs)
            except ProtocolError as exc:
                raise CoordinatorError(str(exc)) from exc
            self.tick(now)

    def get_shares(self, client_id: int, now: Optional[float] = None) -> Dict[int, bytes]:
        """Only released once the SHARE stage is closed, so every client sees the same final U2."""
        with self._lock:
            self.tick(now)
            if self.stage not in (Stage.MASKED, Stage.REVEAL) or self.round is None:
                raise CoordinatorError(f"share stage still open (stage={self.stage.value})")
            if client_id not in self.enrolled:
                raise CoordinatorError("client not enrolled")
            return self._cohort_srv(client_id).shares_for(client_id)

    def post_masked(self, client_id: int, tensors: Mapping[str, np.ndarray], now: Optional[float] = None) -> None:
        with self._lock:
            self.tick(now)
            self._require(client_id, Stage.MASKED)
            try:
                self._cohort_srv(client_id).add_masked(client_id, dict(tensors))
            except ProtocolError as exc:
                raise CoordinatorError(str(exc)) from exc
            self.tick(now)

    def survivors(self, client_id: int, now: Optional[float] = None) -> Dict[str, List[int]]:
        with self._lock:
            self.tick(now)
            if self.stage != Stage.REVEAL or self.round is None:
                raise CoordinatorError(f"masked stage still open (stage={self.stage.value})")
            if client_id not in self.enrolled:
                raise CoordinatorError("client not enrolled")
            srv = self._cohort_srv(client_id)
            return {"survivors": srv.survivors, "dropped": srv.dropped}

    def post_reveal(self, client_id: int, rev: Reveal, now: Optional[float] = None) -> None:
        with self._lock:
            self.tick(now)
            self._require(client_id, Stage.REVEAL)
            try:
                self._cohort_srv(client_id).add_reveal(rev)
            except ProtocolError as exc:
                raise CoordinatorError(str(exc)) from exc
            self.tick(now)

    def health(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "stage": self.stage.value, "round": self.round_number,
                "dp_enabled": self.dp_enabled,
                "epsilon": self.accountant.get_epsilon() if self.dp_enabled else None,
                "delta": self.cfg.target_delta if self.dp_enabled else None,
                "rounds_remaining": self.accountant.rounds_remaining(self.cfg.noise_multiplier, 1.0, cap=10_000)
                if self.dp_enabled else None,
                "min_cohorts": self.cfg.min_cohorts, "robust": self.cfg.robust,
            }
