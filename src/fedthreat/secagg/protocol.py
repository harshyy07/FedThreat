"""Pairwise-masking secure aggregation (Bonawitz et al., CCS 2017), per cohort.

Trust boundary: ``SecAggClient`` runs inside an organization. ``SecAggServer`` runs on the
aggregation server and only ever sees (a) signed public keys, (b) AES-GCM ciphertexts it
cannot read, (c) masked uint32 vectors, and (d) Shamir shares that survivors choose to
reveal. Nothing in this module lets the server see an individual plaintext update as
long as the security assumptions below hold.

Stages (per cohort):
  0 advertise : each client publishes (c_pub, s_pub) signed with its enrolled Ed25519 key
  1 share     : each client Shamir-shares its self-mask seed b_i and its mask private key
                s_i among cohort members, encrypting share j under a c-key shared with j
  2 masked    : y_i = enc(delta_i) + PRG(b_i) + sum_j +/- PRG(s_ij)    (mod 2^32)
  3 reveal    : for every survivor u a client reveals its share of b_u; for every dropped
                client d it reveals its share of s_d -- NEVER both for the same client
  4 finalize  : server removes survivors' self-masks and dropouts' pairwise masks

Security assumptions (also stated in docs/threat_model.md):
  * server is honest-but-curious for secure aggregation (an actively malicious server that
    lies about the survivor set needs Bonawitz's extra consistency round: NOT implemented);
  * fewer than ``threshold`` cohort members collude with the server;
  * the enrolled-identity registry (Ed25519 keys) is correct.
"""

from __future__ import annotations

import json
import math
import secrets
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Set, Tuple

import numpy as np
from cryptography.hazmat.primitives.asymmetric import ed25519

from fedthreat.secagg import keys as K
from fedthreat.secagg.masking import MaskGenerator
from fedthreat.secagg.quantize import Quantizer
from fedthreat.secagg.shamir import ShamirSecretSharing

Update = Dict[str, np.ndarray]
RingUpdate = Dict[str, np.ndarray]  # uint32 tensors


class ProtocolError(Exception):
    """Raised when a peer or the server violates the protocol."""


def default_threshold(n: int) -> int:
    """Shamir threshold: > n/2 so a minority cannot reconstruct, <= n so dropouts are tolerated."""
    return max(2, math.ceil(2 * n / 3))


def _pair_ctx(kind: bytes, round_id: bytes, a: int, b: int) -> bytes:
    lo, hi = sorted((a, b))
    return b"fedthreat/secagg/" + kind + b"|" + round_id + lo.to_bytes(4, "big") + hi.to_bytes(4, "big")


# ----------------------------------------------------------------------------- messages
@dataclass(frozen=True)
class Advertisement:
    client_id: int
    c_pub: bytes
    s_pub: bytes
    signature: bytes

    @staticmethod
    def signed_bytes(round_id: bytes, client_id: int, c_pub: bytes, s_pub: bytes) -> bytes:
        return b"fedthreat/secagg/adv|" + round_id + client_id.to_bytes(4, "big") + c_pub + s_pub


@dataclass
class Reveal:
    client_id: int
    b_shares: Dict[int, Tuple[int, int]] = field(default_factory=dict)  # survivor u -> (x, y)
    s_shares: Dict[int, Tuple[int, int]] = field(default_factory=dict)  # dropped d  -> (x, y)


# ----------------------------------------------------------------------------- client
class SecAggClient:
    """Organization-side participant. Holds all private key material for one round."""

    def __init__(
        self,
        client_id: int,
        identity_key: ed25519.Ed25519PrivateKey,
        round_id: bytes,
        quantizer: Quantizer,
    ):
        self.client_id = client_id
        self.identity_key = identity_key
        self.round_id = round_id
        self.quantizer = quantizer
        self.c_keys = K.KeyPair()  # encrypts share messages
        self.s_keys = K.KeyPair()  # derives pairwise mask seeds
        self.self_seed = secrets.token_bytes(32)  # b_i
        self.members: List[int] = []
        self.threshold = 0
        self._ads: Dict[int, Advertisement] = {}
        self.held: Dict[int, Tuple[int, int, int]] = {}  # sender -> (x, b_y, s_y)
        self.active: Set[int] = set()  # clients whose shares arrived (U2), incl. self
        self._revealed: Optional[Tuple[frozenset, frozenset]] = None

    # stage 0
    def advertise(self) -> Advertisement:
        msg = Advertisement.signed_bytes(
            self.round_id, self.client_id, self.c_keys.public_bytes, self.s_keys.public_bytes
        )
        return Advertisement(
            self.client_id, self.c_keys.public_bytes, self.s_keys.public_bytes, K.sign(self.identity_key, msg)
        )

    def receive_advertisements(
        self, ads: Mapping[int, Advertisement], identity_pubs: Mapping[int, ed25519.Ed25519PublicKey]
    ) -> None:
        for cid, ad in ads.items():
            if ad.client_id != cid or cid not in identity_pubs:
                raise ProtocolError(f"advertisement for unknown client {cid}")
            msg = Advertisement.signed_bytes(self.round_id, cid, ad.c_pub, ad.s_pub)
            if not K.verify(identity_pubs[cid], ad.signature, msg):
                raise ProtocolError(f"bad advertisement signature from client {cid}")
        if self.client_id not in ads:
            raise ProtocolError("own advertisement missing from cohort list")
        if ads[self.client_id].s_pub != self.s_keys.public_bytes:
            raise ProtocolError("server substituted our advertisement")
        self._ads = dict(ads)
        self.members = sorted(ads)
        self.threshold = default_threshold(len(self.members))
        if len(self.members) < self.threshold:
            raise ProtocolError("cohort too small")

    # stage 1
    def make_share_messages(self) -> Dict[int, bytes]:
        xs = [m + 1 for m in self.members]
        b_shares = ShamirSecretSharing.split_secret(self.self_seed, self.threshold, len(xs), xs)
        s_shares = ShamirSecretSharing.split_secret(self.s_keys.private_bytes, self.threshold, len(xs), xs)
        out: Dict[int, bytes] = {}
        for (x, b_y), (_, s_y), peer in zip(b_shares, s_shares, self.members):
            if peer == self.client_id:
                self.held[self.client_id] = (x, b_y, s_y)
                continue
            plaintext = json.dumps({"from": self.client_id, "to": peer, "x": x, "b": hex(b_y), "s": hex(s_y)}).encode()
            key = self.c_keys.agree(
                K.KeyPair.from_public_bytes(self._ads[peer].c_pub),
                _pair_ctx(b"enc", self.round_id, self.client_id, peer),
            )
            aad = self.round_id + self.client_id.to_bytes(4, "big") + peer.to_bytes(4, "big")
            out[peer] = K.aead_encrypt(key, plaintext, aad)
        return out

    def receive_share_messages(self, msgs: Mapping[int, bytes]) -> None:
        for sender, blob in msgs.items():
            if sender not in self._ads or sender == self.client_id:
                raise ProtocolError(f"share from unexpected client {sender}")
            key = self.c_keys.agree(
                K.KeyPair.from_public_bytes(self._ads[sender].c_pub),
                _pair_ctx(b"enc", self.round_id, sender, self.client_id),
            )
            aad = self.round_id + sender.to_bytes(4, "big") + self.client_id.to_bytes(4, "big")
            try:
                data = json.loads(K.aead_decrypt(key, blob, aad))
            except Exception as exc:  # InvalidTag, JSON error
                raise ProtocolError(f"undecryptable share from {sender}") from exc
            if data["from"] != sender or data["to"] != self.client_id or data["x"] != self.client_id + 1:
                raise ProtocolError(f"malformed share from {sender}")
            self.held[sender] = (data["x"], int(data["b"], 16), int(data["s"], 16))
        self.active = set(self.held)  # senders + ourselves
        if len(self.active) < self.threshold:
            raise ProtocolError("too few clients completed the share stage")

    # stage 2
    def masked_input(self, update: Update) -> RingUpdate:
        if not self.active:
            raise ProtocolError("share stage not completed")
        out: RingUpdate = {}
        for name, arr in update.items():
            y = self.quantizer.encode(arr).astype(np.uint32)
            y = y + MaskGenerator.expand_mask(
                self.self_seed, y.shape, label=f"self:{self.client_id}:{name}", round_id=self.round_id
            )
            for peer in sorted(self.active - {self.client_id}):
                seed = self.s_keys.agree(
                    K.KeyPair.from_public_bytes(self._ads[peer].s_pub),
                    _pair_ctx(b"pair", self.round_id, self.client_id, peer),
                )
                lo, hi = sorted((self.client_id, peer))
                m = MaskGenerator.expand_mask(seed, y.shape, label=f"pair:{lo}:{hi}:{name}", round_id=self.round_id)
                y = y + m if self.client_id < peer else y - m
            out[name] = y.astype(np.uint32)
        return out

    # stage 3
    def reveal(self, survivors: Iterable[int], dropped: Iterable[int]) -> Reveal:
        surv, drop = frozenset(survivors), frozenset(dropped)
        if surv & drop:
            raise ProtocolError("a client cannot be both survivor and dropout")
        if not (surv | drop) <= self.active or self.client_id not in surv:
            raise ProtocolError("inconsistent survivor/dropout sets")
        if len(surv) < self.threshold:
            raise ProtocolError("refusing to reveal: fewer survivors than threshold")
        if self._revealed is not None and self._revealed != (surv, drop):
            raise ProtocolError("refusing second, different reveal request")  # blocks b+s extraction
        self._revealed = (surv, drop)
        rev = Reveal(self.client_id)
        for u in surv:
            x, b_y, _ = self.held[u]
            rev.b_shares[u] = (x, b_y)
        for d in drop:
            x, _, s_y = self.held[d]
            rev.s_shares[d] = (x, s_y)
        return rev


# ----------------------------------------------------------------------------- server
class SecAggServer:
    """Server-side state for ONE cohort. Never receives plaintext updates."""

    def __init__(
        self,
        round_id: bytes,
        members: Iterable[int],
        identity_pubs: Mapping[int, ed25519.Ed25519PublicKey],
        quantizer: Quantizer,
        shapes: Mapping[str, Tuple[int, ...]],
    ):
        self.round_id = round_id
        self.members = sorted(members)
        self.threshold = default_threshold(len(self.members))
        self.identity_pubs = identity_pubs
        self.quantizer = quantizer
        self.shapes = {k: tuple(v) for k, v in shapes.items()}
        self.ads: Dict[int, Advertisement] = {}
        self.share_msgs: Dict[int, Dict[int, bytes]] = {}  # sender -> {recipient: blob}
        self.masked: Dict[int, RingUpdate] = {}
        self.reveals: Dict[int, Reveal] = {}

    def add_advertisement(self, ad: Advertisement) -> None:
        if ad.client_id not in self.members or ad.client_id in self.ads:
            raise ProtocolError("unexpected or duplicate advertisement")
        msg = Advertisement.signed_bytes(self.round_id, ad.client_id, ad.c_pub, ad.s_pub)
        if not K.verify(self.identity_pubs[ad.client_id], ad.signature, msg):
            raise ProtocolError("bad advertisement signature")
        if len(ad.c_pub) != 32 or len(ad.s_pub) != 32:
            raise ProtocolError("bad public key length")
        self.ads[ad.client_id] = ad

    def add_shares(self, sender: int, msgs: Mapping[int, bytes]) -> None:
        if sender not in self.ads or sender in self.share_msgs:
            raise ProtocolError("unexpected or duplicate share upload")
        if set(msgs) != set(self.ads) - {sender}:
            raise ProtocolError("share upload must address exactly the other advertisers")
        self.share_msgs[sender] = dict(msgs)

    def shares_for(self, recipient: int) -> Dict[int, bytes]:
        return {s: m[recipient] for s, m in self.share_msgs.items() if recipient in m}

    @property
    def active(self) -> Set[int]:  # U2
        return set(self.share_msgs)

    def add_masked(self, client_id: int, tensors: RingUpdate) -> None:
        if client_id not in self.active or client_id in self.masked:
            raise ProtocolError("unexpected or duplicate masked input")
        if set(tensors) != set(self.shapes):
            raise ProtocolError("masked input tensor names do not match model")
        for k, v in tensors.items():
            if v.dtype != np.uint32 or tuple(v.shape) != self.shapes[k]:
                raise ProtocolError(f"bad dtype/shape for {k}")
        self.masked[client_id] = tensors

    @property
    def survivors(self) -> List[int]:  # U3
        return sorted(self.masked)

    @property
    def dropped(self) -> List[int]:
        return sorted(self.active - set(self.masked))

    def add_reveal(self, rev: Reveal) -> None:
        if rev.client_id not in self.masked or rev.client_id in self.reveals:
            raise ProtocolError("unexpected or duplicate reveal")
        if set(rev.b_shares) != set(self.survivors) or set(rev.s_shares) != set(self.dropped):
            raise ProtocolError("reveal does not match survivor/dropout sets")
        self.reveals[rev.client_id] = rev

    def finalize(self) -> Dict[str, np.ndarray]:
        """Returns the exact float32 SUM of survivors' updates (never an individual one)."""
        surv, drop = self.survivors, self.dropped
        t = self.threshold
        if len(surv) < t or len(self.reveals) < t:
            raise ProtocolError("not enough survivors/reveals to unmask")
        revealers = sorted(self.reveals)[:t]
        total = {k: np.zeros(shape, dtype=np.uint32) for k, shape in self.shapes.items()}
        for cid in surv:
            for k in total:
                total[k] = total[k] + self.masked[cid][k]
        # 1) strip survivors' self-masks
        for u in surv:
            shares = [self.reveals[r].b_shares[u] for r in revealers]
            b_u = ShamirSecretSharing.reconstruct_secret(shares)
            for k in total:
                total[k] = total[k] - MaskGenerator.expand_mask(
                    b_u, self.shapes[k], label=f"self:{u}:{k}", round_id=self.round_id
                )
        # 2) strip pairwise masks that survivors applied against dropped clients
        for d in drop:
            shares = [self.reveals[r].s_shares[d] for r in revealers]
            s_priv = ShamirSecretSharing.reconstruct_secret(shares)
            kp = K.KeyPair.from_private_bytes(s_priv)
            if kp.public_bytes != self.ads[d].s_pub:
                raise ProtocolError(f"reconstructed key for dropped client {d} fails integrity check")
            for v in surv:
                seed = kp.agree(K.KeyPair.from_public_bytes(self.ads[v].s_pub), _pair_ctx(b"pair", self.round_id, d, v))
                lo, hi = sorted((d, v))
                for k in total:
                    m = MaskGenerator.expand_mask(seed, self.shapes[k], label=f"pair:{lo}:{hi}:{k}", round_id=self.round_id)
                    # survivor v added +m if v < d else -m ; remove exactly that
                    total[k] = total[k] - m if v < d else total[k] + m
        return {k: self.quantizer.decode_sum(v) for k, v in total.items()}


# ----------------------------------------------------------------------------- in-process driver
def simulate_secure_sum(
    updates: Mapping[int, Update],
    quantizer: Optional[Quantizer] = None,
    drop_after_masking: Iterable[int] = (),
    drop_before_sharing: Iterable[int] = (),
) -> Tuple[Dict[str, np.ndarray], List[int]]:
    """Runs the complete protocol in ONE process (simulation / tests only).

    This is NOT a security boundary: the caller holds every plaintext update. It exists to
    exercise the real protocol code, e.g. inside the simulation harness and unit tests.
    Returns (sum of survivors' updates, survivor ids).
    """
    ids = sorted(updates)
    q = quantizer or Quantizer(max_terms=max(8, len(ids)))
    round_id = secrets.token_bytes(16)
    idents = {i: ed25519.Ed25519PrivateKey.generate() for i in ids}
    pubs = {i: k.public_key() for i, k in idents.items()}
    shapes = {k: v.shape for k, v in next(iter(updates.values())).items()}
    clients = {i: SecAggClient(i, idents[i], round_id, q) for i in ids}
    server = SecAggServer(round_id, ids, pubs, q, shapes)
    for c in clients.values():
        server.add_advertisement(c.advertise())
    for c in clients.values():
        c.receive_advertisements(server.ads, pubs)
    sharers = [i for i in ids if i not in set(drop_before_sharing)]
    for i in sharers:
        server.add_shares(i, clients[i].make_share_messages())
    for i in sharers:
        clients[i].receive_share_messages(server.shares_for(i))
    for i in sharers:
        if i in set(drop_after_masking):
            continue
        server.add_masked(i, clients[i].masked_input(updates[i]))
    for i in server.survivors:
        server.add_reveal(clients[i].reveal(server.survivors, server.dropped))
    return server.finalize(), server.survivors
