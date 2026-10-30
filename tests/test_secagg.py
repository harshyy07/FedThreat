import os
import secrets

import numpy as np
import pytest
from cryptography.hazmat.primitives.asymmetric import ed25519

from fedthreat.secagg import keys as K
from fedthreat.secagg.masking import MaskGenerator
from fedthreat.secagg.protocol import (
    ProtocolError, SecAggClient, SecAggServer, simulate_secure_sum,
)
from fedthreat.secagg.quantize import Quantizer
from fedthreat.secagg.shamir import PRIME, ShamirSecretSharing


def _updates(n, d=50, seed=0):
    rng = np.random.default_rng(seed)
    return {i: {"w": rng.normal(0, 0.3, d).astype(np.float32), "b": rng.normal(0, 0.3, (3, 4)).astype(np.float32)}
            for i in range(n)}


def _expected(updates, ids):
    return {k: sum(updates[i][k] for i in ids) for k in next(iter(updates.values()))}


def test_keypair_ecdh_agreement():
    a, b = K.KeyPair(), K.KeyPair()
    ctx = b"ctx"
    assert a.agree(b.public_key, ctx) == b.agree(a.public_key, ctx)
    assert a.agree(b.public_key, ctx) != a.agree(b.public_key, b"other")


def test_shamir_recovery_and_validation():
    secret = os.urandom(32)
    shares = ShamirSecretSharing.split_secret(secret, 3, 5)
    assert ShamirSecretSharing.reconstruct_secret([shares[0], shares[2], shares[4]]) == secret
    assert ShamirSecretSharing.reconstruct_secret(shares[:3]) == secret
    with pytest.raises(ValueError):
        ShamirSecretSharing.split_secret(b"short", 3, 5)
    with pytest.raises(ValueError):
        ShamirSecretSharing.reconstruct_secret([shares[0], shares[0]])


def test_shamir_prime_is_expected_constant():
    assert PRIME == 2**256 + 297


def test_exact_sum_no_dropout():
    ups = _updates(5)
    out, surv = simulate_secure_sum(ups)
    assert surv == list(range(5))
    for k, v in _expected(ups, surv).items():
        np.testing.assert_allclose(out[k], v, atol=2e-4)  # quantization step 1/65536 per term


def test_dropout_after_masking_is_recovered():
    """Regression: old implementation returned garbage when a client dropped."""
    ups = _updates(6)
    out, surv = simulate_secure_sum(ups, drop_after_masking=[3, 5])
    assert surv == [0, 1, 2, 4]
    for k, v in _expected(ups, surv).items():
        np.testing.assert_allclose(out[k], v, atol=2e-4)


def test_dropout_before_sharing_is_handled():
    ups = _updates(6)
    out, surv = simulate_secure_sum(ups, drop_before_sharing=[1])
    assert 1 not in surv
    for k, v in _expected(ups, surv).items():
        np.testing.assert_allclose(out[k], v, atol=2e-4)


def test_too_many_dropouts_fails_loudly():
    ups = _updates(6)  # threshold = 4
    with pytest.raises(ProtocolError):
        simulate_secure_sum(ups, drop_after_masking=[0, 1, 2])


def _setup_cohort(n=4, d=20):
    q = Quantizer(max_terms=8)
    rid = secrets.token_bytes(16)
    idents = {i: ed25519.Ed25519PrivateKey.generate() for i in range(n)}
    pubs = {i: k.public_key() for i, k in idents.items()}
    clients = {i: SecAggClient(i, idents[i], rid, q) for i in range(n)}
    server = SecAggServer(rid, range(n), pubs, q, {"w": (d,)})
    for c in clients.values():
        server.add_advertisement(c.advertise())
    for c in clients.values():
        c.receive_advertisements(server.ads, pubs)
    for i, c in clients.items():
        server.add_shares(i, c.make_share_messages())
    for i, c in clients.items():
        c.receive_share_messages(server.shares_for(i))
    return q, rid, idents, pubs, clients, server


def test_server_view_of_one_client_is_not_its_plaintext():
    """Server holds masked vector + ciphertexts. Without >= t shares it cannot strip masks."""
    q, rid, _, _, clients, server = _setup_cohort()
    x = np.full(20, 0.25, dtype=np.float32)
    masked = clients[0].masked_input({"w": x})["w"]
    plain_ring = q.encode(x)
    assert not np.array_equal(masked, plain_ring)
    # masked values are ~uniform over the ring: mean near 2^31, not near the plaintext
    big = clients[0].masked_input({"w": np.zeros(5000, np.float32)})["w"].astype(np.float64)
    assert abs(big.mean() / 2**32 - 0.5) < 0.03


def test_client_refuses_to_reveal_both_secrets_for_one_peer():
    _, _, _, _, clients, server = _setup_cohort()
    c = clients[0]
    c.reveal({0, 1, 2}, {3})
    with pytest.raises(ProtocolError):
        c.reveal({0, 1, 2, 3}, set())      # server now claims 3 survived -> would leak b_3 after s_3
    with pytest.raises(ProtocolError):
        clients[1].reveal({0, 1, 2}, {2})  # overlap


def test_client_refuses_reveal_below_threshold():
    _, _, _, _, clients, _ = _setup_cohort()
    with pytest.raises(ProtocolError):
        clients[0].reveal({0, 1}, {2, 3})  # threshold for n=4 is 3


def test_forged_advertisement_rejected():
    q = Quantizer()
    rid = secrets.token_bytes(16)
    idents = {i: ed25519.Ed25519PrivateKey.generate() for i in range(3)}
    pubs = {i: k.public_key() for i, k in idents.items()}
    server = SecAggServer(rid, range(3), pubs, q, {"w": (4,)})
    evil = SecAggClient(1, ed25519.Ed25519PrivateKey.generate(), rid, q)  # not org 1's identity key
    with pytest.raises(ProtocolError):
        server.add_advertisement(evil.advertise())


def test_server_substituting_a_key_is_detected_by_clients():
    q, rid, idents, pubs, clients, server = _setup_cohort()
    fresh = {i: SecAggClient(i, idents[i], rid, q) for i in range(4)}
    ads = {i: c.advertise() for i, c in fresh.items()}
    swapped = dict(ads)
    other = SecAggClient(2, idents[2], rid, q).advertise()  # validly signed but different keys
    swapped[2] = other
    with pytest.raises(ProtocolError):
        fresh[2].receive_advertisements(swapped, pubs)


def test_tampered_share_ciphertext_rejected():
    q, rid, idents, pubs, _, _ = _setup_cohort()
    cl = {i: SecAggClient(i, idents[i], rid, q) for i in range(4)}
    ads = {i: c.advertise() for i, c in cl.items()}
    for c in cl.values():
        c.receive_advertisements(ads, pubs)
    msgs = cl[0].make_share_messages()
    blob = bytearray(msgs[1]); blob[-1] ^= 1
    with pytest.raises(ProtocolError):
        cl[1].receive_share_messages({0: bytes(blob)})


def test_masks_domain_separated_per_tensor_and_full_seed_used():
    s, rid = os.urandom(32), b"r" * 16
    a = MaskGenerator.expand_mask(s, (8,), label="t1", round_id=rid)
    b = MaskGenerator.expand_mask(s, (8,), label="t2", round_id=rid)
    assert not np.array_equal(a, b)  # same-shape tensors no longer share a mask (regression)
    s2 = s[:8] + bytes(24)
    s3 = s[:8] + b"\xff" * 24
    assert not np.array_equal(MaskGenerator.expand_mask(s2, (8,), label="t", round_id=rid),
                              MaskGenerator.expand_mask(s3, (8,), label="t", round_id=rid))  # no 8-byte truncation
    c = MaskGenerator.expand_mask(s, (8,), label="t1", round_id=b"x" * 16)
    assert not np.array_equal(a, c)  # round separation


def test_quantizer_overflow_is_explicit_not_silent():
    q = Quantizer(max_terms=4)
    with pytest.raises(OverflowError):
        q.encode(np.array([q.max_abs_value * 2]))
    with pytest.raises(ValueError):
        q.encode(np.array([np.nan]))
