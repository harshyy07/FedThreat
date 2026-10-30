"""Key agreement, authenticated encryption and signatures used by secure aggregation."""

from __future__ import annotations

import os

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ed25519, x25519
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

_RAW = dict(encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw)
_RAW_PRIV = dict(
    encoding=serialization.Encoding.Raw,
    format=serialization.PrivateFormat.Raw,
    encryption_algorithm=serialization.NoEncryption(),
)


class KeyPair:
    """X25519 keypair."""

    def __init__(self, private_key: x25519.X25519PrivateKey | None = None):
        self.private_key = private_key or x25519.X25519PrivateKey.generate()
        self.public_key = self.private_key.public_key()

    @property
    def public_bytes(self) -> bytes:
        return self.public_key.public_bytes(**_RAW)

    @property
    def private_bytes(self) -> bytes:
        return self.private_key.private_bytes(**_RAW_PRIV)

    @classmethod
    def from_private_bytes(cls, raw: bytes) -> "KeyPair":
        return cls(x25519.X25519PrivateKey.from_private_bytes(raw))

    @classmethod
    def from_public_bytes(cls, raw: bytes) -> x25519.X25519PublicKey:
        return x25519.X25519PublicKey.from_public_bytes(raw)

    def agree(self, peer_public_key: x25519.X25519PublicKey, context: bytes) -> bytes:
        """X25519 ECDH + HKDF-SHA256 -> 32 bytes. `context` MUST bind round and role."""
        shared = self.private_key.exchange(peer_public_key)
        return HKDF(algorithm=hashes.SHA256(), length=32, salt=None, info=context).derive(shared)


def aead_encrypt(key: bytes, plaintext: bytes, aad: bytes) -> bytes:
    nonce = os.urandom(12)
    return nonce + AESGCM(key).encrypt(nonce, plaintext, aad)


def aead_decrypt(key: bytes, blob: bytes, aad: bytes) -> bytes:
    return AESGCM(key).decrypt(blob[:12], blob[12:], aad)


def sign(private_key: ed25519.Ed25519PrivateKey, message: bytes) -> bytes:
    return private_key.sign(message)


def verify(public_key: ed25519.Ed25519PublicKey, signature: bytes, message: bytes) -> bool:
    try:
        public_key.verify(signature, message)
        return True
    except InvalidSignature:
        return False


def ed25519_public_from_bytes(raw: bytes) -> ed25519.Ed25519PublicKey:
    return ed25519.Ed25519PublicKey.from_public_bytes(raw)


def ed25519_public_bytes(pub: ed25519.Ed25519PublicKey) -> bytes:
    return pub.public_bytes(**_RAW)
