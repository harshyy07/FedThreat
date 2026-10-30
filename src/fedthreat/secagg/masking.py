"""Cryptographic mask expansion (PRG) for secure aggregation.

Each mask is SHAKE-256(domain || seed || tensor-name) expanded to uniform uint32
ring elements. The FULL 32-byte seed is used, and every tensor / role / round is
domain-separated so two tensors never share a mask stream.
"""

from __future__ import annotations

import hashlib

import numpy as np


class MaskGenerator:
    @staticmethod
    def expand_mask(seed: bytes, shape: tuple[int, ...], *, label: str, round_id: bytes) -> np.ndarray:
        if len(seed) != 32:
            raise ValueError("mask seed must be exactly 32 bytes")
        n = int(np.prod(shape, dtype=np.int64)) if len(shape) else 1
        h = hashlib.shake_256()
        h.update(b"fedthreat/secagg/mask/v2\x00")
        h.update(len(round_id).to_bytes(2, "big") + round_id)
        lab = label.encode()
        h.update(len(lab).to_bytes(2, "big") + lab)
        h.update(seed)
        raw = h.digest(4 * n)
        return np.frombuffer(raw, dtype="<u4").astype(np.uint32).reshape(shape)
