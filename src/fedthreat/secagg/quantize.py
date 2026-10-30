"""Fixed-point encoding for secure aggregation over the ring Z_{2^32}.

Secure aggregation only hides values when masks are uniform over the SAME finite
group the data lives in. Floating-point masks do not achieve that, so updates are
quantized to integers and all masking arithmetic is done modulo 2^32.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

MODULUS_BITS = 32
MODULUS = 1 << MODULUS_BITS


class Quantizer:
    """Encodes float tensors as uint32 ring elements and decodes (signed) sums."""

    def __init__(self, scale: float = float(1 << 16), max_terms: int = 64):
        if scale <= 0 or max_terms < 1:
            raise ValueError("scale must be > 0 and max_terms >= 1")
        self.scale = float(scale)
        self.max_terms = int(max_terms)
        # A sum of max_terms values must stay inside the signed range of the ring.
        self.max_abs_int = (MODULUS // 2 - 1) // self.max_terms

    @property
    def max_abs_value(self) -> float:
        """Largest per-coordinate magnitude that can be encoded safely."""
        return self.max_abs_int / self.scale

    def encode(self, arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr, dtype=np.float64)
        if not np.all(np.isfinite(arr)):
            raise ValueError("update contains NaN/Inf")
        q = np.rint(arr * self.scale)
        if np.max(np.abs(q), initial=0.0) > self.max_abs_int:
            raise OverflowError(
                f"value exceeds encodable range +/-{self.max_abs_value:.3f}; "
                "clip the update (and noise) before masking"
            )
        return (q.astype(np.int64) % MODULUS).astype(np.uint32)

    def decode_sum(self, ring_sum: np.ndarray) -> np.ndarray:
        """Interprets ring elements as signed integers and rescales to float32."""
        signed = ring_sum.astype(np.int64)
        signed = np.where(signed >= MODULUS // 2, signed - MODULUS, signed)
        return (signed / self.scale).astype(np.float32)

    def encode_update(self, update: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        return {k: self.encode(v) for k, v in update.items()}
