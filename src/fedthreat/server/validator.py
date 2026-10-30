"""Server-side parameter update verification and validation.

Prevents malicious updates containing NaN/Inf, oversized coordinate norms,
mismatched shapes, or corrupted feature spec hashes from entering aggregation.
"""

from typing import Dict, Tuple, List, Optional
import numpy as np


class UpdateValidationError(ValueError):
    """Raised when a client parameter update violates security constraints."""
    pass


class UpdateValidator:
    """Enforces strict server-side bounds and structural sanity checks on client updates."""

    def __init__(
        self,
        max_l2_norm: float = 10.0,
        expected_spec_hash: Optional[str] = None,
    ):
        self.max_l2_norm = max_l2_norm
        self.expected_spec_hash = expected_spec_hash

    def validate_spec_hash(self, client_spec_hash: str) -> None:
        """Ensures client extracted features using the identical consortium feature spec."""
        if self.expected_spec_hash is not None and client_spec_hash != self.expected_spec_hash:
            raise UpdateValidationError(
                f"Feature spec mismatch: client hash '{client_spec_hash}' != server hash '{self.expected_spec_hash}'"
            )

    def validate_update(
        self,
        delta: Dict[str, np.ndarray],
        expected_shapes: Optional[Dict[str, Tuple[int, ...]]] = None,
    ) -> float:
        """Validates that a delta has valid float values, matches expected model architecture,
        and does not exceed the server-side maximum allowed L2 norm.
        
        Returns:
            The computed L2 norm of the update.
        """
        if not delta:
            raise UpdateValidationError("Update dictionary is empty")

        total_sq_norm = 0.0

        for name, arr in delta.items():
            # 1. Check for NaN or Inf (denial-of-service / poisoning check)
            if not np.all(np.isfinite(arr)):
                raise UpdateValidationError(f"Tensor '{name}' contains NaN or Inf values")

            # 2. Check shape consistency with model architecture
            if expected_shapes and name in expected_shapes:
                if arr.shape != expected_shapes[name]:
                    raise UpdateValidationError(
                        f"Tensor '{name}' shape mismatch: got {arr.shape}, expected {expected_shapes[name]}"
                    )

            # 3. Accumulate L2 norm
            total_sq_norm += float(np.sum(np.square(arr)))

        l2_norm = float(np.sqrt(total_sq_norm))

        # 4. Enforce strict norm bound
        if self.max_l2_norm > 0 and l2_norm > self.max_l2_norm:
            raise UpdateValidationError(
                f"Update L2 norm {l2_norm:.2f} exceeds server maximum allowed bound {self.max_l2_norm:.2f}"
            )

        return l2_norm
