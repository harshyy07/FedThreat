"""Shamir's Secret Sharing over a prime Galois field for dropout resilience."""

import secrets
from typing import List, Tuple

# 257-bit prime strictly > 2^256: 2^256 + 297 (guarantees any 32-byte secret is < PRIME)
PRIME = 0x10000000000000000000000000000000000000000000000000000000000000129


def _eval_poly(coeffs: List[int], x: int, prime: int = PRIME) -> int:
    """Evaluates polynomial at point x modulo prime."""
    result = 0
    for coeff in reversed(coeffs):
        result = (result * x + coeff) % prime
    return result


def _mod_inverse(k: int, prime: int = PRIME) -> int:
    """Computes modular multiplicative inverse using extended Euclidean / pow."""
    return pow(k, -1, prime)


class ShamirSecretSharing:
    """(t, n) threshold secret sharing scheme."""

    @staticmethod
    def split_secret(
        secret_bytes: bytes, threshold: int, num_shares: int, xs: List[int] | None = None
    ) -> List[Tuple[int, int]]:
        """Splits secret bytes into num_shares such that any threshold shares can reconstruct it."""
        if not 2 <= threshold <= num_shares:
            raise ValueError("need 2 <= threshold <= num_shares")
        if len(secret_bytes) != 32:
            raise ValueError("secret must be exactly 32 bytes")
        secret_int = int.from_bytes(secret_bytes, byteorder="big")  # always < PRIME

        # Random polynomial coefficients: a0 = secret, a1..a_{t-1} = random
        # coeffs[i] is coefficient of x^i
        coeffs = [secret_int] + [secrets.randbelow(PRIME) for _ in range(threshold - 1)]

        xs = xs if xs is not None else list(range(1, num_shares + 1))
        if len(xs) != num_shares or len(set(xs)) != num_shares or any(x <= 0 for x in xs):
            raise ValueError("xs must be num_shares distinct positive integers")
        shares = []
        for x in xs:
            y = _eval_poly(coeffs, x, PRIME)
            shares.append((x, y))

        return shares

    @staticmethod
    def reconstruct_secret(shares: List[Tuple[int, int]], expected_len: int = 32) -> bytes:
        """Reconstructs the secret using Lagrange interpolation modulo PRIME."""
        if len({s[0] for s in shares}) != len(shares):
            raise ValueError("duplicate share x-coordinates")
        if any(s[0] % PRIME == 0 for s in shares):
            raise ValueError("invalid share x-coordinate")
        x_s = [s[0] for s in shares]
        y_s = [s[1] for s in shares]
        k = len(shares)
        secret_int = 0

        for i in range(k):
            xi, yi = x_s[i], y_s[i]
            # Lagrange basis l_i(0) = product_{j != i} (-xj) / (xi - xj)
            num = 1
            den = 1
            for j in range(k):
                if i == j:
                    continue
                xj = x_s[j]
                num = (num * (-xj)) % PRIME
                den = (den * (xi - xj)) % PRIME

            lagrange_val = (num * _mod_inverse(den % PRIME, PRIME)) % PRIME
            secret_int = (secret_int + yi * lagrange_val) % PRIME

        if secret_int >= (1 << (8 * expected_len)):
            raise ValueError("reconstruction failed (wrong or insufficient shares)")
        return secret_int.to_bytes(expected_len, byteorder="big")
