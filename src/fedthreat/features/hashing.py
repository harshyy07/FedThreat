import hashlib
import numpy as np

try:
    import mmh3
    HAS_MMH3 = True
except ImportError:
    HAS_MMH3 = False


def hash_feature(val: str, num_bins: int, salt: str = "fedthreat_v1") -> np.ndarray:
    """Hashes a categorical or high-cardinality string into a 1-hot / count bin vector."""
    if HAS_MMH3:
        vec = np.zeros(num_bins, dtype=np.float32)
        if val is None or val == "":
            return vec
        key = f"{salt}:{val}"
        bin_idx = mmh3.hash(key) % num_bins
        vec[bin_idx] = 1.0
        return vec
    return simple_hash(val, num_bins, salt)


def simple_hash(val: str, num_bins: int, salt: str = "fedthreat_v1") -> np.ndarray:
    """Fallback hash using standard hashlib md5 when mmh3 is not installed."""
    import hashlib
    vec = np.zeros(num_bins, dtype=np.float32)
    if val is None or val == "":
        return vec
    key = f"{salt}:{val}".encode("utf-8")
    bin_idx = int(hashlib.md5(key).hexdigest(), 16) % num_bins
    vec[bin_idx] = 1.0
    return vec
