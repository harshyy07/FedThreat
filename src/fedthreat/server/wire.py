"""JSON wire encoding for secure-aggregation messages and signed request envelopes."""

from __future__ import annotations

import base64
import json
from typing import Any, Dict, Mapping, Tuple

import numpy as np

from fedthreat.secagg.protocol import Advertisement, Reveal


def b64e(b: bytes) -> str:
    return base64.b64encode(b).decode("ascii")


def b64d(s: str) -> bytes:
    return base64.b64decode(s.encode("ascii"), validate=True)


def enc_array(a: np.ndarray, dtype: str) -> Dict[str, Any]:
    return {"shape": list(a.shape), "data": b64e(np.ascontiguousarray(a, dtype=dtype).tobytes())}


def dec_array(d: Mapping[str, Any], dtype: str, expect_shape: Tuple[int, ...] | None = None) -> np.ndarray:
    arr = np.frombuffer(b64d(d["data"]), dtype=dtype)
    shape = tuple(int(x) for x in d["shape"])
    if int(np.prod(shape, dtype=np.int64)) != arr.size:
        raise ValueError("array size does not match shape")
    if expect_shape is not None and shape != tuple(expect_shape):
        raise ValueError("unexpected tensor shape")
    return arr.reshape(shape).copy()


def enc_ad(ad: Advertisement) -> Dict[str, Any]:
    return {"client_id": ad.client_id, "c_pub": b64e(ad.c_pub), "s_pub": b64e(ad.s_pub), "sig": b64e(ad.signature)}


def dec_ad(d: Mapping[str, Any]) -> Advertisement:
    return Advertisement(int(d["client_id"]), b64d(d["c_pub"]), b64d(d["s_pub"]), b64d(d["sig"]))


def enc_reveal(r: Reveal) -> Dict[str, Any]:
    f = lambda m: {str(k): [x, hex(y)] for k, (x, y) in m.items()}  # noqa: E731
    return {"b_shares": f(r.b_shares), "s_shares": f(r.s_shares)}


def dec_reveal(client_id: int, d: Mapping[str, Any]) -> Reveal:
    g = lambda m: {int(k): (int(v[0]), int(v[1], 16)) for k, v in m.items()}  # noqa: E731
    return Reveal(client_id, g(d["b_shares"]), g(d["s_shares"]))


def canonical(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def signing_input(path: str, client_id: int, ts_ms: int, nonce: str, payload: Any) -> bytes:
    return b"fedthreat/req/v1|" + path.encode() + b"|" + str(client_id).encode() + b"|" + str(ts_ms).encode() \
        + b"|" + nonce.encode() + b"|" + canonical(payload)
