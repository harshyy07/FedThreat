"""Create a development PKI + org identity keys + enrollment registry.

Layout (everything under --out, which is git-ignored):
  private_ca/ca.key           CA private key. Keep OFFLINE. Never mount into any container.
  ca.crt                      CA certificate (public)
  server/                     server.crt server.key ca.crt enrolled_orgs.json
  org<i>/                     client.crt client.key identity.key ca.crt enrolled_orgs.json
enrolled_orgs.json maps org id -> hex Ed25519 identity public key; the server derives client identity
from request signatures checked against it (see server/app.py). In production, issue certificates
from your real CA / HSM and distribute the registry through the consortium's governance process.
"""

import argparse
import datetime as dt
import ipaddress
import json
import os
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, ed25519
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID


def _write(path: Path, data: bytes, secret: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600 if secret else 0o644)
    with os.fdopen(fd, "wb") as f:
        f.write(data)


def _pem_key(k) -> bytes:
    return k.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())


def _cert(subject, issuer, pub, signer, san=None, ca=False, eku=None, days=365):
    now = dt.datetime.now(dt.timezone.utc)
    b = (x509.CertificateBuilder().subject_name(subject).issuer_name(issuer).public_key(pub)
         .serial_number(x509.random_serial_number()).not_valid_before(now - dt.timedelta(minutes=5))
         .not_valid_after(now + dt.timedelta(days=days))
         .add_extension(x509.BasicConstraints(ca=ca, path_length=0 if ca else None), critical=True))
    if san:
        b = b.add_extension(x509.SubjectAlternativeName(san), critical=False)
    if eku:
        b = b.add_extension(x509.ExtendedKeyUsage([eku]), critical=False)
    return b.sign(signer, hashes.SHA256())


def generate(out: Path, num_orgs: int = 9) -> Path:
    name = lambda cn: x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, cn)])  # noqa: E731
    ca_key = ec.generate_private_key(ec.SECP256R1())
    ca = _cert(name("fedthreat-ca"), name("fedthreat-ca"), ca_key.public_key(), ca_key, ca=True, days=730)
    ca_pem = ca.public_bytes(serialization.Encoding.PEM)
    _write(out / "private_ca" / "ca.key", _pem_key(ca_key), secret=True)
    _write(out / "ca.crt", ca_pem)

    registry = {}
    idents = {}
    for i in range(num_orgs):
        k = ed25519.Ed25519PrivateKey.generate()
        idents[i] = k
        registry[str(i)] = k.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw).hex()
    reg_bytes = json.dumps(registry, indent=2).encode()

    skey = ec.generate_private_key(ec.SECP256R1())
    scert = _cert(name("fedthreat-server"), ca.subject, skey.public_key(), ca_key,
                  san=[x509.DNSName("localhost"), x509.DNSName("server"), x509.IPAddress(ipaddress.ip_address("127.0.0.1"))],
                  eku=ExtendedKeyUsageOID.SERVER_AUTH)
    _write(out / "server" / "server.crt", scert.public_bytes(serialization.Encoding.PEM))
    _write(out / "server" / "server.key", _pem_key(skey), secret=True)
    _write(out / "server" / "ca.crt", ca_pem)
    _write(out / "server" / "enrolled_orgs.json", reg_bytes)

    for i in range(num_orgs):
        d = out / f"org{i}"
        ck = ec.generate_private_key(ec.SECP256R1())
        cc = _cert(name(f"org-{i}"), ca.subject, ck.public_key(), ca_key, eku=ExtendedKeyUsageOID.CLIENT_AUTH)
        _write(d / "client.crt", cc.public_bytes(serialization.Encoding.PEM))
        _write(d / "client.key", _pem_key(ck), secret=True)
        _write(d / "identity.key", _pem_key(idents[i]), secret=True)
        _write(d / "ca.crt", ca_pem)
        _write(d / "enrolled_orgs.json", reg_bytes)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="certs")
    ap.add_argument("--orgs", type=int, default=9)
    a = ap.parse_args()
    print("generated PKI in", generate(Path(a.out), a.orgs))
