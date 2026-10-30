"""Mutual-TLS HTTP client construction (explicit SSLContext: behaves identically across httpx versions)."""

import ssl

import httpx


def mtls_client(base_url: str, ca_file: str, cert_file: str, key_file: str, timeout: float = 60.0) -> httpx.Client:
    ctx = ssl.create_default_context(cafile=ca_file)  # verifies the server against OUR consortium CA only
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    ctx.load_cert_chain(cert_file, key_file)
    return httpx.Client(base_url=base_url, verify=ctx, timeout=timeout)
