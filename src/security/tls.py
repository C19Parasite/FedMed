"""TLS configuration and credential loading for Flower gRPC communication."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple
from fedmed.utils.logging import get_logger

logger = get_logger(__name__)


def load_server_tls_credentials(
    ca_path: str | Path, cert_path: str | Path, key_path: str | Path
) -> Tuple[bytes, bytes, bytes]:
    """Load CA certificate, server certificate, and server private key for Flower server.

    Flower expects a tuple of (ca_cert_bytes, server_cert_bytes, server_key_bytes).

    Args:
        ca_path: Path to CA root certificate.
        cert_path: Path to signed server certificate.
        key_path: Path to server private key.

    Returns:
        Tuple of (ca_cert, server_cert, server_key) as bytes.
    """
    ca_bytes = Path(ca_path).read_bytes()
    cert_bytes = Path(cert_path).read_bytes()
    key_bytes = Path(key_path).read_bytes()
    logger.info("Loaded server TLS credentials (CA=%s, Cert=%s)", ca_path, cert_path)
    return (ca_bytes, cert_bytes, key_bytes)


def load_client_tls_credentials(ca_path: str | Path) -> bytes:
    """Load CA root certificate for Flower client validation.

    Args:
        ca_path: Path to CA root certificate.

    Returns:
        CA certificate bytes.
    """
    ca_bytes = Path(ca_path).read_bytes()
    logger.info("Loaded client TLS CA root certificate from %s", ca_path)
    return ca_bytes


__all__ = ["load_server_tls_credentials", "load_client_tls_credentials"]
