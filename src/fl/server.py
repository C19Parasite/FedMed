"""Flower aggregation server launcher with TLS support."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Tuple
import flwr as fl

from fedmed.utils.logging import get_logger
from src.security.tls import load_server_tls_credentials

logger = get_logger(__name__)


def start_server_process(
    cfg: Mapping[str, Any],
    server_address: Optional[str] = None,
    num_rounds: Optional[int] = None,
    strategy: Optional[fl.server.strategy.Strategy] = None,
) -> fl.server.history.History:
    """Launch Flower aggregation server with optional TLS encryption.

    Args:
        cfg: Unified configuration dictionary.
        server_address: Optional host:port address (defaults to cfg.server.address).
        num_rounds: Optional round count override.
        strategy: Aggregation strategy instance.

    Returns:
        Training history object.
    """
    if server_address is None:
        server_address = cfg.get("server", {}).get("address", "127.0.0.1:8080")

    if num_rounds is None:
        if cfg.get("debug", False):
            num_rounds = int(cfg.get("server", {}).get("debug_rounds", 2))
        else:
            num_rounds = int(cfg.get("server", {}).get("rounds", 3))

    # TLS certificates (Phase 4)
    certificates = None
    if cfg.get("tls", {}).get("enabled", False):
        ca_path = cfg.get("tls", {}).get("ca_cert", "certs/ca.crt")
        cert_path = cfg.get("tls", {}).get("server_cert", "certs/server.crt")
        key_path = cfg.get("tls", {}).get("server_key", "certs/server.key")
        try:
            certificates = load_server_tls_credentials(ca_path, cert_path, key_path)
            logger.info("Server enabling TLS gRPC encryption with certs from %s", cert_path)
        except Exception as e:
            logger.error("Failed to load TLS certs (%s); starting plaintext", e)
            certificates = None

    logger.info("Starting FedMed FL Server on %s for %d rounds (TLS=%s)", server_address, num_rounds, bool(certificates))

    return fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        certificates=certificates,
    )


__all__ = ["start_server_process"]
