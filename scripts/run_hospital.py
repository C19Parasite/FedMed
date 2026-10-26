"""CLI entry point to launch an isolated hospital silo client node."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import flwr as fl
import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
from fedmed.data.partition import create_or_load_partitions, get_hospital_loaders
from fedmed.utils.logging import get_logger
from fedmed.utils.seed import set_seed

from src.fl.client import FedMedClient
from src.model.unet import build_unet
from src.security.tls import load_client_tls_credentials

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Start FedMed Hospital Client Node.")
    parser.add_argument("--id", type=str, default="hospital_1", help="Hospital ID (e.g. hospital_1)")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--server-address", type=str, default=None, help="Server address override")
    parser.add_argument("--tls", action="store_true", help="Force TLS encryption")
    parser.add_argument("--he", action="store_true", help="Force Homomorphic Encryption")
    parser.add_argument("--dp", action="store_true", help="Force Differential Privacy")
    parser.add_argument("--dropout-round", type=int, default=None, help="Simulate crash in this round")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.tls:
        cfg.setdefault("tls", {})["enabled"] = True
    if args.he:
        cfg.setdefault("he", {})["enabled"] = True
    if args.dp:
        cfg.setdefault("dp", {})["enabled"] = True

    set_seed(int(cfg.get("seed", 42)))

    server_address = args.server_address or cfg.get("server", {}).get("address", "127.0.0.1:8080")

    # Load private hospital loaders
    _ = create_or_load_partitions(cfg)
    loaders = get_hospital_loaders(args.id, cfg)
    train_loader = loaders.train
    val_loader = loaders.val

    # Initialize model
    model = build_unet(cfg)
    device = "cuda" if torch.cuda.is_available() and cfg.get("training", {}).get("device") == "cuda" else "cpu"

    client = FedMedClient(
        hospital_id=args.id,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        device=device,
        dropout_round=args.dropout_round,
    )

    # TLS Credentials (Phase 4)
    root_certificates = None
    insecure = True
    if cfg.get("tls", {}).get("enabled", False):
        ca_path = cfg.get("tls", {}).get("ca_cert", "certs/ca.crt")
        try:
            root_certificates = load_client_tls_credentials(ca_path)
            insecure = False
            logger.info("[%s] Connecting via TLS with CA cert %s", args.id, ca_path)
        except Exception as e:
            logger.warning("[%s] Could not load CA cert (%s); falling back to insecure", args.id, e)

    logger.info("[%s] Connecting to Flower server at %s...", args.id, server_address)
    fl.client.start_client(
        server_address=server_address,
        client=client.to_client(),
        root_certificates=root_certificates,
        insecure=insecure,
    )


if __name__ == "__main__":
    main()
