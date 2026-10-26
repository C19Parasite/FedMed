"""CLI entry point to launch the FedMed central Flower server."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
from fedmed.data.loaders import get_loaders
from fedmed.data.partition import create_or_load_partitions, get_global_eval_loaders
from fedmed.models.params import get_parameters
from fedmed.utils.logging import get_logger
from fedmed.utils.seed import set_seed
from flwr.common import ndarrays_to_parameters

from src.fl.server import start_server_process
from src.fl.strategy import FedMedStrategy
from src.model.unet import build_unet

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Start FedMed Central Flower Server.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--rounds", type=int, default=None, help="Override number of training rounds")
    parser.add_argument("--tls", action="store_true", help="Force TLS encryption")
    parser.add_argument("--he", action="store_true", help="Force Homomorphic Encryption")
    parser.add_argument("--dp", action="store_true", help="Force Differential Privacy")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.rounds is not None:
        cfg["server"]["rounds"] = args.rounds
        cfg["server"]["debug_rounds"] = args.rounds
    if args.tls:
        cfg.setdefault("tls", {})["enabled"] = True
    if args.he:
        cfg.setdefault("he", {})["enabled"] = True
    if args.dp:
        cfg.setdefault("dp", {})["enabled"] = True

    set_seed(int(cfg.get("seed", 42)))

    # Ensure partitions exist and get server global validation loader
    _ = create_or_load_partitions(cfg)
    global_loaders = get_global_eval_loaders(cfg)
    global_val_loader = global_loaders["val"]

    # Initialize model
    model = build_unet(cfg)
    device = "cuda" if torch.cuda.is_available() and cfg.get("training", {}).get("device") == "cuda" else "cpu"
    model.to(device)

    # Server provides identical initial weights to all clients
    initial_weights = get_parameters(model)
    initial_parameters = ndarrays_to_parameters(initial_weights)

    strategy = FedMedStrategy(
        model=model,
        global_val_loader=global_val_loader,
        cfg=cfg,
        device=device,
        initial_parameters=initial_parameters,
        min_fit_clients=int(cfg.get("server", {}).get("min_fit_clients", 2)),
        min_available_clients=int(cfg.get("server", {}).get("min_available_clients", 3)),
        min_evaluate_clients=int(cfg.get("server", {}).get("min_fit_clients", 2)),
    )

    start_server_process(cfg, strategy=strategy)


if __name__ == "__main__":
    main()
