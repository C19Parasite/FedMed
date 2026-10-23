"""CLI script to orchestrate Flower central server and 3 mock hospital client nodes."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import sys
import time
import flwr as fl
from monai.data import DataLoader

from fedmed.data.datasets import discover_cases
from fedmed.data.loaders import _make_dataset
from fedmed.data.partition import create_or_load_partitions, get_hospital_loaders
from fedmed.data.transforms import get_eval_transforms, get_train_transforms
from fedmed.fl.client import HospitalClient
from fedmed.fl.server import partition_cases_for_hospitals, start_fedmed_server
from fedmed.fl.strategy import FedMedFedAvg
from fedmed.models.unet3d import get_model
from fedmed.utils.config import load_config
from fedmed.utils.logging import get_logger
from fedmed.utils.seed import set_seed

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI options for federated node execution."""
    parser = argparse.ArgumentParser(
        description="Run FedMed Flower central server and mock hospital nodes."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/federated_hospitals.yaml",
        help="Path to federated YAML configuration file.",
    )
    parser.add_argument(
        "--role",
        type=str,
        default="orchestrate",
        choices=["server", "client", "orchestrate"],
        help="Role to run: 'server', 'client', or 'orchestrate' (launches server + 3 hospitals).",
    )
    parser.add_argument(
        "--hospital-id",
        type=str,
        default="hospital_1",
        help="Hospital identifier when role is 'client'.",
    )
    parser.add_argument(
        "--server-address",
        type=str,
        default=None,
        help="Flower gRPC server address (e.g. 127.0.0.1:8080).",
    )
    return parser.parse_args()


def build_hospital_client(
    cfg: dict,
    hospital_id: str,
    hospital_cases: list[dict] | None = None,
    manifest: dict | None = None,
) -> HospitalClient:
    """Construct a HospitalClient instance with its private data partition."""
    if manifest is not None or hospital_cases is None:
        loaders = get_hospital_loaders(hospital_id, cfg, manifest=manifest)
        train_loader = loaders.train
        val_loader = loaders.val
    else:
        # Fallback when explicit cases list is passed
        n_train = max(1, int(len(hospital_cases) * 0.75))
        train_cases = hospital_cases[:n_train]
        val_cases = hospital_cases[n_train:] or hospital_cases[:1]

        train_ds = _make_dataset(cfg, train_cases, get_train_transforms(cfg))
        val_ds = _make_dataset(cfg, val_cases, get_eval_transforms(cfg))

        batch_size = int(cfg["data"]["batch_size"])
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    model = get_model(cfg)
    return HospitalClient(
        hospital_id=hospital_id,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        device=cfg.get("training", {}).get("device", "cpu"),
    )


def run_orchestration(cfg: dict, server_address: str) -> None:
    """Launch server and 3 mock hospital client nodes concurrently in threads."""
    logger.info("Initializing multi-node orchestration for 3 mock hospitals")
    hospitals = cfg.get("hospitals", [])
    num_hospitals = len(hospitals)

    # Patient-level partitioning across hospitals with global validation hold-out
    manifest = create_or_load_partitions(cfg, print_table=True)

    clients = []
    for idx, hosp in enumerate(hospitals):
        hosp_id = hosp["id"]
        hosp_patients = manifest.get("hospitals", {}).get(hosp_id, {}).get("patient_ids", [])
        logger.info(
            "Configured hospital node [%s] (%s) on mock port %s with %s private patient cases",
            hosp_id,
            hosp["name"],
            hosp["port"],
            len(hosp_patients),
        )
        client = build_hospital_client(cfg, hosp_id, manifest=manifest)
        clients.append((hosp, client))

    # Run in thread pool: start server then start the 3 hospital clients
    def start_server_task():
        num_rounds = int(cfg.get("server", {}).get("num_rounds", 2))
        start_fedmed_server(
            server_address=server_address,
            num_rounds=num_rounds,
            min_clients=num_hospitals,
        )

    def start_client_task(client_obj: HospitalClient):
        fl.client.start_client(
            server_address=server_address,
            client=client_obj.to_client(),
        )

    with ThreadPoolExecutor(max_workers=num_hospitals + 1) as executor:
        server_future = executor.submit(start_server_task)
        time.sleep(2)  # Give server time to bind port

        client_futures = [executor.submit(start_client_task, client) for _, client in clients]
        # Wait for server to finish
        server_future.result()
        for f in client_futures:
            f.result()

    logger.info("Orchestration of 3 mock hospital nodes completed successfully.")


def main() -> None:
    """CLI entry point for running federated nodes."""
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]))

    server_addr = args.server_address or cfg.get("server", {}).get("address", "127.0.0.1:8080")

    if args.role == "server":
        num_rounds = int(cfg.get("server", {}).get("num_rounds", 3))
        start_fedmed_server(server_address=server_addr, num_rounds=num_rounds)

    elif args.role == "client":
        all_cases = discover_cases(cfg)
        client = build_hospital_client(cfg, args.hospital_id, all_cases)
        logger.info("Starting hospital client [%s] connecting to %s", args.hospital_id, server_addr)
        fl.client.start_client(server_address=server_addr, client=client.to_client())

    elif args.role == "orchestrate":
        run_orchestration(cfg, server_addr)


if __name__ == "__main__":
    main()
