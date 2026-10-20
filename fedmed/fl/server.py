"""Flower central server and multi-hospital simulation coordinator."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Optional
import flwr as fl
import torch

from fedmed.fl.client import HospitalClient
from fedmed.fl.strategy import FedMedFedAvg
from fedmed.models.unet3d import get_model
from fedmed.utils.logging import get_logger

logger = get_logger(__name__)


def start_fedmed_server(
    server_address: str = "127.0.0.1:8080",
    num_rounds: int = 3,
    min_clients: int = 3,
    strategy: Optional[fl.server.strategy.Strategy] = None,
) -> fl.server.history.History:
    """Start standalone Flower aggregation server.

    Args:
        server_address: Host and port to listen on.
        num_rounds: Number of federated training rounds.
        min_clients: Minimum required hospital clients.
        strategy: Aggregation strategy (defaults to FedMedFedAvg).

    Returns:
        Federated training history.
    """
    if strategy is None:
        strategy = FedMedFedAvg(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=min_clients,
            min_evaluate_clients=min_clients,
            min_available_clients=min_clients,
        )

    logger.info("Starting FedMed central server on %s for %s rounds", server_address, num_rounds)
    return fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )


def partition_cases_for_hospitals(
    cases: List[Dict[str, Any]], num_hospitals: int = 3
) -> List[List[Dict[str, Any]]]:
    """Deterministically partition list of patient cases across hospital silos.

    Args:
        cases: Discovered patient case records.
        num_hospitals: Number of hospital silos.

    Returns:
        List of partitioned case sub-lists.
    """
    if num_hospitals <= 0:
        raise ValueError(f"num_hospitals must be positive, got {num_hospitals}")
    partitions: List[List[Dict[str, Any]]] = [[] for _ in range(num_hospitals)]
    for idx, case in enumerate(cases):
        partitions[idx % num_hospitals].append(case)
    return partitions
