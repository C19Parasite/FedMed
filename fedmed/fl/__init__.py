"""Federated learning framework (Flower server, clients, strategies)."""

from fedmed.fl.client import HospitalClient
from fedmed.fl.server import partition_cases_for_hospitals, start_fedmed_server
from fedmed.fl.strategy import FedMedFedAvg, aggregate_eval_metrics

__all__ = [
    "FedMedFedAvg",
    "HospitalClient",
    "aggregate_eval_metrics",
    "partition_cases_for_hospitals",
    "start_fedmed_server",
]
