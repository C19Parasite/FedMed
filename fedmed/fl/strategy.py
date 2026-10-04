"""Flower federated aggregation strategy with BraTS metric aggregation."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union
import flwr as fl
from flwr.common import Metrics, Scalar
from flwr.server.strategy import FedAvg

from fedmed.utils.logging import get_logger

logger = get_logger(__name__)


def aggregate_eval_metrics(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate evaluation metrics weighted by number of evaluation examples.

    Args:
        metrics: List of (num_examples, metrics_dict) from participating clients.

    Returns:
        Weighted average metrics dictionary.
    """
    total_examples = sum(num_examples for num_examples, _ in metrics)
    if total_examples == 0:
        return {}

    aggregated: Dict[str, float] = {}
    metric_keys = ["dice_tc", "dice_wt", "dice_et", "dice_mean", "val_loss"]

    for key in metric_keys:
        weighted_sum = sum(
            num_examples * float(client_metrics.get(key, 0.0))
            for num_examples, client_metrics in metrics
            if key in client_metrics
        )
        aggregated[key] = weighted_sum / total_examples

    logger.info(
        "Aggregated Round Metrics -> Mean Dice: %.4f (TC: %.4f, WT: %.4f, ET: %.4f)",
        aggregated.get("dice_mean", 0.0),
        aggregated.get("dice_tc", 0.0),
        aggregated.get("dice_wt", 0.0),
        aggregated.get("dice_et", 0.0),
    )
    return aggregated


class FedMedFedAvg(FedAvg):
    """FedMed Federated Averaging strategy customized for BraTS multi-silo metrics."""

    def __init__(
        self,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 3,
        min_evaluate_clients: int = 3,
        min_available_clients: int = 3,
        **kwargs,
    ) -> None:
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_metrics_aggregation_fn=aggregate_eval_metrics,
            **kwargs,
        )
