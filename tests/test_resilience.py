"""Unit tests for Node Dropout Resilience and Weight Renormalization."""

from __future__ import annotations

import numpy as np
import pytest
from flwr.common import FitRes, Parameters, Status, Code, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy

from src.fl.strategy import FedMedStrategy


class MockClientProxy(ClientProxy):
    def get_run_status(self):
        return None
    def get_properties(self, ins, timeout, group_id=None):
        return None
    def get_parameters(self, ins, timeout, group_id=None):
        return None
    def fit(self, ins, timeout, group_id=None):
        return None
    def evaluate(self, ins, timeout, group_id=None):
        return None
    def reconnect(self, ins, timeout, group_id=None):
        return None


def test_aggregation_renormalization_2_of_3_clients():
    """Verify hand-computed weighted average when 1 of 3 clients drops out."""
    w1 = [np.array([1.0, 2.0, 3.0], dtype=np.float32)]
    w2 = [np.array([4.0, 5.0, 6.0], dtype=np.float32)]
    # w3 dropped out!

    n1 = 10
    n2 = 30
    total_responding = n1 + n2

    # Hand-computed expected weighted average
    expected_w = (w1[0] * n1 + w2[0] * n2) / total_responding

    res1 = FitRes(
        status=Status(code=Code.OK, message="Success"),
        parameters=ndarrays_to_parameters(w1),
        num_examples=n1,
        metrics={"hospital_id": "hospital_1", "train_loss": 0.5},
    )
    res2 = FitRes(
        status=Status(code=Code.OK, message="Success"),
        parameters=ndarrays_to_parameters(w2),
        num_examples=n2,
        metrics={"hospital_id": "hospital_2", "train_loss": 0.4},
    )

    results = [(MockClientProxy("cid1"), res1), (MockClientProxy("cid2"), res2)]
    failures = [(MockClientProxy("cid3"), RuntimeError("Connection reset"))]

    strategy = FedMedStrategy(
        model=None,
        global_val_loader=None,
        cfg={"server": {"min_fit_clients": 2}, "hospitals": [1, 2, 3]},
    )

    agg_params, metrics = strategy.aggregate_fit(server_round=1, results=results, failures=failures)
    assert agg_params is not None
    assert metrics["participating_clients"] == 2

    agg_weights = parameters_to_ndarrays(agg_params)
    assert np.allclose(agg_weights[0], expected_w, atol=1e-6)
