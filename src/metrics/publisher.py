"""Centralized metrics publisher and WebSocket streaming broadcaster."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import threading
from typing import Any, Callable, Dict, List, Optional
from fedmed.utils.logging import get_logger

logger = get_logger(__name__)

# In-process pub/sub subscriber callbacks (thread-safe)
_SUBSCRIBERS: List[Callable[[Dict[str, Any]], None]] = []
_LOCK = threading.Lock()


def register_metrics_subscriber(callback: Callable[[Dict[str, Any]], None]) -> None:
    """Register a callback or queue putter to receive live metrics JSON on every round completion."""
    with _LOCK:
        if callback not in _SUBSCRIBERS:
            _SUBSCRIBERS.append(callback)


def unregister_metrics_subscriber(callback: Callable[[Dict[str, Any]], None]) -> None:
    """Remove a previously registered subscriber callback."""
    with _LOCK:
        if callback in _SUBSCRIBERS:
            _SUBSCRIBERS.remove(callback)


def notify_subscribers(metrics: Dict[str, Any]) -> None:
    """Notify all registered subscribers with new metrics payload."""
    with _LOCK:
        subscribers_copy = list(_SUBSCRIBERS)

    for cb in subscribers_copy:
        try:
            cb(metrics)
        except Exception as e:
            logger.warning("Error notifying metrics subscriber: %s", e)


def publish_round_metrics(
    round_num: int,
    global_val_dice: float,
    global_val_loss: float,
    hospital_metrics: Dict[str, Dict[str, float]],
    participating_clients: int,
    total_clients: int = 3,
    bytes_sent: int = 0,
    bytes_received: int = 0,
    round_duration_sec: float = 0.0,
    tls_enabled: bool = False,
    he_enabled: bool = False,
    dp_enabled: bool = False,
    epsilon: float = 0.0,
    sigma: float = 0.0,
    csv_path: str | Path = "results/metrics.csv",
    json_path: str | Path = "results/metrics.json",
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Single point of truth for publishing federated learning round metrics.

    Appends metrics to CSV and JSON files, and broadcasts the event to WebSocket subscribers.

    Args:
        round_num: Current federated training round index.
        global_val_dice: Mean BraTS Dice on server-side global validation holdout.
        global_val_loss: Loss on server-side global validation holdout.
        hospital_metrics: Dict mapping hospital_id to local training loss and local dice.
        participating_clients: Number of clients that successfully completed this round.
        total_clients: Total configured hospital pool.
        bytes_sent: Estimated bytes transmitted from server.
        bytes_received: Estimated bytes received by server.
        round_duration_sec: Execution duration of this round.
        tls_enabled: Whether TLS gRPC was active.
        he_enabled: Whether CKKS homomorphic encryption was active.
        dp_enabled: Whether Differential Privacy was active.
        epsilon: Cumulative DP privacy budget spent so far.
        sigma: DP Gaussian noise multiplier.
        csv_path: CSV output file path.
        json_path: JSON output file path.
        extra: Optional additional dictionary fields.

    Returns:
        Structured round metrics record.
    """
    now_utc = datetime.now(timezone.utc).isoformat()

    record: Dict[str, Any] = {
        "event": "round_complete",
        "round": int(round_num),
        "global_val_dice": round(float(global_val_dice), 4),
        "global_val_loss": round(float(global_val_loss), 4),
        "participating_clients": int(participating_clients),
        "total_clients": int(total_clients),
        "hospital_metrics": hospital_metrics,
        "bytes_sent": int(bytes_sent),
        "bytes_received": int(bytes_received),
        "round_duration_sec": round(float(round_duration_sec), 2),
        "privacy": {
            "tls": bool(tls_enabled),
            "he": bool(he_enabled),
            "dp": bool(dp_enabled),
            "epsilon": round(float(epsilon), 4) if dp_enabled else 0.0,
            "sigma": float(sigma) if dp_enabled else 0.0,
        },
        "timestamp": now_utc,
    }
    if extra:
        record.update(extra)

    # 1. Append to CSV
    csv_file = Path(csv_path)
    csv_file.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_file.exists()

    csv_fields = [
        "round",
        "global_val_dice",
        "global_val_loss",
        "participating_clients",
        "total_clients",
        "bytes_sent",
        "bytes_received",
        "round_duration_sec",
        "tls",
        "he",
        "dp",
        "epsilon",
        "sigma",
        "timestamp",
    ]

    csv_row = {
        "round": record["round"],
        "global_val_dice": record["global_val_dice"],
        "global_val_loss": record["global_val_loss"],
        "participating_clients": record["participating_clients"],
        "total_clients": record["total_clients"],
        "bytes_sent": record["bytes_sent"],
        "bytes_received": record["bytes_received"],
        "round_duration_sec": record["round_duration_sec"],
        "tls": record["privacy"]["tls"],
        "he": record["privacy"]["he"],
        "dp": record["privacy"]["dp"],
        "epsilon": record["privacy"]["epsilon"],
        "sigma": record["privacy"]["sigma"],
        "timestamp": record["timestamp"],
    }

    try:
        with open(csv_file, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields)
            if not file_exists:
                writer.writeheader()
            writer.writerow(csv_row)
    except Exception as e:
        logger.error("Failed to append to %s: %s", csv_path, e)

    # 2. Append to JSON list
    json_file = Path(json_path)
    json_file.parent.mkdir(parents=True, exist_ok=True)

    all_records: List[Dict[str, Any]] = []
    if json_file.exists():
        try:
            content = json_file.read_text(encoding="utf-8")
            if content.strip():
                all_records = json.loads(content)
        except Exception as e:
            logger.warning("Could not parse existing %s, resetting list: %s", json_path, e)

    all_records.append(record)
    try:
        json_file.write_text(json.dumps(all_records, indent=2), encoding="utf-8")
    except Exception as e:
        logger.error("Failed to write to %s: %s", json_path, e)

    # 3. Broadcast to WebSocket subscribers
    notify_subscribers(record)

    logger.info(
        "Round %d complete: Global Val Dice=%.4f, Val Loss=%.4f (Participants: %d of %d, Epsilon: %.2f)",
        record["round"],
        record["global_val_dice"],
        record["global_val_loss"],
        record["participating_clients"],
        record["total_clients"],
        record["privacy"]["epsilon"],
    )
    return record


__all__ = ["publish_round_metrics", "register_metrics_subscriber", "unregister_metrics_subscriber"]
