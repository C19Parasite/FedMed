"""FastAPI application providing WebSocket streaming for live metrics and REST APIs for segmentation masks."""

from __future__ import annotations

import asyncio
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import yaml

from src.metrics.publisher import register_metrics_subscriber, unregister_metrics_subscriber

app = FastAPI(
    title="FedMed Live Metrics and Segmentation API",
    description="Streams real-time federated training metrics over WebSocket and serves brain tumor MRI demo slice overlays.",
    version="1.0.0",
)

# Enable CORS for React Vite dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Active WebSocket connections
_ACTIVE_WEBSOCKETS: List[WebSocket] = []
_LOOP: Optional[asyncio.AbstractEventLoop] = None


@app.on_event("startup")
async def startup_event() -> None:
    global _LOOP
    _LOOP = asyncio.get_running_loop()

    # Register in-process callback with metrics publisher
    def on_new_metric(metric_data: Dict[str, Any]) -> None:
        if _LOOP and _ACTIVE_WEBSOCKETS:
            payload_str = json.dumps(metric_data)
            for ws in list(_ACTIVE_WEBSOCKETS):
                try:
                    asyncio.run_coroutine_threadsafe(ws.send_text(payload_str), _LOOP)
                except Exception:
                    pass

    register_metrics_subscriber(on_new_metric)


@app.get("/api/status")
async def get_status() -> Dict[str, Any]:
    """Return system operational status, latest round, and active privacy safeguards."""
    metrics_file = Path("results/metrics.json")
    latest_round = 0
    global_dice = 0.0
    epsilon = 0.0
    hospitals = {"hospital_1": "online", "hospital_2": "online", "hospital_3": "online"}

    if metrics_file.exists():
        try:
            records = json.loads(metrics_file.read_text(encoding="utf-8"))
            if records:
                last = records[-1]
                latest_round = last.get("round", 0)
                global_dice = last.get("global_val_dice", 0.0)
                epsilon = last.get("privacy", {}).get("epsilon", 0.0)
        except Exception:
            pass

    # Read config for privacy flags
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    return {
        "status": "idle" if latest_round >= cfg.get("server", {}).get("rounds", 3) else "running",
        "current_round": latest_round,
        "total_rounds": cfg.get("server", {}).get("rounds", 3),
        "best_global_dice": global_dice,
        "hospitals": hospitals,
        "privacy": {
            "tls": cfg.get("tls", {}).get("enabled", False),
            "he": cfg.get("he", {}).get("enabled", False),
            "dp": cfg.get("dp", {}).get("enabled", False),
            "noise_multiplier": cfg.get("dp", {}).get("noise_multiplier", 1.0),
            "cumulative_epsilon": epsilon,
        },
    }


@app.get("/api/history")
async def get_history() -> List[Dict[str, Any]]:
    """Return full historical progression of federated learning rounds."""
    metrics_file = Path("results/metrics.json")
    if metrics_file.exists():
        try:
            return json.loads(metrics_file.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []


@app.get("/api/config")
async def get_safe_config() -> Dict[str, Any]:
    """Return a strictly safe subset of the configuration (no private keys, certs, or local paths)."""
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    return {
        "debug": cfg.get("debug", False),
        "server": {
            "rounds": cfg.get("server", {}).get("rounds", 3),
            "min_fit_clients": cfg.get("server", {}).get("min_fit_clients", 2),
        },
        "hospitals": [h.get("name") for h in cfg.get("hospitals", [])],
        "model": {
            "name": cfg.get("model", {}).get("name", "monai_unet"),
            "norm": cfg.get("model", {}).get("norm", "INSTANCE"),
            "in_channels": cfg.get("model", {}).get("in_channels", 4),
            "out_channels": cfg.get("model", {}).get("out_channels", 3),
        },
        "privacy": {
            "tls_enabled": cfg.get("tls", {}).get("enabled", False),
            "he_enabled": cfg.get("he", {}).get("enabled", False),
            "dp_enabled": cfg.get("dp", {}).get("enabled", False),
            "clip_norm": cfg.get("dp", {}).get("clip_norm", 1.0),
            "noise_multiplier": cfg.get("dp", {}).get("noise_multiplier", 1.0),
        },
    }


@app.get("/api/ablation")
async def get_ablation_results() -> List[Dict[str, Any]]:
    """Return measured ablation table comparing privacy and security configurations."""
    csv_file = Path("results/ablation.csv")
    if not csv_file.exists():
        return []
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


@app.get("/api/cases")
async def get_cases() -> List[Dict[str, Any]]:
    """Return public demo cases with slice depths and Dice metrics."""
    manifest_file = Path("results/demo_cases/manifest.json")
    if manifest_file.exists():
        return json.loads(manifest_file.read_text(encoding="utf-8"))
    return []


@app.get("/api/cases/{case_id}/slice/{z}")
async def get_slice_image(
    case_id: str,
    z: int,
    view: str = Query("mri", regex="^(mri|gt|pred|central|overlay|gt_wt|gt_tc|gt_et|pred_wt|pred_tc|pred_et)$"),
    modality: Optional[str] = Query(None),
) -> FileResponse:
    """Serve individual slice PNG image for demo case visualization."""
    case_dirs = [
        Path("results/demo_cases") / case_id,
        Path("dashboard/public/cases") / case_id,
        Path("results/demo_cases") / f"demo_case_{case_id.replace('case_', '')}",
    ]

    target_dir = None
    for cd in case_dirs:
        if cd.exists():
            target_dir = cd
            break

    if not target_dir:
        raise HTTPException(status_code=404, detail="Case not found")

    if view == "mri" and modality and (target_dir / f"slice_{z}_mri_{modality.lower()}.png").exists():
        file_name = f"slice_{z}_mri_{modality.lower()}.png"
    else:
        file_name = f"slice_{z}_{view}.png"

    img_path = target_dir / file_name
    if not img_path.exists():
        raise HTTPException(status_code=404, detail=f"Slice {z} view {view} not found")

    return FileResponse(img_path, media_type="image/png")


@app.websocket("/ws/metrics")
async def websocket_metrics_endpoint(websocket: WebSocket) -> None:
    """WebSocket streaming endpoint broadcasting real-time round metrics to dashboard clients."""
    await websocket.accept()
    _ACTIVE_WEBSOCKETS.append(websocket)

    # Immediately push existing history so freshly connected dashboard reconstructs state
    history_file = Path("results/metrics.json")
    if history_file.exists():
        try:
            records = json.loads(history_file.read_text(encoding="utf-8"))
            for rec in records:
                await websocket.send_text(json.dumps(rec))
        except Exception:
            pass

    try:
        while True:
            # Keep socket alive and listen for client heartbeats or pings
            _ = await websocket.receive_text()
    except WebSocketDisconnect:
        if websocket in _ACTIVE_WEBSOCKETS:
            _ACTIVE_WEBSOCKETS.remove(websocket)


__all__ = ["app"]
