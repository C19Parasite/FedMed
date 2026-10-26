#!/usr/bin/env bash
set -e

echo "====================================================================="
echo " FedMed: End-to-End Live Demonstration Launcher (Debug Mode)"
echo "====================================================================="

mkdir -p outputs

PYTHON=".venv/bin/python"
if [ ! -f "$PYTHON" ]; then
    PYTHON="python3"
fi

echo "[1/5] Starting FastAPI Backend on port 8000..."
$PYTHON -m uvicorn src.api.ws_server:app --host 127.0.0.1 --port 8000 > outputs/api.log 2>&1 &
API_PID=$!

sleep 2

echo "[2/5] Starting Flower Server on 127.0.0.1:8080..."
$PYTHON scripts/run_server.py --config config.yaml > outputs/server.log 2>&1 &
SERVER_PID=$!

sleep 3

echo "[3/5] Starting Hospital Silo 1..."
$PYTHON scripts/run_hospital.py --id hospital_1 --config config.yaml > outputs/hospital_1.log 2>&1 &
H1_PID=$!

echo "[4/5] Starting Hospital Silo 2..."
$PYTHON scripts/run_hospital.py --id hospital_2 --config config.yaml > outputs/hospital_2.log 2>&1 &
H2_PID=$!

echo "[5/5] Starting Hospital Silo 3..."
$PYTHON scripts/run_hospital.py --id hospital_3 --config config.yaml > outputs/hospital_3.log 2>&1 &
H3_PID=$!

echo "Launching React Dashboard on http://localhost:5173..."
(cd dashboard && npm run dev) &
DASH_PID=$!

cleanup() {
    echo "Stopping all FedMed processes..."
    kill $API_PID $SERVER_PID $H1_PID $H2_PID $H3_PID $DASH_PID 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "FedMed is running. Press Ctrl+C to terminate."
wait
