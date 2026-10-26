#!/usr/bin/env bash
set -e

mkdir -p outputs

PYTHON=".venv/bin/python"
if [ ! -f "$PYTHON" ]; then
    PYTHON="python3"
fi

echo "====================================================================="
echo " FedMed: Starting Central Server and 3 Hospital Silos"
echo "====================================================================="

echo "[1/4] Starting Server..."
$PYTHON scripts/run_server.py --config config.yaml > outputs/server.log 2>&1 &
SERVER_PID=$!

sleep 3

echo "[2/4] Starting Hospital 1..."
$PYTHON scripts/run_hospital.py --id hospital_1 --config config.yaml > outputs/hospital_1.log 2>&1 &
H1_PID=$!

echo "[3/4] Starting Hospital 2..."
$PYTHON scripts/run_hospital.py --id hospital_2 --config config.yaml > outputs/hospital_2.log 2>&1 &
H2_PID=$!

echo "[4/4] Starting Hospital 3..."
$PYTHON scripts/run_hospital.py --id hospital_3 --config config.yaml > outputs/hospital_3.log 2>&1 &
H3_PID=$!

trap "echo 'Terminating FedMed processes...'; kill $SERVER_PID $H1_PID $H2_PID $H3_PID 2>/dev/null; exit 0" SIGINT SIGTERM

echo "All nodes started. Logs streaming to outputs/*.log. Press Ctrl+C to stop."
wait $SERVER_PID $H1_PID $H2_PID $H3_PID
