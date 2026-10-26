@echo off
setlocal enabledelayedexpansion

:: Ensure execution from repository root directory
cd /d "%~dp0.."

echo =====================================================================
echo  FedMed: End-to-End Live Demonstration Launcher (Debug Mode)
echo =====================================================================
echo  This script launches:
echo   1. FastAPI WebSocket & REST Server (Port 8000)
echo   2. Flower Central FL Server (Port 8080)
echo   3. Hospital Silo 1, Hospital Silo 2, Hospital Silo 3
echo   4. React + Recharts Live Monitoring Dashboard (Vite on Port 3000)
echo =====================================================================

if not exist outputs mkdir outputs

set PYTHON=.venv\Scripts\python.exe
if not exist "%PYTHON%" (
    set PYTHON=python
)

echo [1/5] Starting FastAPI Live Metrics & Slice Montage Server...
start "FedMed FastAPI Backend" cmd /k "%PYTHON% -m uvicorn src.api.ws_server:app --host 127.0.0.1 --port 8000"

timeout /t 2 /nobreak >nul

echo [2/5] Starting Flower Central FL Server...
start "FedMed Flower Server" cmd /k "%PYTHON% scripts\run_server.py --config config.yaml"

timeout /t 3 /nobreak >nul

echo [3/5] Starting Hospital Silo 1 (General Hospital Scribe)...
start "FedMed Hospital 1" cmd /k "%PYTHON% scripts\run_hospital.py --id hospital_1 --config config.yaml"

echo [4/5] Starting Hospital Silo 2 (University Health Alliance)...
start "FedMed Hospital 2" cmd /k "%PYTHON% scripts\run_hospital.py --id hospital_2 --config config.yaml"

echo [5/5] Starting Hospital Silo 3 (Memorial Cancer Institute)...
start "FedMed Hospital 3" cmd /k "%PYTHON% scripts\run_hospital.py --id hospital_3 --config config.yaml"

timeout /t 2 /nobreak >nul

echo.
echo [Dashboard] Launching Vite dev server...
start "FedMed React Dashboard" cmd /k "cd /d "%~dp0..\dashboard" && npm run dev"

timeout /t 3 /nobreak >nul

echo Opening Dashboard in your default browser...
start http://localhost:3000

echo =====================================================================
echo  FedMed System is LIVE!
echo   - Dashboard UI:    http://localhost:3000
echo   - REST API Docs:   http://localhost:8000/docs
echo   - WebSocket Feed:  ws://127.0.0.1:8000/ws/metrics
echo =====================================================================
echo Close the respective terminal windows to terminate individual nodes.
pause
