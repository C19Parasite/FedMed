@echo off
setlocal enabledelayedexpansion

echo =====================================================================
echo  FedMed: Starting Central Server and 3 Hospital Silos
echo =====================================================================

if not exist outputs mkdir outputs

set PYTHON=.venv\Scripts\python.exe
if not exist %PYTHON% (
    set PYTHON=python
)

echo [1/4] Launching Flower Server on 127.0.0.1:8080...
start "FedMed Server" /B %PYTHON% scripts\run_server.py --config config.yaml > outputs\server.log 2>&1
set SERVER_PID=%ERRORLEVEL%

timeout /t 3 /nobreak >nul

echo [2/4] Launching Hospital 1 (General Hospital Scribe)...
start "FedMed Hospital 1" /B %PYTHON% scripts\run_hospital.py --id hospital_1 --config config.yaml > outputs\hospital_1.log 2>&1

echo [3/4] Launching Hospital 2 (University Health Alliance)...
start "FedMed Hospital 2" /B %PYTHON% scripts\run_hospital.py --id hospital_2 --config config.yaml > outputs\hospital_2.log 2>&1

echo [4/4] Launching Hospital 3 (Memorial Cancer Institute)...
start "FedMed Hospital 3" /B %PYTHON% scripts\run_hospital.py --id hospital_3 --config config.yaml > outputs\hospital_3.log 2>&1

echo.
echo All nodes started. Logs streaming to outputs\*.log
echo Press Ctrl+C or run 'taskkill /F /IM python.exe' to terminate nodes.
echo =====================================================================

pause
