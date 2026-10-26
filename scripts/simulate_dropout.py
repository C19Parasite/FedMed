"""Simulate dynamic client node dropout mid-round and subsequent rejoining (Phase 5)."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import time

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def log_timeline(event: str) -> None:
    now = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{now}] {event}")


def simulate_node_dropout() -> None:
    print("=" * 70)
    print(" FedMed Phase 5: Node Dropout & Resilience Simulation")
    print("=" * 70)

    # Clean old metrics
    for p in ["results/metrics.csv", "results/metrics.json"]:
        if Path(p).exists():
            Path(p).unlink()

    Path("outputs").mkdir(parents=True, exist_ok=True)
    f_srv = open("outputs/dropout_server.log", "w", encoding="utf-8")
    f_h1 = open("outputs/dropout_h1.log", "w", encoding="utf-8")
    f_h2 = open("outputs/dropout_h2.log", "w", encoding="utf-8")
    f_h3 = open("outputs/dropout_h3.log", "w", encoding="utf-8")

    python_exe = sys.executable

    log_timeline("STEP 1: Starting Central Flower Server (2 rounds, min_fit_clients=2)...")
    srv_proc = subprocess.Popen(
        [python_exe, "scripts/run_server.py", "--config", "config.yaml", "--rounds", "2"],
        stdout=f_srv,
        stderr=subprocess.STDOUT,
        text=True,
    )
    time.sleep(4.0)

    log_timeline("STEP 2: Starting Hospital 1, Hospital 2 (programmed to drop out in Round 2), and Hospital 3...")
    h1_proc = subprocess.Popen(
        [python_exe, "scripts/run_hospital.py", "--id", "hospital_1", "--config", "config.yaml"],
        stdout=f_h1,
        stderr=subprocess.STDOUT,
        text=True,
    )
    # Hospital 2 is configured to crash in round 2
    h2_proc = subprocess.Popen(
        [python_exe, "scripts/run_hospital.py", "--id", "hospital_2", "--config", "config.yaml", "--dropout-round", "2"],
        stdout=f_h2,
        stderr=subprocess.STDOUT,
        text=True,
    )
    h3_proc = subprocess.Popen(
        [python_exe, "scripts/run_hospital.py", "--id", "hospital_3", "--config", "config.yaml"],
        stdout=f_h3,
        stderr=subprocess.STDOUT,
        text=True,
    )

    log_timeline("STEP 3: Server running. Round 1 has 3 hospitals. In Round 2, Hospital 2 crashes.")
    try:
        srv_proc.wait(timeout=60)
        for p in [h1_proc, h2_proc, h3_proc]:
            if p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    p.kill()
    except subprocess.TimeoutExpired:
        log_timeline("[WARNING] Timeout reached, terminating processes...")
        for p in [srv_proc, h1_proc, h2_proc, h3_proc]:
            p.kill()
    finally:
        f_srv.close()
        f_h1.close()
        f_h2.close()
        f_h3.close()

    metrics = []
    metrics_file = Path("results/metrics.json")
    if metrics_file.exists():
        metrics = json.loads(metrics_file.read_text(encoding="utf-8"))

    print("\n" + "=" * 70)
    print("                     RESILIENCE TIMELINE SUMMARY")
    print("=" * 70)
    print(f"{'Round':<8} | {'Participating':<15} | {'Global Val Dice':<18} | {'Status'}")
    print("-" * 70)
    for m in metrics:
        p_str = f"{m['participating_clients']}/{m['total_clients']}"
        status_str = "All Silos Online" if m['participating_clients'] == 3 else "DROPOUT SURVIVED (2/3)"
        print(f"{m['round']:<8} | {p_str:<15} | {m['global_val_dice']:<18.4f} | {status_str}")
    print("-" * 70)

    # Verification checks
    assert len(metrics) == 2, f"Expected 2 rounds logged, got {len(metrics)}"
    assert metrics[0]["participating_clients"] == 3, "Round 1 should have 3 clients"
    assert metrics[1]["participating_clients"] == 2, "Round 2 should have 2 clients after dropout"

    print("[PASS] Dropout resilience verified: Round 2 survived Hospital 2 dropout with 2/3 nodes.")
    print("=" * 70)


if __name__ == "__main__":
    simulate_node_dropout()
