"""TLS verification suite testing certificate validation, plaintext rejection, and untrusted CA rejection."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import time
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.generate_certs import generate_tls_certificates
from src.security.tls import load_client_tls_credentials, load_server_tls_credentials


def run_tls_check() -> dict[str, bool]:
    """Test Flower TLS gRPC: correct CA connection, plaintext rejection, wrong CA rejection."""
    certs = generate_tls_certificates("certs")
    python_exe = sys.executable

    # Create dummy wrong CA for test (c)
    dummy_wrong_ca = Path("certs/wrong_ca.crt")
    wrong_certs = generate_tls_certificates("certs/wrong_tmp")
    dummy_wrong_ca.write_bytes(wrong_certs["ca_cert"].read_bytes())

    results = {}

    # Start TLS Server in subprocess on port 8095
    port = 8095
    server_code = f"""
import yaml, torch, flwr as fl
from src.fl.server import start_server_process
from src.fl.strategy import FedMedStrategy
from src.model.unet import build_unet
from fedmed.data.partition import create_or_load_partitions, get_global_eval_loaders
from fedmed.models.params import get_parameters
from flwr.common import ndarrays_to_parameters

with open('config.yaml') as f:
    cfg = yaml.safe_load(f)
cfg['server']['rounds'] = 1
cfg['server']['debug_rounds'] = 1
cfg['server']['address'] = '127.0.0.1:{port}'
cfg['server']['min_fit_clients'] = 1
cfg['server']['min_available_clients'] = 1
cfg['tls']['enabled'] = True
cfg['tls']['ca_cert'] = 'certs/ca.crt'
cfg['tls']['server_cert'] = 'certs/server.crt'
cfg['tls']['server_key'] = 'certs/server.key'

_ = create_or_load_partitions(cfg)
loaders = get_global_eval_loaders(cfg)
model = build_unet(cfg)
init_params = ndarrays_to_parameters(get_parameters(model))
strategy = FedMedStrategy(
    model=model,
    global_val_loader=loaders['val'],
    cfg=cfg,
    initial_parameters=init_params,
    min_fit_clients=1,
    min_available_clients=1,
    min_evaluate_clients=1,
)
start_server_process(cfg, server_address='127.0.0.1:{port}', num_rounds=1, strategy=strategy)
"""

    f_log = open("outputs/tls_server.log", "w", encoding="utf-8")
    server_proc = subprocess.Popen(
        [python_exe, "-c", server_code],
        stdout=f_log,
        stderr=subprocess.STDOUT,
        text=True,
    )
    time.sleep(4.0)

    # Helper to run client snippet
    def run_client(insecure: bool, ca_file: str | None) -> int:
        client_code = f"""
import yaml, torch, flwr as fl
from src.fl.client import FedMedClient
from src.model.unet import build_unet
from fedmed.data.partition import create_or_load_partitions, get_hospital_loaders
from src.security.tls import load_client_tls_credentials

with open('config.yaml') as f:
    cfg = yaml.safe_load(f)
cfg['tls']['enabled'] = True
_ = create_or_load_partitions(cfg)
loaders = get_hospital_loaders('hospital_1', cfg)
model = build_unet(cfg)
client = FedMedClient('hospital_1', model, loaders.train, loaders.val, cfg, device='cpu')

root_certs = load_client_tls_credentials('{ca_file}') if '{ca_file}' != 'None' else None
fl.client.start_client(
    server_address='127.0.0.1:{port}',
    client=client.to_client(),
    root_certificates=root_certs,
    insecure={insecure},
    max_retries=1,
)
"""
        proc = subprocess.run(
            [python_exe, "-c", client_code],
            capture_output=True,
            text=True,
            timeout=25,
        )
        return proc.returncode

    try:
        # Test (b): Plaintext connection to TLS server is rejected
        print("Testing (b) Plaintext connection rejection...")
        code_b = run_client(insecure=True, ca_file=None)
        results["plaintext_rejected"] = (code_b != 0)
        print(f" -> Plaintext connection exit code: {code_b} (rejected={results['plaintext_rejected']})")

        # Test (c): Connection with wrong CA is rejected
        print("Testing (c) Wrong CA rejection...")
        code_c = run_client(insecure=False, ca_file="certs/wrong_ca.crt")
        results["wrong_ca_rejected"] = (code_c != 0)
        print(f" -> Wrong CA connection exit code: {code_c} (rejected={results['wrong_ca_rejected']})")

        # Test (a): Valid connection with correct CA completes round
        print("Testing (a) Valid TLS connection with correct CA...")
        code_a = run_client(insecure=False, ca_file="certs/ca.crt")
        results["valid_tls_passed"] = (code_a == 0)
        print(f" -> Valid TLS connection exit code: {code_a} (passed={results['valid_tls_passed']})")

    finally:
        server_proc.kill()
        f_log.close()

    return results


def test_tls_security_suite():
    """Pytest entrypoint for TLS verification."""
    results = run_tls_check()
    assert results["valid_tls_passed"], "Valid TLS connection should complete round"
    assert results["plaintext_rejected"], "Plaintext connection to TLS server must be rejected"
    assert results["wrong_ca_rejected"], "Connection with untrusted CA must be rejected"


if __name__ == "__main__":
    print("=" * 60)
    print(" FedMed Phase 4: TLS Security Verification")
    print("=" * 60)
    res = run_tls_check()
    print("-" * 60)
    for k, v in res.items():
        status = "PASS" if v else "FAIL"
        print(f"[{status}] {k}")
    print("=" * 60)
