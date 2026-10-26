"""Generate self-signed Certificate Authority (CA) and server TLS certificates for FedMed gRPC."""

from __future__ import annotations

import datetime
from pathlib import Path
import ipaddress
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID


def generate_tls_certificates(output_dir: str | Path = "certs") -> dict[str, Path]:
    """Generate CA and server certificate with SAN localhost & 127.0.0.1.

    Args:
        output_dir: Directory where certs and keys are written.

    Returns:
        Dictionary of created file paths.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    ca_key_file = out_path / "ca.key"
    ca_cert_file = out_path / "ca.crt"
    server_key_file = out_path / "server.key"
    server_cert_file = out_path / "server.crt"

    # 1. Generate CA Private Key and Self-Signed Certificate
    ca_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    ca_subject = x509.Name([
        x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "FedMed Privacy Consortium"),
        x509.NameAttribute(NameOID.COMMON_NAME, "FedMed Root CA"),
    ])
    now = datetime.datetime.now(datetime.timezone.utc)
    ca_cert = (
        x509.CertificateBuilder()
        .subject_name(ca_subject)
        .issuer_name(ca_subject)
        .public_key(ca_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=3650))
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=True,
                crl_sign=True,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .sign(ca_key, hashes.SHA256())
    )

    # 2. Generate Server Private Key and Certificate signed by CA
    server_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    server_subject = x509.Name([
        x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "FedMed Central Hospital Silo Server"),
        x509.NameAttribute(NameOID.COMMON_NAME, "localhost"),
    ])
    server_cert = (
        x509.CertificateBuilder()
        .subject_name(server_subject)
        .issuer_name(ca_subject)
        .public_key(server_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=825))
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(
            x509.SubjectAlternativeName([
                x509.DNSName("localhost"),
                x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
            ]),
            critical=False,
        )
        .sign(ca_key, hashes.SHA256())
    )

    # Serialize to PEM
    ca_key_file.write_bytes(
        ca_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    ca_cert_file.write_bytes(ca_cert.public_bytes(serialization.Encoding.PEM))

    server_key_file.write_bytes(
        server_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    server_cert_file.write_bytes(server_cert.public_bytes(serialization.Encoding.PEM))

    print(f"Successfully generated TLS certificates in {out_path.resolve()}:")
    print(f" - CA Root Certificate: {ca_cert_file}")
    print(f" - Server Certificate:  {server_cert_file}")
    print(f" - Server Private Key:  {server_key_file}")

    return {
        "ca_cert": ca_cert_file,
        "ca_key": ca_key_file,
        "server_cert": server_cert_file,
        "server_key": server_key_file,
    }


if __name__ == "__main__":
    generate_tls_certificates()
