"""Generate TenSEAL CKKS homomorphic encryption context and keys for FedMed."""

from __future__ import annotations

import argparse
from pathlib import Path
import tenseal as ts


def generate_he_contexts(
    output_dir: str | Path = "keys",
    poly_modulus_degree: int = 8192,
    coeff_mod_bit_sizes: list[int] | None = None,
    scale_bits: int = 40,
) -> dict[str, Path]:
    """Generate TenSEAL CKKS encryption contexts: secret context for hospitals and public context for server.

    Security & Simplification Disclosure:
    In a real production hospital deployment, multi-party computation or threshold homomorphic
    encryption would be employed so no single hospital holds the complete decryption key.
    Here, the 3 hospital silos share one secret context as a documented pedagogical simplification.
    Crucially, the central server receives ONLY the public context (with secret key permanently dropped),
    meaning the server can homomorphically add ciphertexts but can NEVER decrypt any individual update.

    Args:
        output_dir: Output directory for serialized contexts.
        poly_modulus_degree: Polynomial modulus degree (default 8192 -> 4096 slots).
        coeff_mod_bit_sizes: Bit sizes for RNS primes (default [60, 40, 40, 60]).
        scale_bits: Scale factor in bits (default 40 -> 2^40).

    Returns:
        Dictionary mapping context types to saved file paths.
    """
    if coeff_mod_bit_sizes is None:
        coeff_mod_bit_sizes = [60, 40, 40, 60]

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    secret_path = out_path / "hospital_context.tenseal"
    public_path = out_path / "server_context.tenseal"

    # Create secret CKKS context with secret key
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=poly_modulus_degree,
        coeff_mod_bit_sizes=coeff_mod_bit_sizes,
    )
    context.global_scale = 2 ** scale_bits
    context.generate_galois_keys()

    # Save full secret context (for hospital silos only)
    secret_bytes = context.serialize(save_secret_key=True)
    secret_path.write_bytes(secret_bytes)

    # Drop secret key to make context public (for central aggregation server)
    context.make_context_public()
    public_bytes = context.serialize()
    public_path.write_bytes(public_bytes)

    print(f"Generated CKKS encryption contexts in {out_path.resolve()}:")
    print(f" - Hospital secret context: {secret_path} ({len(secret_bytes)} bytes)")
    print(f" - Server public context:   {public_path} ({len(public_bytes)} bytes)")

    return {
        "secret_context": secret_path,
        "public_context": public_path,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate CKKS homomorphic encryption keys.")
    parser.add_argument("--keys-dir", type=str, default="keys", help="Directory to save keys.")
    parser.add_argument("--poly-degree", type=int, default=8192, help="Polynomial modulus degree.")
    args = parser.parse_args()
    generate_he_contexts(output_dir=args.keys_dir, poly_modulus_degree=args.poly_degree)
