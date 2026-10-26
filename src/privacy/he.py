"""TenSEAL CKKS Homomorphic Encryption utilities for encrypted federated aggregation."""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import numpy as np
import tenseal as ts
from fedmed.utils.logging import get_logger

logger = get_logger(__name__)


def load_hospital_context(context_path: str | Path = "keys/hospital_context.tenseal") -> ts.Context:
    """Load the hospital's private CKKS context containing the secret key."""
    raw = Path(context_path).read_bytes()
    ctx = ts.context_from(raw)
    return ctx


def load_server_context(context_path: str | Path = "keys/server_context.tenseal") -> ts.Context:
    """Load the server's public CKKS context with the secret key removed."""
    raw = Path(context_path).read_bytes()
    ctx = ts.context_from(raw)
    return ctx


def flatten_tensors(tensors: Sequence[np.ndarray]) -> Tuple[np.ndarray, List[Tuple[Tuple[int, ...], np.dtype]]]:
    """Flatten a list of multi-dimensional numpy arrays into a single 1D float vector and record metadata."""
    shapes_and_types = [(t.shape, t.dtype) for t in tensors]
    flat = np.concatenate([t.reshape(-1).astype(np.float32) for t in tensors])
    return flat, shapes_and_types


def unflatten_tensors(flat: np.ndarray, shapes_and_types: List[Tuple[Tuple[int, ...], np.dtype]]) -> List[np.ndarray]:
    """Reconstruct a list of multi-dimensional tensors from a 1D vector using metadata."""
    tensors = []
    idx = 0
    for shape, dtype in shapes_and_types:
        size = int(np.prod(shape)) if shape else 1
        chunk = flat[idx : idx + size].reshape(shape).astype(dtype)
        tensors.append(chunk)
        idx += size
    return tensors


def encrypt_update(
    update_tensors: Sequence[np.ndarray],
    context: ts.Context,
    num_examples: int = 1,
    slot_count: int = 4096,
) -> Tuple[List[bytes], Dict[str, Any]]:
    """Flatten update, multiply by num_examples (n_i), split into CKKS slot chunks, and encrypt.

    Mathematical Operation:
        c_i = Enc(n_i * Delta_i)

    Args:
        update_tensors: List of numpy arrays representing model weight deltas.
        context: Hospital CKKS context (with secret key).
        num_examples: Number of local training examples (n_i).
        slot_count: Maximum slots per CKKS ciphertext chunk (default 4096).

    Returns:
        Tuple of (serialized_ciphertext_chunks, metadata_dictionary).
    """
    start_time = time.perf_counter()
    flat_vector, shapes_and_types = flatten_tensors(update_tensors)
    scaled_vector = (flat_vector * float(num_examples)).tolist()

    total_elements = len(scaled_vector)

    encrypted_chunks: List[bytes] = []
    for i in range(0, total_elements, slot_count):
        chunk = scaled_vector[i : i + slot_count]
        enc_vec = ts.ckks_vector(context, chunk)
        encrypted_chunks.append(enc_vec.serialize())

    enc_time = time.perf_counter() - start_time
    total_cipher_bytes = sum(len(c) for c in encrypted_chunks)
    plain_bytes = flat_vector.nbytes

    meta = {
        "shapes_and_types": shapes_and_types,
        "total_elements": total_elements,
        "slot_count": slot_count,
        "num_chunks": len(encrypted_chunks),
        "num_examples": num_examples,
        "encryption_time_sec": enc_time,
        "ciphertext_bytes": total_cipher_bytes,
        "plaintext_bytes": plain_bytes,
    }
    return encrypted_chunks, meta


def homomorphic_aggregate_chunks(
    client_encrypted_chunks: List[List[bytes]],
    server_context: ts.Context,
) -> Tuple[List[bytes], float]:
    """Sum ciphertexts element-wise across clients using only homomorphic addition on the server.

    Server operates entirely on ciphertext. It has NO secret key and cannot read any value.

    Args:
        client_encrypted_chunks: List of client submissions; each submission is a list of serialized chunks.
        server_context: Public server context.

    Returns:
        Tuple of (aggregated_serialized_chunks, aggregation_duration_seconds).
    """
    start_time = time.perf_counter()
    if not client_encrypted_chunks:
        return [], 0.0

    num_chunks = len(client_encrypted_chunks[0])
    aggregated_chunks: List[bytes] = []

    for chunk_idx in range(num_chunks):
        # Initialize accumulator with first client's chunk
        accum = ts.ckks_vector_from(server_context, client_encrypted_chunks[0][chunk_idx])
        for client_idx in range(1, len(client_encrypted_chunks)):
            next_vec = ts.ckks_vector_from(server_context, client_encrypted_chunks[client_idx][chunk_idx])
            accum = accum + next_vec
        aggregated_chunks.append(accum.serialize())

    agg_time = time.perf_counter() - start_time
    return aggregated_chunks, agg_time


def decrypt_and_average(
    aggregated_chunks: List[bytes],
    shapes_and_types: List[Tuple[Tuple[int, ...], np.dtype]],
    total_elements: int,
    total_n: int,
    hospital_context: ts.Context,
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """Decrypt the aggregated sum on the hospital client and divide by total_n to obtain weighted average.

    Mathematical Operation:
        Delta_avg = Dec(Sum_i Enc(n_i * Delta_i)) / total_n = (Sum_i n_i * Delta_i) / total_n

    Why division happens on clients:
    1. CKKS ciphertext division by an integer is homomorphic scalar multiplication by (1/total_n).
       Executing it post-decryption in plaintext avoids extra homomorphic scalar multiplication noise.
    2. Resilience: If any hospital drops out mid-round, the server simply passes total_n = sum(n_responding),
       and client post-decryption division is automatically exact for surviving hospitals without re-encrypting.

    Args:
        aggregated_chunks: Serialized chunks representing the homomorphic sum.
        shapes_and_types: Original tensor shapes.
        total_elements: Length of flattened vector.
        total_n: Sum of local examples across responding clients.
        hospital_context: Secret context containing private key.

    Returns:
        Tuple of (list_of_averaged_update_arrays, metadata_dictionary).
    """
    start_time = time.perf_counter()
    decrypted_vals: List[float] = []

    for chunk_bytes in aggregated_chunks:
        vec = ts.ckks_vector_from(hospital_context, chunk_bytes)
        decrypted_vals.extend(vec.decrypt())

    # Truncate to original length (since CKKS vectors may be padded to slot count)
    decrypted_array = np.array(decrypted_vals[:total_elements], dtype=np.float32)

    # Divide by total_n to finish weighted FedAvg aggregation
    divisor = max(1, total_n)
    averaged_array = decrypted_array / float(divisor)

    # Reconstruct original tensor shapes
    reconstructed_tensors = unflatten_tensors(averaged_array, shapes_and_types)

    dec_time = time.perf_counter() - start_time
    meta = {
        "decryption_time_sec": dec_time,
        "total_elements": total_elements,
        "total_n": total_n,
    }
    return reconstructed_tensors, meta


__all__ = [
    "load_hospital_context",
    "load_server_context",
    "flatten_tensors",
    "unflatten_tensors",
    "encrypt_update",
    "homomorphic_aggregate_chunks",
    "decrypt_and_average",
]
