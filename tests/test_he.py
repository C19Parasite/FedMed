"""Unit tests for TenSEAL CKKS Homomorphic Encryption in FedMed."""

from __future__ import annotations

import numpy as np
import pytest
import tenseal as ts

from src.privacy.he import (
    decrypt_and_average,
    encrypt_update,
    flatten_tensors,
    homomorphic_aggregate_chunks,
    load_hospital_context,
    load_server_context,
    unflatten_tensors,
)
from scripts.generate_he_keys import generate_he_contexts


@pytest.fixture(scope="module")
def he_contexts(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("he_keys")
    paths = generate_he_contexts(
        output_dir=tmp_dir,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60],
        scale_bits=40,
    )
    secret_ctx = load_hospital_context(paths["secret_context"])
    public_ctx = load_server_context(paths["public_context"])
    return secret_ctx, public_ctx


def test_server_context_cannot_decrypt(he_contexts):
    """Verify cryptographic boundary: the server context lacks the secret key and CANNOT decrypt."""
    secret_ctx, public_ctx = he_contexts

    data = [1.5, 2.5, 3.5]
    # Hospital encrypts data
    enc_vec = ts.ckks_vector(secret_ctx, data)
    cipher_bytes = enc_vec.serialize()

    # Server loads ciphertext using its public context
    server_vec = ts.ckks_vector_from(public_ctx, cipher_bytes)

    # Decrypt attempt MUST raise an exception
    with pytest.raises(Exception):
        _ = server_vec.decrypt()


def test_tensor_flatten_unflatten_roundtrip():
    """Verify multi-tensor flattening and restructuring preserve exact shapes and values."""
    t1 = np.random.randn(2, 3, 4).astype(np.float32)
    t2 = np.random.randn(5).astype(np.float32)
    t3 = np.random.randn(1, 2, 2, 2).astype(np.float32)

    flat, meta = flatten_tensors([t1, t2, t3])
    reconstructed = unflatten_tensors(flat, meta)

    assert len(reconstructed) == 3
    assert np.allclose(t1, reconstructed[0])
    assert np.allclose(t2, reconstructed[1])
    assert np.allclose(t3, reconstructed[2])


def test_homomorphic_addition_3_clients(he_contexts):
    """Verify that 3-client CKKS aggregation matches weighted FedAvg within 1e-3 precision."""
    secret_ctx, public_ctx = he_contexts

    # 3 mock hospital updates
    np.random.seed(42)
    shapes = [(4, 4), (2, 3)]
    client1_tensors = [np.random.randn(*s).astype(np.float32) * 0.1 for s in shapes]
    client2_tensors = [np.random.randn(*s).astype(np.float32) * 0.1 for s in shapes]
    client3_tensors = [np.random.randn(*s).astype(np.float32) * 0.1 for s in shapes]

    n1, n2, n3 = 10, 20, 30
    total_n = n1 + n2 + n3

    # Ground-truth plaintext weighted average
    true_avg = [
        (t1 * n1 + t2 * n2 + t3 * n3) / total_n
        for t1, t2, t3 in zip(client1_tensors, client2_tensors, client3_tensors)
    ]

    # Client-side encryption: Enc(n_i * Delta_i)
    chunks1, meta1 = encrypt_update(client1_tensors, secret_ctx, num_examples=n1)
    chunks2, meta2 = encrypt_update(client2_tensors, secret_ctx, num_examples=n2)
    chunks3, meta3 = encrypt_update(client3_tensors, secret_ctx, num_examples=n3)

    # Server-side homomorphic summation
    agg_chunks, agg_time = homomorphic_aggregate_chunks([chunks1, chunks2, chunks3], public_ctx)

    # Client-side decryption and division by total_n
    decrypted_avg, dec_meta = decrypt_and_average(
        agg_chunks,
        meta1["shapes_and_types"],
        meta1["total_elements"],
        total_n,
        secret_ctx,
    )

    # Verify numerical precision
    for true_t, dec_t in zip(true_avg, decrypted_avg):
        max_abs_err = float(np.max(np.abs(true_t - dec_t)))
        assert max_abs_err < 1e-3, f"CKKS error too high: {max_abs_err}"


def test_homomorphic_addition_dropout_2_clients(he_contexts):
    """Verify that dropout (2 of 3 clients) aggregates accurately with total_n renormalization."""
    secret_ctx, public_ctx = he_contexts

    np.random.seed(99)
    shapes = [(8, 8)]
    client1_tensors = [np.random.randn(*s).astype(np.float32) * 0.05 for s in shapes]
    client2_tensors = [np.random.randn(*s).astype(np.float32) * 0.05 for s in shapes]

    n1, n2 = 15, 25
    total_n = n1 + n2

    true_avg = [(t1 * n1 + t2 * n2) / total_n for t1, t2 in zip(client1_tensors, client2_tensors)]

    chunks1, meta1 = encrypt_update(client1_tensors, secret_ctx, num_examples=n1)
    chunks2, meta2 = encrypt_update(client2_tensors, secret_ctx, num_examples=n2)

    agg_chunks, _ = homomorphic_aggregate_chunks([chunks1, chunks2], public_ctx)

    decrypted_avg, _ = decrypt_and_average(
        agg_chunks,
        meta1["shapes_and_types"],
        meta1["total_elements"],
        total_n,
        secret_ctx,
    )

    for true_t, dec_t in zip(true_avg, decrypted_avg):
        max_abs_err = float(np.max(np.abs(true_t - dec_t)))
        assert max_abs_err < 1e-3, f"CKKS dropout aggregation error too high: {max_abs_err}"
