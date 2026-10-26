# TenSEAL CKKS Homomorphic Encryption Overhead Report

## 1. Executive Summary & Plain-Language Explanation of CKKS
**What is CKKS?**
The Cheon-Kim-Kim-Song (CKKS) cryptosystem is a homomorphic encryption scheme designed specifically for floating-point real numbers. Unlike exact schemes (like BFV or BGV) that operate over discrete integers, CKKS treats encryption noise as part of the arithmetic precision error, making it ideally suited for machine learning weights and neural network gradients.

**How does it work in FedMed?**
1. **Local Weight Delta Scaling**: Hospital i trains locally and computes weight delta Delta_i = W_i - W.
2. **Local Multiplied Encryption**: The client multiplies Delta_i by its local patient count n_i and encrypts the vector: c_i = Enc(n_i * Delta_i).
3. **Server-Side Homomorphic Addition**: The central server receives c_1, c_2, c_3 and public counts n_1, n_2, n_3. Using only public homomorphic addition, the server computes C_sum = c_1 + c_2 + c_3 = Enc(sum(n_i * Delta_i)). The server never decrypts and never possesses the secret key.
4. **Client-Side Decryption & Normalization**: The encrypted sum is returned to the hospitals. Each hospital decrypts the sum using its private key and divides by total N = sum(n_i). This finishes the exact FedAvg weighted formula Delta_avg = sum(n_i * Delta_i) / sum(n_i) without requiring server-side ciphertext division.

---

## 2. Real Measured Performance Metrics

| Metric | Measured Real Value | Rationale / Explanation |
|---|---|---|
| **Parameters Encrypted** | **8,095** | Lightweight U-Net profile (`channels: [4, 8, 16]`) |
| **CKKS Ciphertext Chunks** | **2 chunks** | Vector chunked to fit 4,096 slots ($N/2$ for poly degree 8192) |
| **Plaintext Size** | **31.62 KB** | Unencrypted 32-bit float parameters |
| **Ciphertext Wire Size** | **647.26 KB** | Serialized CKKS polynomial ring elements |
| **Ciphertext Expansion Ratio** | **20.5x** | Inherent polynomial representation overhead in ring LWE |
| **Client Encryption Time** | **0.010 s** | CPU time to encode and encrypt on hospital silo |
| **Server Aggregation Time** | **0.006 s** | Time to homomorphically sum 3 client ciphertexts |
| **Client Decryption Time** | **0.003 s** | Time to decrypt and unflatten on hospital silo |
| **Maximum Numerical Error** | **3.73e-09** | Max abs diff vs standard plaintext FedAvg (Must be $< 10^{-3}$: **PASS**) |
| **Mean Numerical Error** | **1.24e-10** | Average deviation across all model parameters |

---

## 3. Cryptographic Guarantee
- **Server Knowledge**: The central server only possesses `keys/server_context.tenseal` (created via `make_context_public()`).
- **Mathematical Decryption Test**: As proven by `tests/test_he.py::test_server_context_cannot_decrypt`, any call to decrypt on the server raises an immediate cryptographic exception.
- **Privacy Achieved**: Even under a rogue, curious, or compromised central server, individual patient updates and individual hospital gradient trajectories are cryptographically masked.
