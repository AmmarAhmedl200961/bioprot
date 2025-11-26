"""
Protection Module - Cancelable Biometric Template Transforms

Implements two protection methods for 512-dimensional face embeddings:
1. Ortho+Sign: Orthonormal transformation + sign binarization
2. Perm+LUT: Permutation + Look-Up Table quantization
"""

import numpy as np
from scipy.linalg import qr
from typing import Tuple, Optional, Dict, Any
import hashlib
import struct


def seed_to_rng(seed: int) -> np.random.Generator:
    """Convert integer seed to numpy random generator."""
    return np.random.default_rng(seed)


def generate_ortho_matrix(dim: int, rng: np.random.Generator) -> np.ndarray:
    """Generate random orthonormal matrix via QR decomposition."""
    random_matrix = rng.standard_normal((dim, dim))
    Q, R = qr(random_matrix)
    # Ensure proper orthonormal matrix
    D = np.diag(np.sign(np.diag(R)))
    Q = Q @ D
    return Q


def protect_ortho_sign(embedding: np.ndarray, seed: int) -> np.ndarray:
    """Apply Ortho+Sign protection.
    
    Args:
        embedding: 512-dim float embedding vector
        seed: User-specific secret seed from KMS
        
    Returns:
        512-bit binary template (as uint8 array of 0s and 1s)
    """
    if embedding.ndim != 1:
        embedding = embedding.flatten()
    
    dim = len(embedding)
    rng = seed_to_rng(seed)
    Q = generate_ortho_matrix(dim, rng)
    
    # Transform and binarize
    transformed = Q @ embedding
    binary_template = (transformed >= 0).astype(np.uint8)
    
    return binary_template


def generate_permutation(dim: int, rng: np.random.Generator) -> np.ndarray:
    """Generate random permutation indices."""
    indices = np.arange(dim)
    rng.shuffle(indices)
    return indices


def generate_lut(n_bins: int, rng: np.random.Generator) -> np.ndarray:
    """Generate random Look-Up Table."""
    lut = np.arange(n_bins)
    rng.shuffle(lut)
    return lut


def protect_perm_lut(embedding: np.ndarray, seed: int, n_bins: int = 256) -> np.ndarray:
    """Apply Perm+LUT protection.
    
    Args:
        embedding: 512-dim float embedding vector
        seed: User-specific secret seed from KMS
        n_bins: Number of quantization bins (default 256 for 8-bit)
        
    Returns:
        512-element quantized template (uint8)
    """
    if embedding.ndim != 1:
        embedding = embedding.flatten()
    
    dim = len(embedding)
    rng = seed_to_rng(seed)
    
    # Generate permutation and LUT
    perm = generate_permutation(dim, rng)
    lut = generate_lut(n_bins, rng)
    
    # Normalize to [0, 1] then quantize
    e_min, e_max = embedding.min(), embedding.max()
    if e_max - e_min > 1e-10:
        normalized = (embedding - e_min) / (e_max - e_min)
    else:
        normalized = np.zeros_like(embedding) + 0.5
    
    quantized = np.clip((normalized * (n_bins - 1)).astype(int), 0, n_bins - 1)
    
    # Apply LUT then permutation
    lut_applied = lut[quantized]
    protected = lut_applied[perm]
    
    return protected.astype(np.uint8)


def hamming_similarity(t1: np.ndarray, t2: np.ndarray) -> float:
    """Calculate Hamming similarity between two binary templates."""
    if t1.shape != t2.shape:
        raise ValueError(f"Template shapes must match: {t1.shape} vs {t2.shape}")
    
    n_bits = len(t1)
    hamming_dist = np.sum(t1 != t2)
    similarity = 1.0 - (hamming_dist / n_bits)
    
    return float(similarity)


def match_protected(template1: np.ndarray, template2: np.ndarray, 
                   method: str = "ortho") -> float:
    """Match two protected templates.
    
    Args:
        template1: First protected template
        template2: Second protected template
        method: Protection method ('ortho' or 'perm')
        
    Returns:
        Similarity score in [0, 1]
    """
    if method == "ortho":
        return hamming_similarity(template1, template2)
    elif method == "perm":
        # For Perm+LUT, use normalized L1 distance
        max_diff = 255 * len(template1)  # Maximum possible L1
        l1_dist = np.sum(np.abs(template1.astype(float) - template2.astype(float)))
        similarity = 1.0 - (l1_dist / max_diff)
        return float(similarity)
    else:
        raise ValueError(f"Unknown method: {method}")


def pack_bits(binary_array: np.ndarray) -> bytes:
    """Pack binary array into bytes."""
    # Pad to multiple of 8
    padded_len = ((len(binary_array) + 7) // 8) * 8
    padded = np.zeros(padded_len, dtype=np.uint8)
    padded[:len(binary_array)] = binary_array
    
    # Pack into bytes
    packed = np.packbits(padded)
    return packed.tobytes()


def unpack_bits(packed_bytes: bytes, n_bits: int) -> np.ndarray:
    """Unpack bytes into binary array."""
    packed = np.frombuffer(packed_bytes, dtype=np.uint8)
    unpacked = np.unpackbits(packed)
    return unpacked[:n_bits]


def serialize_template(template: np.ndarray, method: str) -> bytes:
    """Serialize protected template to bytes."""
    header = struct.pack("<4sII", b"BPRT", len(template), 
                        0 if method == "ortho" else 1)
    
    if method == "ortho":
        data = pack_bits(template)
    else:
        data = template.astype(np.uint8).tobytes()
    
    return header + data


def deserialize_template(data: bytes) -> Tuple[np.ndarray, str]:
    """Deserialize protected template from bytes."""
    magic, length, method_code = struct.unpack("<4sII", data[:12])
    
    if magic != b"BPRT":
        raise ValueError("Invalid template format")
    
    method = "ortho" if method_code == 0 else "perm"
    payload = data[12:]
    
    if method == "ortho":
        template = unpack_bits(payload, length)
    else:
        template = np.frombuffer(payload, dtype=np.uint8)[:length]
    
    return template, method


def verify_cancelability(embedding: np.ndarray, seed1: int, seed2: int,
                        method: str = "ortho") -> Dict[str, float]:
    """Verify cancelability by comparing templates from different seeds."""
    if method == "ortho":
        t1 = protect_ortho_sign(embedding, seed1)
        t2 = protect_ortho_sign(embedding, seed2)
    else:
        t1 = protect_perm_lut(embedding, seed1)
        t2 = protect_perm_lut(embedding, seed2)
    
    similarity = match_protected(t1, t2, method)
    
    return {
        "seed1": seed1,
        "seed2": seed2,
        "similarity": similarity,
        "cancelable": similarity < 0.6  # Templates should be dissimilar
    }
