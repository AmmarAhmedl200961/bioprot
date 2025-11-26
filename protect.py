"""
Cancelable template protection methods for face embeddings.

Implements:
- Ortho+Sign: Orthonormal projection + sign binarization
- Perm+LUT: Permutation + group-wise LUT quantization
"""

import base64
import hashlib
import hmac
import json
from typing import Tuple, Dict, Any, Optional

import numpy as np
from numpy.random import Generator


# ============================================================================
# Deterministic RNG utilities
# ============================================================================

def seed_to_rng(seed: bytes, label: str = "") -> Generator:
    """
    Derive a deterministic NumPy RNG from a seed and optional label.
    
    Uses HMAC-SHA256(seed, label) to derive entropy for the RNG.
    
    Args:
        seed: User-specific seed bytes (typically 32 bytes)
        label: Optional label to derive different RNGs from same seed
        
    Returns:
        NumPy Generator for deterministic random number generation
    """
    # Derive entropy using HMAC-SHA256
    h = hmac.new(seed, label.encode('utf-8'), hashlib.sha256)
    entropy_bytes = h.digest()
    
    # Convert first 8 bytes to uint64 for seed
    entropy = np.frombuffer(entropy_bytes[:8], dtype=np.uint64)[0]
    
    # Create deterministic RNG using SeedSequence
    seed_seq = np.random.SeedSequence(entropy)
    return np.random.default_rng(seed_seq)


def generate_seed(nbytes: int = 32) -> bytes:
    """
    Generate a cryptographically secure random seed.
    
    Args:
        nbytes: Number of random bytes to generate
        
    Returns:
        Random seed bytes
    """
    import secrets
    return secrets.token_bytes(nbytes)


# ============================================================================
# Bit packing utilities
# ============================================================================

def pack_bits(bits: np.ndarray) -> bytes:
    """
    Pack a binary array into bytes (little-endian bit order).
    
    Args:
        bits: 1D array of 0s and 1s
        
    Returns:
        Packed bytes
    """
    bits = bits.astype(np.uint8)
    # Pad to multiple of 8
    n_pad = (8 - len(bits) % 8) % 8
    if n_pad > 0:
        bits = np.concatenate([bits, np.zeros(n_pad, dtype=np.uint8)])
    
    # Pack using numpy's packbits (big-endian by default, but we document our convention)
    packed = np.packbits(bits)
    return packed.tobytes()


def unpack_bits(packed: bytes, n_bits: int) -> np.ndarray:
    """
    Unpack bytes into a binary array.
    
    Args:
        packed: Packed bytes
        n_bits: Number of bits to extract
        
    Returns:
        1D array of 0s and 1s (length n_bits)
    """
    arr = np.frombuffer(packed, dtype=np.uint8)
    bits = np.unpackbits(arr)
    return bits[:n_bits]


def bits_to_base64(bits: np.ndarray) -> str:
    """
    Convert binary array to base64 string.
    
    Args:
        bits: 1D array of 0s and 1s
        
    Returns:
        Base64-encoded string
    """
    packed = pack_bits(bits)
    return base64.b64encode(packed).decode('ascii')


def base64_to_bits(b64: str, n_bits: int) -> np.ndarray:
    """
    Convert base64 string back to binary array.
    
    Args:
        b64: Base64-encoded string
        n_bits: Number of bits to extract
        
    Returns:
        1D array of 0s and 1s
    """
    packed = base64.b64decode(b64.encode('ascii'))
    return unpack_bits(packed, n_bits)


# ============================================================================
# Ortho+Sign method
# ============================================================================

def generate_ortho_matrix(seed: bytes, dim: int) -> np.ndarray:
    """
    Generate a deterministic orthonormal matrix Q via QR decomposition.
    
    Args:
        seed: User-specific seed
        dim: Dimension of the square matrix (embedding dimension)
        
    Returns:
        Orthonormal matrix Q of shape (dim, dim)
    """
    rng = seed_to_rng(seed, "ortho_Q")
    
    # Generate random Gaussian matrix
    G = rng.normal(size=(dim, dim))
    
    # QR decomposition
    Q, R = np.linalg.qr(G)
    
    # Normalize Q to have determinant +1 (resolve sign ambiguity)
    # Force diagonal of R to be positive
    signs = np.sign(np.diag(R))
    signs[signs == 0] = 1
    Q = Q * signs[np.newaxis, :]
    
    return Q


def protect_ortho_sign(embedding: np.ndarray, seed: bytes, 
                       normalize: bool = True) -> np.ndarray:
    """
    Apply Ortho+Sign protection to an embedding.
    
    Args:
        embedding: Face embedding vector (1D array)
        seed: User-specific seed
        normalize: Whether to L2-normalize the embedding first
        
    Returns:
        Binary template (1D array of 0s and 1s)
    """
    x = embedding.copy()
    
    # Normalize if requested
    if normalize:
        norm = np.linalg.norm(x)
        if norm > 0:
            x = x / norm
    
    # Generate orthonormal matrix
    dim = len(x)
    Q = generate_ortho_matrix(seed, dim)
    
    # Project
    y = Q @ x
    
    # Binarize
    b = (y >= 0).astype(np.uint8)
    
    return b


def match_ortho_sign(template1: np.ndarray, template2: np.ndarray) -> float:
    """
    Compute similarity between two Ortho+Sign templates.
    
    Uses normalized Hamming similarity: 1 - (hamming_distance / n_bits)
    
    Args:
        template1: First binary template
        template2: Second binary template
        
    Returns:
        Similarity score in [0, 1]
    """
    if len(template1) != len(template2):
        raise ValueError("Templates must have the same length")
    
    # Hamming distance
    hamming = np.sum(template1 != template2)
    
    # Normalized similarity
    similarity = 1.0 - (hamming / len(template1))
    
    return similarity


# ============================================================================
# Perm+LUT method
# ============================================================================

def generate_permutation(seed: bytes, size: int) -> np.ndarray:
    """
    Generate a deterministic permutation from seed.
    
    Args:
        seed: User-specific seed
        size: Size of the permutation
        
    Returns:
        Permutation array (indices)
    """
    rng = seed_to_rng(seed, "perm")
    perm = np.arange(size)
    rng.shuffle(perm)
    return perm


def generate_lut(seed: bytes, n_groups: int, n_bins: int, 
                 bits_per_bin: int = 1) -> np.ndarray:
    """
    Generate deterministic lookup tables for each group.
    
    Args:
        seed: User-specific seed
        n_groups: Number of groups
        n_bins: Number of quantization bins per group
        bits_per_bin: Number of output bits per bin
        
    Returns:
        LUT array of shape (n_groups, n_bins, bits_per_bin) with 0s and 1s
    """
    rng = seed_to_rng(seed, "lut")
    
    # Generate random bits for each (group, bin) -> bits_per_bin bits
    lut = rng.integers(0, 2, size=(n_groups, n_bins, bits_per_bin), 
                       dtype=np.uint8)
    
    return lut


def quantize_groups(embedding: np.ndarray, group_size: int, 
                    n_bins: int) -> Tuple[np.ndarray, int]:
    """
    Quantize embedding into groups and bins.
    
    Uses uniform quantization on normalized [-1, 1] range.
    
    Args:
        embedding: Normalized embedding vector
        group_size: Number of elements per group
        n_bins: Number of quantization bins
        
    Returns:
        Tuple of (quantized indices array, n_groups)
    """
    n = len(embedding)
    n_groups = (n + group_size - 1) // group_size
    
    # Pad if necessary
    if n % group_size != 0:
        pad_size = n_groups * group_size - n
        embedding = np.concatenate([embedding, np.zeros(pad_size)])
    
    # Reshape into groups
    groups = embedding.reshape(n_groups, group_size)
    
    # Define uniform bin edges on [-1, 1]
    # For n_bins=4: [-inf, -0.5, 0, 0.5, inf] -> bins [0,1,2,3]
    bin_edges = np.linspace(-1, 1, n_bins + 1)
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf
    
    # Quantize each group independently
    quantized = np.zeros((n_groups, group_size), dtype=np.int32)
    for g in range(n_groups):
        for i in range(group_size):
            quantized[g, i] = np.digitize(groups[g, i], bin_edges) - 1
            # Ensure in valid range [0, n_bins-1]
            quantized[g, i] = np.clip(quantized[g, i], 0, n_bins - 1)
    
    return quantized, n_groups


def protect_perm_lut(embedding: np.ndarray, seed: bytes,
                     group_size: int = 8, n_bins: int = 4,
                     bits_per_bin: int = 1, normalize: bool = True) -> np.ndarray:
    """
    Apply Perm+LUT protection to an embedding.
    
    Args:
        embedding: Face embedding vector
        seed: User-specific seed
        group_size: Number of elements per group
        n_bins: Number of quantization bins
        bits_per_bin: Output bits per quantized bin
        normalize: Whether to L2-normalize the embedding first
        
    Returns:
        Binary template (1D array of 0s and 1s)
    """
    x = embedding.copy()
    
    # Normalize
    if normalize:
        norm = np.linalg.norm(x)
        if norm > 0:
            x = x / norm
    
    # Quantize into groups
    quantized, n_groups = quantize_groups(x, group_size, n_bins)
    
    # Generate LUT
    lut = generate_lut(seed, n_groups, n_bins, bits_per_bin)
    
    # Apply LUT: for each group and element, map quantized bin to bits
    bits_list = []
    for g in range(n_groups):
        for i in range(group_size):
            bin_idx = quantized[g, i]
            bits = lut[g, bin_idx]
            bits_list.append(bits)
    
    # Concatenate all bits
    all_bits = np.concatenate(bits_list)
    
    # Apply permutation
    perm = generate_permutation(seed, len(all_bits))
    permuted_bits = all_bits[perm]
    
    return permuted_bits


def match_perm_lut(template1: np.ndarray, template2: np.ndarray) -> float:
    """
    Compute similarity between two Perm+LUT templates.
    
    Uses normalized Hamming similarity.
    
    Args:
        template1: First binary template
        template2: Second binary template
        
    Returns:
        Similarity score in [0, 1]
    """
    if len(template1) != len(template2):
        raise ValueError("Templates must have the same length")
    
    hamming = np.sum(template1 != template2)
    similarity = 1.0 - (hamming / len(template1))
    
    return similarity


# ============================================================================
# Unified API
# ============================================================================

def protect_embedding(embedding: np.ndarray, seed: bytes, 
                      method: str = "ortho", **kwargs) -> Dict[str, Any]:
    """
    Unified API to protect an embedding with specified method.
    
    Args:
        embedding: Face embedding vector
        seed: User-specific seed
        method: Protection method ("ortho" or "permlut")
        **kwargs: Method-specific parameters
        
    Returns:
        Dictionary with template data including:
        - method: protection method name
        - params: method parameters
        - template: binary template (numpy array)
        - template_b64: base64-encoded template
    """
    if method == "ortho":
        normalize = kwargs.get("normalize", True)
        template = protect_ortho_sign(embedding, seed, normalize=normalize)
        params = {
            "d": len(embedding),
            "nbits": len(template),
            "normalize": normalize
        }
    elif method == "permlut":
        group_size = kwargs.get("group_size", 8)
        n_bins = kwargs.get("n_bins", 4)
        bits_per_bin = kwargs.get("bits_per_bin", 1)
        normalize = kwargs.get("normalize", True)
        
        template = protect_perm_lut(embedding, seed, group_size=group_size,
                                    n_bins=n_bins, bits_per_bin=bits_per_bin,
                                    normalize=normalize)
        params = {
            "d": len(embedding),
            "group_size": group_size,
            "n_bins": n_bins,
            "bits_per_bin": bits_per_bin,
            "nbits": len(template),
            "normalize": normalize
        }
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return {
        "method": method,
        "params": params,
        "template": template,
        "template_b64": bits_to_base64(template)
    }


def match_protected(template1_data: Dict[str, Any], 
                   template2_data: Dict[str, Any]) -> float:
    """
    Unified API to match two protected templates.
    
    Args:
        template1_data: First template dictionary (from protect_embedding)
        template2_data: Second template dictionary
        
    Returns:
        Similarity score in [0, 1]
    """
    method1 = template1_data["method"]
    method2 = template2_data["method"]
    
    if method1 != method2:
        raise ValueError(f"Cannot match templates of different methods: {method1} vs {method2}")
    
    # Extract templates - handle both numpy arrays and base64-encoded templates
    params1 = template1_data["params"]
    params2 = template2_data["params"]
    
    if "template" in template1_data and isinstance(template1_data["template"], np.ndarray):
        t1 = template1_data["template"]
    else:
        t1 = base64_to_bits(template1_data["template_b64"], params1["nbits"])
    
    if "template" in template2_data and isinstance(template2_data["template"], np.ndarray):
        t2 = template2_data["template"]
    else:
        t2 = base64_to_bits(template2_data["template_b64"], params2["nbits"])
    
    # Match based on method
    if method1 == "ortho":
        return match_ortho_sign(t1, t2)
    elif method1 == "permlut":
        return match_perm_lut(t1, t2)
    else:
        raise ValueError(f"Unknown method: {method1}")


def serialize_template(template_data: Dict[str, Any], user_id: str,
                       seed_version: int = 1) -> str:
    """
    Serialize a protected template to JSON string.
    
    Args:
        template_data: Template dictionary from protect_embedding
        user_id: User identifier
        seed_version: Version of the seed used
        
    Returns:
        JSON string
    """
    from datetime import datetime
    
    # Remove numpy array if present (keep base64)
    serializable = {
        "user_id": user_id,
        "method": template_data["method"],
        "params": {**template_data["params"], "seed_version": seed_version},
        "template_b64": template_data["template_b64"],
        "created_at": datetime.utcnow().isoformat() + "Z"
    }
    
    return json.dumps(serializable, indent=2)


def deserialize_template(json_str: str) -> Dict[str, Any]:
    """
    Deserialize a protected template from JSON string.
    
    Args:
        json_str: JSON string
        
    Returns:
        Template dictionary
    """
    data = json.loads(json_str)
    
    # Add method and params at top level for compatibility
    return {
        "user_id": data["user_id"],
        "method": data["method"],
        "params": data["params"],
        "template_b64": data["template_b64"],
        "created_at": data.get("created_at")
    }
