# API Reference

This document provides detailed API documentation for BioProt modules.

## protect.py

Core protection algorithms for biometric template transformation.

### Functions

#### `generate_seed()`

Generate a cryptographically secure random seed.

```python
def generate_seed() -> bytes:
    """
    Generate a 32-byte random seed for template protection.
    
    Returns:
        bytes: 32 random bytes suitable for seeding transforms.
    
    Example:
        >>> seed = generate_seed()
        >>> len(seed)
        32
    """
```

#### `seed_to_rng(seed)`

Convert a seed to a NumPy random generator.

```python
def seed_to_rng(seed: bytes) -> np.random.Generator:
    """
    Create deterministic random generator from seed.
    
    Args:
        seed: Bytes to use as seed (hashed to 64-bit integer).
    
    Returns:
        np.random.Generator: Seeded random number generator.
    
    Example:
        >>> rng = seed_to_rng(b'my_secret_seed')
        >>> rng.random()  # Always same value for same seed
        0.123456...
    """
```

#### `protect_ortho_sign(embedding, seed)`

Apply Ortho+Sign transformation to an embedding.

```python
def protect_ortho_sign(
    embedding: np.ndarray,
    seed: bytes
) -> dict:
    """
    Protect embedding using orthonormal projection + sign binarization.
    
    Args:
        embedding: 512-dimensional face embedding (np.ndarray).
        seed: 32-byte user seed from KMS.
    
    Returns:
        dict: Protected template with keys:
            - 'method': 'ortho'
            - 'bits': Binary array (512 elements)
            - 'packed': Base64-encoded packed bits (64 bytes)
            - 'dim': 512
    
    Example:
        >>> emb = np.random.randn(512)
        >>> seed = generate_seed()
        >>> template = protect_ortho_sign(emb, seed)
        >>> template['method']
        'ortho'
        >>> len(template['packed'])
        88  # Base64 of 64 bytes
    """
```

#### `protect_perm_lut(embedding, seed, k=8, L=4)`

Apply Perm+LUT transformation to an embedding.

```python
def protect_perm_lut(
    embedding: np.ndarray,
    seed: bytes,
    k: int = 8,
    L: int = 4
) -> dict:
    """
    Protect embedding using permutation + lookup table quantization.
    
    Args:
        embedding: 512-dimensional face embedding.
        seed: 32-byte user seed from KMS.
        k: Group size (default 8, must divide 512).
        L: Number of quantization bins (default 4).
    
    Returns:
        dict: Protected template with keys:
            - 'method': 'permlut'
            - 'bits': Binary array
            - 'packed': Base64-encoded packed bits
            - 'k': Group size used
            - 'L': Bins used
    
    Example:
        >>> template = protect_perm_lut(emb, seed, k=8, L=4)
        >>> template['method']
        'permlut'
    """
```

#### `protect_embedding(embedding, seed, method='ortho')`

High-level protection function with method selection.

```python
def protect_embedding(
    embedding: np.ndarray,
    seed: bytes,
    method: str = 'ortho'
) -> dict:
    """
    Protect an embedding using specified method.
    
    Args:
        embedding: 512-dimensional face embedding.
        seed: 32-byte user seed from KMS.
        method: Protection method ('ortho' or 'permlut').
    
    Returns:
        dict: Protected template.
    
    Raises:
        ValueError: If method is not recognized.
    
    Example:
        >>> template = protect_embedding(emb, seed, method='ortho')
    """
```

#### `match_protected(template1, template2)`

Compute similarity between two protected templates.

```python
def match_protected(
    template1: dict,
    template2: dict
) -> float:
    """
    Compute Hamming similarity between protected templates.
    
    Args:
        template1: First protected template (dict or serialized).
        template2: Second protected template (dict or serialized).
    
    Returns:
        float: Similarity score in [0, 1].
            - 1.0 = identical templates
            - 0.5 = random/uncorrelated
            - 0.0 = completely different
    
    Raises:
        ValueError: If templates have different methods.
    
    Example:
        >>> score = match_protected(enrolled_template, probe_template)
        >>> if score > 0.80:
        ...     print("Match!")
    """
```

#### `serialize_template(template)` / `deserialize_template(data)`

Template serialization for storage.

```python
def serialize_template(template: dict) -> str:
    """
    Serialize template to JSON string for storage.
    
    Args:
        template: Protected template dictionary.
    
    Returns:
        str: JSON string representation.
    """

def deserialize_template(data: str) -> dict:
    """
    Deserialize template from JSON string.
    
    Args:
        data: JSON string from serialize_template().
    
    Returns:
        dict: Protected template dictionary.
    """
```

---

## kms_sim.py

Key Management Simulator for secure seed storage.

### Class: LocalKMS

```python
class LocalKMS:
    """
    Local file-based Key Management Simulator.
    
    Uses PBKDF2 for key derivation and Fernet for encryption.
    
    Attributes:
        store_path (Path): Path to encrypted storage file.
    
    Example:
        >>> kms = LocalKMS("my_passphrase", store_path="keys.bin")
        >>> seed = kms.create_user_key("alice")
        >>> retrieved = kms.get_user_seed("alice")
        >>> seed == retrieved
        True
    """
```

#### `__init__(passphrase, store_path=None)`

```python
def __init__(
    self,
    passphrase: str,
    store_path: Optional[str] = None
):
    """
    Initialize KMS with encryption passphrase.
    
    Args:
        passphrase: Password for encrypting stored keys.
        store_path: Path to storage file (default: ./kms_store.bin).
    """
```

#### `create_user_key(user_id)`

```python
def create_user_key(self, user_id: str) -> bytes:
    """
    Create and store a new key for a user.
    
    Args:
        user_id: Unique user identifier.
    
    Returns:
        bytes: 32-byte random seed.
    
    Raises:
        ValueError: If user already exists.
    """
```

#### `get_user_seed(user_id)`

```python
def get_user_seed(self, user_id: str) -> bytes:
    """
    Retrieve stored seed for a user.
    
    Args:
        user_id: User identifier.
    
    Returns:
        bytes: User's 32-byte seed.
    
    Raises:
        KeyError: If user not found.
    """
```

#### `rotate_user_key(user_id)`

```python
def rotate_user_key(self, user_id: str) -> bytes:
    """
    Generate new key for user, invalidating old one.
    
    Args:
        user_id: User identifier.
    
    Returns:
        bytes: New 32-byte seed.
    
    Raises:
        KeyError: If user not found.
    """
```

#### `delete_user(user_id)`

```python
def delete_user(self, user_id: str) -> bool:
    """
    Remove user from KMS.
    
    Args:
        user_id: User identifier.
    
    Returns:
        bool: True if deleted, False if not found.
    """
```

#### `list_users()`

```python
def list_users(self) -> List[str]:
    """
    Get list of all enrolled users.
    
    Returns:
        List[str]: User identifiers.
    """
```

### Function: `get_or_create_seed(kms, user_id)`

```python
def get_or_create_seed(
    kms: LocalKMS,
    user_id: str
) -> bytes:
    """
    Get existing seed or create new one.
    
    Convenience function for enrollment workflows.
    
    Args:
        kms: LocalKMS instance.
        user_id: User identifier.
    
    Returns:
        bytes: User's seed (existing or newly created).
    """
```

---

## evaluate.py

Evaluation metrics and security tests.

### Functions

#### `compute_roc_metrics(genuine_scores, impostor_scores)`

```python
def compute_roc_metrics(
    genuine_scores: List[float],
    impostor_scores: List[float]
) -> dict:
    """
    Compute ROC curve and related metrics.
    
    Args:
        genuine_scores: Match scores for genuine pairs.
        impostor_scores: Match scores for impostor pairs.
    
    Returns:
        dict: Metrics including:
            - 'auc': Area Under ROC Curve
            - 'eer': Equal Error Rate
            - 'tar_at_far_01': TAR at FAR=0.1%
            - 'tar_at_far_001': TAR at FAR=0.01%
            - 'fpr': False Positive Rates
            - 'tpr': True Positive Rates
            - 'thresholds': Decision thresholds
    """
```

#### `naive_inversion_test(templates, embeddings, method)`

```python
def naive_inversion_test(
    templates: List[dict],
    embeddings: List[np.ndarray],
    method: str
) -> dict:
    """
    Test irreversibility using naive matrix inversion.
    
    For Ortho+Sign: Attempts pseudo-inverse of projection.
    
    Args:
        templates: List of protected templates.
        embeddings: Corresponding original embeddings.
        method: Protection method used.
    
    Returns:
        dict: Results including:
            - 'attack_success': bool (True if inversion worked)
            - 'reconstruction_error': float
            - 'correlation': float (original vs reconstructed)
    """
```

#### `test_revocation(kms, user_id, embedding)`

```python
def test_revocation(
    kms: LocalKMS,
    user_id: str,
    embedding: np.ndarray
) -> dict:
    """
    Test that key rotation invalidates old templates.
    
    Args:
        kms: LocalKMS instance.
        user_id: User to test.
        embedding: Test embedding.
    
    Returns:
        dict: Results including:
            - 'pre_rotation_match': bool
            - 'post_rotation_match': bool
            - 'old_vs_new_similarity': float
            - 'revocation_effective': bool
    """
```

---

## CLI Commands

### `enroll`

```bash
python cli.py enroll --user USER_ID --embedding PATH [--method {ortho,permlut}]
```

Enroll a new user with their embedding.

### `verify`

```bash
python cli.py verify --user USER_ID --embedding PATH [--threshold FLOAT]
```

Verify a probe embedding against enrolled template.

### `rotate`

```bash
python cli.py rotate --user USER_ID
```

Rotate user's key, invalidating previous templates.

### `inspect`

```bash
python cli.py inspect --user USER_ID
```

Display template metadata and statistics.

---

## Error Handling

### Common Exceptions

```python
# User not found
KeyError: "User 'unknown' not found in KMS"

# Invalid embedding dimension
ValueError: "Embedding must be 512-dimensional, got 256"

# Method mismatch
ValueError: "Cannot match templates with different methods: ortho vs permlut"

# Missing passphrase
ValueError: "KMS passphrase not set. Use BIOPROT_KMS_PASSPHRASE environment variable"
```

### Return Codes (CLI)

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | General error |
| 2 | User not found |
| 3 | Verification failed (no match) |
| 4 | Invalid input |
