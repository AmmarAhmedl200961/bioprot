# Security Model

This document describes the security properties, threat model, and limitations of BioProt.

## Threat Model

### Assets to Protect

1. **Biometric Templates**: Protected representations of face embeddings
2. **User Seeds**: Cryptographic seeds used for transform generation
3. **KMS Storage**: Encrypted file containing all user keys

### Adversary Capabilities

We assume an adversary who may:

1. **Obtain template database**: Access to stored protected templates
2. **Know the algorithm**: Full knowledge of Ortho+Sign and Perm+LUT transforms
3. **Have computing resources**: Can perform brute-force and ML-based attacks
4. **NOT have KMS access**: Cannot decrypt the key management storage

### Attack Vectors

| Attack | Description | Mitigation |
|--------|-------------|------------|
| Template inversion | Reconstruct embedding from template | Irreversible transforms (sign binarization, quantization) |
| Cross-matching | Link templates across databases | User-specific random seeds |
| Replay attack | Reuse stolen template | Key rotation revokes old templates |
| Brute-force seeds | Guess user seeds | 256-bit random seeds (2²⁵⁶ space) |
| Side-channel | Timing attacks on matching | Constant-time Hamming distance |

## Security Properties

### 1. Irreversibility

**Definition**: Given a protected template T = f(x, s), it should be computationally infeasible to recover the original embedding x.

**Implementation**:
- Ortho+Sign: Sign binarization destroys magnitude information
- Perm+LUT: Quantization loses precision, LUT is one-way

**Verification**: See `evaluate.py::naive_inversion_test()` and `regressor_inversion_test()`

### 2. Cancelability

**Definition**: If a template is compromised, the system can issue a new template that:
- Does not match the old template
- Works correctly for legitimate authentication

**Implementation**:
- Key rotation generates new random seed
- New seed produces uncorrelated template
- Old template becomes invalid (random similarity ~0.5)

**Verification**: See `evaluate.py::test_revocation()`

### 3. Unlinkability

**Definition**: Templates generated with different seeds should be statistically independent, preventing cross-database linking.

**Implementation**:
- Each user has unique seed
- Same embedding + different seeds → uncorrelated templates
- No pattern leakage between templates

**Analysis**:
```
For seeds s₁ ≠ s₂ and embedding x:
- T₁ = f(x, s₁)
- T₂ = f(x, s₂)
- E[similarity(T₁, T₂)] ≈ 0.5 (random chance)
- Var[similarity(T₁, T₂)] → 0 as dim → ∞
```

### 4. Performance Preservation

**Definition**: Protected templates should maintain recognition accuracy comparable to raw embeddings.

**Implementation**:
- Ortho+Sign: Orthonormal projection approximately preserves distances
- Both methods: Proper threshold calibration

**Metrics**:
- AUC should be > 0.99 for good systems
- TAR @ FAR=0.1% should be > 95%

## KMS Security

### Key Derivation

```
master_key = PBKDF2(
    password=user_passphrase,
    salt=random_16_bytes,
    iterations=100000,
    hash=SHA256
)
```

### Encryption

- Algorithm: Fernet (AES-128-CBC + HMAC-SHA256)
- Each user's seed encrypted separately
- Integrity protected (HMAC)

### Storage Format

```json
{
    "user_id": {
        "encrypted_seed": "base64...",
        "version": 1,
        "created_at": "2024-01-01T00:00:00Z",
        "rotated_at": null
    }
}
```

## Limitations

### Research Prototype

This is a research prototype demonstrating concepts. For production:

1. Use Hardware Security Module (HSM) for key storage
2. Implement proper access control
3. Add audit logging
4. Use secure enclaves for matching

### Known Weaknesses

1. **Local KMS**: File-based storage is not as secure as HSM
2. **Single passphrase**: All users share KMS encryption key
3. **No liveness detection**: Assumes genuine face presentation
4. **Fixed embedding model**: Tied to 512-dimensional FaceNet

### Future Improvements

1. HSM/TPM integration
2. Per-user encryption keys
3. Distributed key management
4. Template update without re-enrollment
5. Secure multi-party matching

## Compliance Notes

### GDPR Considerations

- Biometric data is "special category" under GDPR
- Templates may still be considered biometric data
- Key rotation supports right to erasure
- Irreversibility supports data minimization

### ISO/IEC 24745

This implementation aligns with ISO/IEC 24745 principles:
- Irreversibility (Section 5.2)
- Renewability/Cancelability (Section 5.3)
- Unlinkability (Section 5.4)

## References

1. ISO/IEC 24745:2011 "Biometric template protection"
2. NIST SP 800-76-2 "Biometric Specifications for PIV"
3. Article 9 GDPR "Processing of special categories of data"
