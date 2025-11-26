**Project Proposal: Two Lightweight Information-Security Measures for a Facial-Recognition Prototype**

---

### 1. Introduction and Motivation
Facial recognition systems rely on deep neural network embeddings that encode individual identity. However, storing raw embeddings poses serious security and privacy risks. If a database of embeddings is compromised, users' biometric information can be exposed, and unlike passwords, biometric data cannot be changed.

This project proposes a minimal yet practical prototype that demonstrates how two lightweight, cancelable-template protection techniques can secure face embeddings. The goal is to provide a pluggable protection layer that can be applied to any deep learning face recognition stack (default: `facenet-pytorch`).

---

### 2. Project Goal
Build a **prototype** that implements two template protection schemes:
1. **Orthonormal Projection + Sign Binarization (Ortho+Sign)**
2. **Permutation + Group-wise LUT Quantization (Perm+LUT)**

The system will demonstrate **enrollment**, **verification**, **revocation**, and **irreversibility checks** using simple evaluation scripts. The resulting codebase should be modular, compact, and adaptable.

---

### 3. Objectives
- Implement two cancelable-template methods for face embeddings.
- Design a local Key Management System (KMS) simulator to securely manage user-specific seeds.
- Provide CLI tools for enrollment, verification, and key rotation.
- Evaluate the performance trade-offs between security and recognition accuracy.

---

### 4. Proposed Security Measures

#### 4.1 Ortho+Sign Method
**Process:**
1. Each user gets a deterministic seed from KMS.
2. The seed generates a random orthonormal matrix \( Q \) via QR decomposition.
3. Embedding \( x \) is projected to \( y = Qx \).
4. Binarization: \( b_i = 1 \text{ if } y_i \ge 0 \text{ else } 0 \).

**Advantages:**
- Templates are irreversible without the user-specific seed.
- Simple key rotation (seed change) instantly invalidates old templates.
- Low computational cost.

**Revocation:** Change the user seed in KMS, generating a new \( Q \) and therefore a new binary template.

---

#### 4.2 Perm+LUT Method
**Process:**
1. Generate a per-user permutation \( P \) and lookup table (LUT) from the seed.
2. Normalize and quantize embedding into discrete bins.
3. Apply LUT-based mapping and permutation.

**Advantages:**
- Fast and memory-efficient.
- Simple revocation and regeneration.
- Configurable trade-off between accuracy and security.

**Revocation:** Rotate user seed to produce new \( P \) and LUT mappings.

---

### 5. System Architecture

| Component | Description |
|------------|-------------|
| **protect.py** | Implements both protection methods via a unified API for embedding transformation and matching. |
| **kms_sim.py** | Simulates a local, encrypted KMS using file-backed secure key storage. |
| **CLI Tool** | Supports enrollment, verification, and key rotation. |
| **evaluate.py** | Measures accuracy (ROC/AUC) and irreversibility metrics. |

**High-level flow:**
1. User enrolls via CLI → embedding computed → protect.py transforms it → template stored.
2. Verification uses the same seed to transform probe embedding and match.
3. Key rotation regenerates new templates.

---

### 6. Evaluation Metrics
- **Accuracy:** ROC/AUC, TAR @ FAR (1e-2, 1e-3, 1e-4)
- **Template Security:** Irreversibility, revocability
- **Efficiency:** Template size, computation time
- **Usability:** Integration ease with facenet-pytorch or other models

**Datasets:** Lightweight subset of LFW or any small dataset with multiple face images per identity.

---

### 7. Threat Model
**Threats Mitigated:**
- Template leakage from database.
- Cross-matching of templates across databases.

**Assumptions:**
- KMS is secure; if compromised, templates can be regenerated.
- Attacks such as malware or physical access are out of scope.

---

### 8. Revocation and Key Lifecycle
- **Rotate Key:** Update user seed in KMS.
- **Re-enroll:** New template generated using new seed.
- **Migration:** Support versioned templates during transition.

---

### 9. Implementation Plan (5-Day Schedule)
| Day | Tasks |
|-----|-------|
| 1 | Finalize design parameters, repo structure, implement Ortho+Sign. |
| 2 | Implement KMS simulator with encryption and key rotation. |
| 3 | Integrate CLI for enroll/verify/rotate-key. |
| 4 | Add Perm+LUT and unit tests. |
| 5 | Implement evaluation metrics, prepare report and results. |

---

### 10. Expected Deliverables
- `protect.py`: Protection algorithms.
- `kms_sim.py`: Local encrypted KMS.
- CLI scripts for enrollment, verification, key rotation.
- `evaluate.py`: Evaluation scripts.
- Documentation and example usage.

---

### 11. Risks and Mitigation
| Risk | Mitigation |
|------|-------------|
| Accuracy degradation | Tune quantization parameters and thresholds. |
| KMS compromise | Highlight best practices, migrate to HSM-ready design. |
| Determinism bugs | Unit testing for consistent seed-to-template generation. |

---

### 12. Conclusion
This project will demonstrate two practical, cancelable-template security mechanisms that can be integrated into any face recognition pipeline. The resulting prototype will showcase how simple mathematical transformations and lightweight key management can add strong protection and revocability to biometric systems without sacrificing usability.

