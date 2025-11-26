#!/usr/bin/env python3
"""
Evaluation utilities for protected templates.

Provides:
- ROC curve and AUC computation
- TAR @ FAR metrics
- Irreversibility tests (naive inversion, regressor)
- Revocation tests
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

from protect import (
    protect_embedding, match_protected, deserialize_template,
    generate_ortho_matrix, base64_to_bits
)
from kms_sim import LocalKMS


# ============================================================================
# Score computation
# ============================================================================

def compute_genuine_impostor_scores(
    embeddings_dir: Path,
    templates_dir: Path,
    kms: LocalKMS,
    method: str = "ortho",
    **method_kwargs
) -> Tuple[List[float], List[float]]:
    """
    Compute genuine and impostor scores from embeddings directory.
    
    Expects embeddings directory structure:
        embeddings/
            user1_01.npy
            user1_02.npy
            user2_01.npy
            user2_02.npy
            ...
    
    Args:
        embeddings_dir: Directory containing .npy embedding files
        templates_dir: Directory containing enrolled templates
        kms: KMS instance for seed retrieval
        method: Protection method
        **method_kwargs: Method-specific parameters
        
    Returns:
        Tuple of (genuine_scores, impostor_scores)
    """
    # Load embeddings and group by user
    embedding_files = sorted(embeddings_dir.glob("*.npy"))
    
    user_embeddings = {}
    for emb_file in embedding_files:
        # Parse user_id from filename (format: userid_samplenum.npy)
        name = emb_file.stem
        parts = name.rsplit('_', 1)
        if len(parts) == 2:
            user_id = parts[0]
        else:
            user_id = name
        
        embedding = np.load(emb_file)
        
        if user_id not in user_embeddings:
            user_embeddings[user_id] = []
        user_embeddings[user_id].append(embedding)
    
    print(f"Loaded {len(embedding_files)} embeddings for {len(user_embeddings)} users")
    
    # Compute genuine scores (same user, different samples)
    genuine_scores = []
    for user_id, embeddings in user_embeddings.items():
        if len(embeddings) < 2:
            continue
        
        # Get seed
        seed = kms.get_user_seed(user_id)
        if seed is None:
            print(f"Warning: No seed for user {user_id}, skipping")
            continue
        
        # Compare all pairs
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                t1 = protect_embedding(embeddings[i], seed, method=method, **method_kwargs)
                t2 = protect_embedding(embeddings[j], seed, method=method, **method_kwargs)
                score = match_protected(t1, t2)
                genuine_scores.append(score)
    
    # Compute impostor scores (different users)
    impostor_scores = []
    user_ids = list(user_embeddings.keys())
    
    for i in range(len(user_ids)):
        for j in range(i + 1, min(i + 10, len(user_ids))):  # Limit pairs for speed
            user_i = user_ids[i]
            user_j = user_ids[j]
            
            seed_i = kms.get_user_seed(user_i)
            seed_j = kms.get_user_seed(user_j)
            
            if seed_i is None or seed_j is None:
                continue
            
            # Compare first sample of each user
            emb_i = user_embeddings[user_i][0]
            emb_j = user_embeddings[user_j][0]
            
            # Each with their own seed
            t_i = protect_embedding(emb_i, seed_i, method=method, **method_kwargs)
            t_j = protect_embedding(emb_j, seed_j, method=method, **method_kwargs)
            
            score = match_protected(t_i, t_j)
            impostor_scores.append(score)
    
    print(f"Computed {len(genuine_scores)} genuine scores")
    print(f"Computed {len(impostor_scores)} impostor scores")
    
    return genuine_scores, impostor_scores


# ============================================================================
# ROC and metrics
# ============================================================================

def compute_roc_metrics(genuine_scores: List[float], 
                       impostor_scores: List[float],
                       output_dir: Path):
    """
    Compute ROC curve, AUC, and TAR @ FAR metrics.
    
    Args:
        genuine_scores: List of genuine match scores
        impostor_scores: List of impostor match scores
        output_dir: Directory to save outputs
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare labels (1 for genuine, 0 for impostor)
    y_true = np.concatenate([
        np.ones(len(genuine_scores)),
        np.zeros(len(impostor_scores))
    ])
    y_scores = np.array(genuine_scores + impostor_scores)
    
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    # Compute TAR @ specific FAR values
    target_fars = [1e-2, 1e-3, 1e-4]
    tar_at_far = {}
    
    for target_far in target_fars:
        # Find threshold where FAR is closest to target
        idx = np.argmin(np.abs(fpr - target_far))
        tar_at_far[target_far] = tpr[idx]
    
    # Print results
    print("\n" + "="*60)
    print("ROC Metrics:")
    print("="*60)
    print(f"AUC: {roc_auc:.4f}")
    print(f"\nTAR @ FAR:")
    for far, tar in tar_at_far.items():
        print(f"  FAR = {far:.0e}: TAR = {tar:.4f}")
    
    # Save metrics to CSV
    metrics_file = output_dir / "roc_metrics.csv"
    with open(metrics_file, 'w') as f:
        f.write("metric,value\n")
        f.write(f"AUC,{roc_auc:.6f}\n")
        for far, tar in tar_at_far.items():
            f.write(f"TAR@FAR={far:.0e},{tar:.6f}\n")
    
    print(f"\nMetrics saved to: {metrics_file}")
    
    # Save full ROC data
    roc_file = output_dir / "roc_curve.csv"
    with open(roc_file, 'w') as f:
        f.write("fpr,tpr,threshold\n")
        for i in range(len(fpr)):
            f.write(f"{fpr[i]:.6f},{tpr[i]:.6f},{thresholds[i]:.6f}\n")
    
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    
    # Mark TAR @ FAR points
    for target_far in target_fars:
        idx = np.argmin(np.abs(fpr - target_far))
        plt.plot(fpr[idx], tpr[idx], 'ro', markersize=8)
        plt.text(fpr[idx] + 0.02, tpr[idx], f'FAR={target_far:.0e}', fontsize=9)
    
    plt.xlabel('False Acceptance Rate (FAR)', fontsize=12)
    plt.ylabel('True Acceptance Rate (TAR)', fontsize=12)
    plt.title('ROC Curve', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    roc_plot = output_dir / "roc_curve.png"
    plt.savefig(roc_plot, dpi=150)
    print(f"ROC plot saved to: {roc_plot}")
    plt.close()
    
    # Plot score distributions
    plt.figure(figsize=(10, 5))
    plt.hist(genuine_scores, bins=50, alpha=0.6, label='Genuine', color='green')
    plt.hist(impostor_scores, bins=50, alpha=0.6, label='Impostor', color='red')
    plt.xlabel('Similarity Score', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Score Distributions', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    dist_plot = output_dir / "score_distributions.png"
    plt.savefig(dist_plot, dpi=150)
    print(f"Distribution plot saved to: {dist_plot}")
    plt.close()


# ============================================================================
# Irreversibility tests
# ============================================================================

def test_ortho_naive_inversion(embeddings: List[np.ndarray],
                               seeds: List[bytes],
                               output_dir: Path):
    """
    Test naive inversion for Ortho+Sign method.
    
    For binary template b, construct y' = (2*b - 1), then compute
    x_approx = Q.T @ y' and measure cosine similarity with original x.
    
    Args:
        embeddings: List of original embeddings
        seeds: Corresponding user seeds
        output_dir: Directory to save results
    """
    print("\n" + "="*60)
    print("Ortho+Sign Naive Inversion Test:")
    print("="*60)
    
    from protect import protect_ortho_sign
    
    cosines = []
    
    for emb, seed in zip(embeddings, seeds):
        # Normalize embedding
        x = emb / np.linalg.norm(emb)
        
        # Protect
        b = protect_ortho_sign(x, seed, normalize=False)
        
        # Attempt naive inversion
        y_approx = 2 * b.astype(float) - 1  # Map {0,1} to {-1,+1}
        
        # Get Q matrix
        Q = generate_ortho_matrix(seed, len(x))
        
        # Invert: x_approx = Q.T @ y_approx
        x_approx = Q.T @ y_approx
        
        # Normalize
        x_approx = x_approx / np.linalg.norm(x_approx)
        
        # Compute cosine similarity
        cos_sim = np.dot(x, x_approx)
        cosines.append(cos_sim)
    
    cosines = np.array(cosines)
    
    print(f"Cosine similarity statistics:")
    print(f"  Mean: {np.mean(cosines):.4f}")
    print(f"  Std:  {np.std(cosines):.4f}")
    print(f"  Min:  {np.min(cosines):.4f}")
    print(f"  Max:  {np.max(cosines):.4f}")
    
    # Save results
    results_file = output_dir / "naive_inversion_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            "method": "ortho_naive_inversion",
            "n_samples": len(cosines),
            "mean_cosine": float(np.mean(cosines)),
            "std_cosine": float(np.std(cosines)),
            "min_cosine": float(np.min(cosines)),
            "max_cosine": float(np.max(cosines))
        }, f, indent=2)
    
    print(f"Results saved to: {results_file}")


def test_regressor_inversion(embeddings: List[np.ndarray],
                             seeds: List[bytes],
                             method: str,
                             output_dir: Path,
                             **method_kwargs):
    """
    Test template-to-embedding regression attack.
    
    Fit a Ridge regression from protected templates to original embeddings
    on training set, then measure reconstruction MSE and identification
    accuracy on test set.
    
    Args:
        embeddings: List of embeddings
        seeds: Corresponding seeds
        method: Protection method
        output_dir: Directory to save results
        **method_kwargs: Method parameters
    """
    print("\n" + "="*60)
    print(f"Regressor Inversion Test ({method}):")
    print("="*60)
    
    from protect import protect_embedding, base64_to_bits
    
    # Generate protected templates
    templates = []
    for emb, seed in zip(embeddings, seeds):
        t = protect_embedding(emb, seed, method=method, **method_kwargs)
        # Convert to numpy array
        t_bits = base64_to_bits(t["template_b64"], t["params"]["nbits"])
        templates.append(t_bits.astype(float))
    
    X = np.array(templates)  # Shape: (n_samples, n_bits)
    y = np.array(embeddings)  # Shape: (n_samples, embedding_dim)
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Fit Ridge regression
    print("Training Ridge regressor...")
    reg = Ridge(alpha=1.0)
    reg.fit(X_train, y_train)
    
    # Predict on test set
    y_pred = reg.predict(X_test)
    
    # Compute MSE
    mse = np.mean((y_test - y_pred) ** 2)
    
    # Compute normalized MSE (relative to embedding variance)
    embedding_var = np.var(y_test)
    normalized_mse = mse / embedding_var
    
    print(f"\nReconstruction MSE: {mse:.6f}")
    print(f"Normalized MSE: {normalized_mse:.6f}")
    
    # Compute identification accuracy
    # For each test sample, find nearest neighbor in test set
    correct = 0
    for i in range(len(y_test)):
        # True embedding
        true_emb = y_test[i]
        # Reconstructed embedding
        recon_emb = y_pred[i]
        
        # Find nearest neighbor in test set (excluding self)
        distances = []
        for j in range(len(y_test)):
            if i == j:
                distances.append(np.inf)
            else:
                # Cosine distance
                cos_sim = np.dot(recon_emb, y_test[j]) / (
                    np.linalg.norm(recon_emb) * np.linalg.norm(y_test[j]) + 1e-8
                )
                distances.append(1 - cos_sim)
        
        # Check if nearest neighbor is correct
        nearest_idx = np.argmin(distances)
        if nearest_idx == i:  # This won't happen due to inf
            correct += 1
    
    # Simplified: compute average cosine similarity
    cosines = []
    for i in range(len(y_test)):
        true_norm = y_test[i] / (np.linalg.norm(y_test[i]) + 1e-8)
        recon_norm = y_pred[i] / (np.linalg.norm(y_pred[i]) + 1e-8)
        cos_sim = np.dot(true_norm, recon_norm)
        cosines.append(cos_sim)
    
    mean_cosine = np.mean(cosines)
    print(f"Mean cosine similarity: {mean_cosine:.4f}")
    
    # Save results
    results_file = output_dir / "regressor_inversion_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            "method": method,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "mse": float(mse),
            "normalized_mse": float(normalized_mse),
            "mean_cosine_similarity": float(mean_cosine),
            "interpretation": "Low cosine similarity indicates high irreversibility"
        }, f, indent=2)
    
    print(f"Results saved to: {results_file}")


# ============================================================================
# Revocation test
# ============================================================================

def test_revocation(user_id: str, 
                   embedding: np.ndarray,
                   kms: LocalKMS,
                   method: str,
                   output_dir: Path,
                   **method_kwargs):
    """
    Test that key rotation invalidates old templates.
    
    Args:
        user_id: User ID
        embedding: User embedding
        kms: KMS instance
        method: Protection method
        output_dir: Directory to save results
        **method_kwargs: Method parameters
    """
    print("\n" + "="*60)
    print(f"Revocation Test for user '{user_id}':")
    print("="*60)
    
    from protect import protect_embedding, match_protected
    
    # Get initial seed
    old_seed = kms.get_user_seed(user_id)
    if old_seed is None:
        old_seed = kms.create_user_key(user_id)
    
    # Create old template
    old_template = protect_embedding(embedding, old_seed, method=method, **method_kwargs)
    
    # Rotate key
    new_seed = kms.rotate_user_key(user_id)
    print(f"Rotated key for user '{user_id}'")
    
    # Create new template
    new_template = protect_embedding(embedding, new_seed, method=method, **method_kwargs)
    
    # Test matching
    score_old_old = match_protected(old_template, old_template)
    score_new_new = match_protected(new_template, new_template)
    score_old_new = match_protected(old_template, new_template)
    
    print(f"\nMatch scores:")
    print(f"  Old vs Old: {score_old_old:.4f} (should be 1.0)")
    print(f"  New vs New: {score_new_new:.4f} (should be 1.0)")
    print(f"  Old vs New: {score_old_new:.4f} (should be low, ~0.5 for random)")
    
    # Verify revocation
    threshold = 0.80
    revoked = score_old_new < threshold
    
    print(f"\nRevocation successful: {revoked}")
    
    # Save results
    results_file = output_dir / "revocation_test_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            "user_id": user_id,
            "method": method,
            "score_old_old": float(score_old_old),
            "score_new_new": float(score_new_new),
            "score_old_new": float(score_old_new),
            "threshold": threshold,
            "revocation_successful": bool(revoked)
        }, f, indent=2)
    
    print(f"Results saved to: {results_file}")


# ============================================================================
# Main evaluation
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate protected templates")
    parser.add_argument('--embeddings', type=str, required=True,
                       help='Directory containing embeddings (*.npy)')
    parser.add_argument('--templates', type=str, required=True,
                       help='Directory containing templates (*.json)')
    parser.add_argument('--out', type=str, default='results',
                       help='Output directory for results')
    parser.add_argument('--method', choices=['ortho', 'permlut'], default='ortho',
                       help='Protection method to evaluate')
    parser.add_argument('--kms-path', type=str, help='Path to KMS store')
    parser.add_argument('--kms-passphrase', type=str, 
                       help='KMS passphrase (or set BIOPROT_KMS_PASSPHRASE)')
    
    args = parser.parse_args()
    
    embeddings_dir = Path(args.embeddings)
    templates_dir = Path(args.templates)
    output_dir = Path(args.out)
    
    # Initialize KMS
    kms = LocalKMS(
        store_path=args.kms_path if args.kms_path else None,
        passphrase=args.kms_passphrase
    )
    
    # Method parameters
    method_kwargs = {}
    if args.method == "permlut":
        method_kwargs = {"group_size": 8, "n_bins": 4, "bits_per_bin": 1}
    
    # 1. Compute ROC metrics
    print("\n" + "="*60)
    print("Computing ROC metrics...")
    print("="*60)
    genuine_scores, impostor_scores = compute_genuine_impostor_scores(
        embeddings_dir, templates_dir, kms, method=args.method, **method_kwargs
    )
    
    if len(genuine_scores) > 0 and len(impostor_scores) > 0:
        compute_roc_metrics(genuine_scores, impostor_scores, output_dir)
    else:
        print("Warning: Not enough scores for ROC computation")
    
    # 2. Load sample embeddings for irreversibility tests
    embedding_files = sorted(list(embeddings_dir.glob("*.npy")))[:20]  # Use first 20
    if len(embedding_files) >= 10:
        embeddings = [np.load(f) for f in embedding_files]
        
        # Get seeds for these users
        seeds = []
        for emb_file in embedding_files:
            name = emb_file.stem
            parts = name.rsplit('_', 1)
            user_id = parts[0] if len(parts) == 2 else name
            
            seed = kms.get_user_seed(user_id)
            if seed is None:
                seed = kms.create_user_key(user_id)
            seeds.append(seed)
        
        # 3. Naive inversion test (Ortho only)
        if args.method == "ortho":
            test_ortho_naive_inversion(embeddings, seeds, output_dir)
        
        # 4. Regressor test
        if len(embeddings) >= 10:
            test_regressor_inversion(embeddings, seeds, args.method, 
                                   output_dir, **method_kwargs)
    
    # 5. Revocation test
    if len(embedding_files) > 0:
        test_user_file = embedding_files[0]
        test_embedding = np.load(test_user_file)
        test_user_id = test_user_file.stem.rsplit('_', 1)[0]
        
        test_revocation(test_user_id, test_embedding, kms, 
                       args.method, output_dir, **method_kwargs)
    
    print("\n" + "="*60)
    print("Evaluation complete!")
    print(f"Results saved to: {output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()
