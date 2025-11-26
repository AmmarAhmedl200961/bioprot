"""
Generate sample face embeddings for testing and demonstration.

This script creates synthetic embeddings that mimic facenet-pytorch output.
"""

import numpy as np
import json
from pathlib import Path


def generate_sample_embeddings(output_dir: Path, 
                               n_users: int = 5,
                               samples_per_user: int = 3,
                               embedding_dim: int = 512,
                               seed: int = 42):
    """
    Generate synthetic face embeddings for testing.
    
    Args:
        output_dir: Directory to save embeddings
        n_users: Number of users
        samples_per_user: Number of samples per user
        embedding_dim: Dimension of embeddings
        seed: Random seed for reproducibility
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    rng = np.random.default_rng(seed)
    
    metadata = {
        "description": "Synthetic face embeddings for testing",
        "n_users": n_users,
        "samples_per_user": samples_per_user,
        "embedding_dim": embedding_dim,
        "users": []
    }
    
    user_names = [f"user{i+1:02d}" for i in range(n_users)]
    
    for user_id in user_names:
        # Generate a base embedding for this user
        base_embedding = rng.normal(0, 1, size=embedding_dim)
        base_embedding = base_embedding / np.linalg.norm(base_embedding)
        
        user_files = []
        
        # Generate variations (same person, slight variations)
        for sample_idx in range(samples_per_user):
            # Add small random noise to create variation
            # Use smaller noise (0.02) for realistic intra-class variance
            noise = rng.normal(0, 0.02, size=embedding_dim)
            embedding = base_embedding + noise
            
            # Normalize
            embedding = embedding / np.linalg.norm(embedding)
            
            # Save
            filename = f"{user_id}_{sample_idx+1:02d}.npy"
            filepath = output_dir / filename
            np.save(filepath, embedding)
            
            user_files.append(filename)
        
        metadata["users"].append({
            "user_id": user_id,
            "files": user_files
        })
    
    # Save metadata
    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Generated {n_users * samples_per_user} sample embeddings")
    print(f"Saved to: {output_dir}")
    print(f"Users: {', '.join(user_names)}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate sample embeddings")
    parser.add_argument("--output", type=str, default="embeddings",
                       help="Output directory (default: embeddings/)")
    parser.add_argument("--n-users", type=int, default=5,
                       help="Number of users (default: 5)")
    parser.add_argument("--samples", type=int, default=3,
                       help="Samples per user (default: 3)")
    parser.add_argument("--dim", type=int, default=512,
                       help="Embedding dimension (default: 512)")
    
    args = parser.parse_args()
    
    generate_sample_embeddings(
        Path(args.output),
        n_users=args.n_users,
        samples_per_user=args.samples,
        embedding_dim=args.dim
    )
