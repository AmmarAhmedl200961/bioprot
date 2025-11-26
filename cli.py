#!/usr/bin/env python3
"""
Command-line interface for biometric template protection.

Commands:
- enroll: Enroll a user with a protected template
- verify: Verify a probe against enrolled template
- rotate: Rotate user's seed (invalidates old templates)
- inspect: Inspect template metadata
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

from protect import (
    protect_embedding, match_protected, serialize_template, 
    deserialize_template
)
from kms_sim import LocalKMS, get_or_create_seed


# ============================================================================
# Embedding utilities
# ============================================================================

def load_embedding_from_file(path: Path) -> np.ndarray:
    """
    Load embedding from .npy file.
    
    Args:
        path: Path to .npy file
        
    Returns:
        Embedding vector
    """
    return np.load(path)


def compute_embedding_from_image(image_path: Path, use_gpu: bool = True) -> np.ndarray:
    """
    Compute face embedding from image using facenet-pytorch.
    
    Args:
        image_path: Path to image file
        use_gpu: Whether to use GPU if available
        
    Returns:
        Embedding vector
    """
    try:
        from facenet_pytorch import MTCNN, InceptionResnetV1
        from PIL import Image
        import torch
        
        # Setup device
        device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        
        # Load models
        mtcnn = MTCNN(device=device, keep_all=False)
        resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
        
        # Load and detect face
        img = Image.open(image_path)
        face = mtcnn(img)
        
        if face is None:
            raise ValueError(f"No face detected in {image_path}")
        
        # Compute embedding
        face = face.unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = resnet(face)
        
        return embedding.cpu().numpy().flatten()
        
    except ImportError:
        raise ImportError(
            "facenet-pytorch not available. Install with: pip install facenet-pytorch\n"
            "Or use --embedding to provide precomputed embeddings."
        )


def get_embedding(image_path: Optional[Path] = None, 
                 embedding_path: Optional[Path] = None,
                 use_gpu: bool = True) -> np.ndarray:
    """
    Get embedding from either image or embedding file.
    
    Args:
        image_path: Path to image file (optional)
        embedding_path: Path to .npy embedding file (optional)
        use_gpu: Whether to use GPU for face detection/recognition
        
    Returns:
        Embedding vector
    """
    if embedding_path:
        return load_embedding_from_file(embedding_path)
    elif image_path:
        return compute_embedding_from_image(image_path, use_gpu=use_gpu)
    else:
        raise ValueError("Must provide either --image or --embedding")


# ============================================================================
# Template storage
# ============================================================================

def get_template_path(user_id: str, templates_dir: Path) -> Path:
    """Get path for user's template file."""
    return templates_dir / f"{user_id}.json"


def save_template(template_json: str, user_id: str, templates_dir: Path):
    """Save template to file."""
    templates_dir.mkdir(parents=True, exist_ok=True)
    template_path = get_template_path(user_id, templates_dir)
    with open(template_path, 'w') as f:
        f.write(template_json)


def load_template(user_id: str, templates_dir: Path) -> dict:
    """Load template from file."""
    template_path = get_template_path(user_id, templates_dir)
    if not template_path.exists():
        raise FileNotFoundError(f"No template found for user '{user_id}'")
    
    with open(template_path, 'r') as f:
        return deserialize_template(f.read())


# ============================================================================
# Commands
# ============================================================================

def cmd_enroll(args):
    """Enroll a user with protected template."""
    # Initialize KMS
    kms = LocalKMS(
        store_path=args.kms_path if args.kms_path else None,
        passphrase=args.kms_passphrase
    )
    
    # Get or create seed
    try:
        seed = get_or_create_seed(kms, args.user)
        seed_version = kms.get_user_version(args.user)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    
    # Get embedding
    try:
        embedding = get_embedding(
            image_path=Path(args.image) if args.image else None,
            embedding_path=Path(args.embedding) if args.embedding else None,
            use_gpu=not args.no_gpu
        )
    except Exception as e:
        print(f"Error loading embedding: {e}", file=sys.stderr)
        return 1
    
    # Protect embedding
    method_kwargs = {}
    if args.method == "permlut":
        method_kwargs = {
            "group_size": args.group_size,
            "n_bins": args.n_bins,
            "bits_per_bin": args.bits_per_bin
        }
    
    template_data = protect_embedding(embedding, seed, method=args.method, **method_kwargs)
    
    # Serialize and save
    template_json = serialize_template(template_data, args.user, seed_version=seed_version)
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(template_json)
        print(f"Template saved to: {args.output}")
    else:
        save_template(template_json, args.user, Path(args.templates_dir))
        template_path = get_template_path(args.user, Path(args.templates_dir))
        print(f"Enrolled user '{args.user}' with {args.method} method")
        print(f"Template saved to: {template_path}")
    
    return 0


def cmd_verify(args):
    """Verify a probe against enrolled template."""
    # Initialize KMS
    kms = LocalKMS(
        store_path=args.kms_path if args.kms_path else None,
        passphrase=args.kms_passphrase
    )
    
    # Get seed
    seed = kms.get_user_seed(args.user)
    if seed is None:
        print(f"Error: User '{args.user}' not found in KMS", file=sys.stderr)
        return 1
    
    # Load enrolled template
    try:
        enrolled_template = load_template(args.user, Path(args.templates_dir))
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    
    # Get probe embedding
    try:
        probe_embedding = get_embedding(
            image_path=Path(args.image) if args.image else None,
            embedding_path=Path(args.embedding) if args.embedding else None,
            use_gpu=not args.no_gpu
        )
    except Exception as e:
        print(f"Error loading probe embedding: {e}", file=sys.stderr)
        return 1
    
    # Protect probe embedding with same method
    method = enrolled_template["method"]
    method_kwargs = {}
    if method == "permlut":
        params = enrolled_template["params"]
        method_kwargs = {
            "group_size": params.get("group_size", 8),
            "n_bins": params.get("n_bins", 4),
            "bits_per_bin": params.get("bits_per_bin", 1)
        }
    
    probe_template = protect_embedding(probe_embedding, seed, method=method, **method_kwargs)
    
    # Match templates
    score = match_protected(enrolled_template, probe_template)
    
    # Determine verdict
    threshold = args.threshold
    verdict = "MATCH" if score >= threshold else "NO MATCH"
    
    print(f"Verification Result:")
    print(f"  User: {args.user}")
    print(f"  Method: {method}")
    print(f"  Score: {score:.4f}")
    print(f"  Threshold: {threshold:.4f}")
    print(f"  Verdict: {verdict}")
    
    return 0 if score >= threshold else 1


def cmd_rotate(args):
    """Rotate user's seed (invalidates old templates)."""
    # Initialize KMS
    kms = LocalKMS(
        store_path=args.kms_path if args.kms_path else None,
        passphrase=args.kms_passphrase
    )
    
    # Rotate key
    try:
        new_seed = kms.rotate_user_key(args.user)
        new_version = kms.get_user_version(args.user)
        print(f"Rotated seed for user '{args.user}'")
        print(f"New seed version: {new_version}")
        print(f"WARNING: Old templates are now invalid. User must re-enroll.")
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    
    return 0


def cmd_inspect(args):
    """Inspect template metadata."""
    if args.template:
        # Load from specified file
        with open(args.template, 'r') as f:
            template = deserialize_template(f.read())
    else:
        # Load from templates directory
        try:
            template = load_template(args.user, Path(args.templates_dir))
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
    
    # Display metadata
    print(f"Template Information:")
    print(f"  User ID: {template['user_id']}")
    print(f"  Method: {template['method']}")
    print(f"  Created: {template.get('created_at', 'unknown')}")
    print(f"  Parameters:")
    for key, value in template['params'].items():
        print(f"    {key}: {value}")
    
    # Show template size
    template_b64 = template['template_b64']
    print(f"  Template size: {len(template_b64)} chars (base64), ~{len(template_b64) * 3 // 4} bytes")
    
    return 0


# ============================================================================
# Main CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Biometric template protection CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Common arguments
    kms_group = argparse.ArgumentParser(add_help=False)
    kms_group.add_argument('--kms-path', type=str, help='Path to KMS store file')
    kms_group.add_argument('--kms-passphrase', type=str, 
                          help='KMS passphrase (or set BIOPROT_KMS_PASSPHRASE env var)')
    
    embedding_group = argparse.ArgumentParser(add_help=False)
    embedding_group.add_argument('--image', type=str, help='Path to face image')
    embedding_group.add_argument('--embedding', type=str, help='Path to precomputed embedding (.npy)')
    embedding_group.add_argument('--no-gpu', action='store_true', help='Disable GPU for face detection')
    
    templates_group = argparse.ArgumentParser(add_help=False)
    templates_group.add_argument('--templates-dir', type=str, default='templates',
                                help='Directory for template storage (default: templates/)')
    
    # Enroll command
    enroll_parser = subparsers.add_parser('enroll', parents=[kms_group, embedding_group, templates_group],
                                          help='Enroll a user')
    enroll_parser.add_argument('--user', required=True, help='User ID')
    enroll_parser.add_argument('--method', choices=['ortho', 'permlut'], default='ortho',
                              help='Protection method (default: ortho)')
    enroll_parser.add_argument('--output', type=str, help='Output template file (default: templates/<user>.json)')
    enroll_parser.add_argument('--group-size', type=int, default=8, 
                              help='Group size for permlut (default: 8)')
    enroll_parser.add_argument('--n-bins', type=int, default=4,
                              help='Number of bins for permlut (default: 4)')
    enroll_parser.add_argument('--bits-per-bin', type=int, default=1,
                              help='Bits per bin for permlut (default: 1)')
    
    # Verify command
    verify_parser = subparsers.add_parser('verify', parents=[kms_group, embedding_group, templates_group],
                                          help='Verify a probe')
    verify_parser.add_argument('--user', required=True, help='User ID to verify against')
    verify_parser.add_argument('--threshold', type=float, default=0.80,
                              help='Similarity threshold (default: 0.80)')
    
    # Rotate command
    rotate_parser = subparsers.add_parser('rotate', parents=[kms_group],
                                          help='Rotate user seed')
    rotate_parser.add_argument('--user', required=True, help='User ID')
    
    # Inspect command
    inspect_parser = subparsers.add_parser('inspect', parents=[templates_group],
                                           help='Inspect template')
    inspect_parser.add_argument('--user', type=str, help='User ID (loads from templates dir)')
    inspect_parser.add_argument('--template', type=str, help='Path to template JSON file')
    
    # Parse and execute
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Execute command
    if args.command == 'enroll':
        return cmd_enroll(args)
    elif args.command == 'verify':
        return cmd_verify(args)
    elif args.command == 'rotate':
        return cmd_rotate(args)
    elif args.command == 'inspect':
        return cmd_inspect(args)
    else:
        print(f"Unknown command: {args.command}", file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
