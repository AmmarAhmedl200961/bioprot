#!/usr/bin/env python3
"""
Gradio web interface for BioProt - Biometric Template Protection.

Works in GitHub Codespaces and remote environments where camera access is not available.
Uses image upload for enrollment and verification.

Usage:
    python app.py

Then open the provided URL in your browser.
"""

import os
import sys
import json
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, List

import numpy as np

# Check for Gradio
try:
    import gradio as gr
except ImportError:
    print("Gradio not installed. Installing...")
    os.system("pip install gradio")
    import gradio as gr

# Check for face recognition dependencies
try:
    import torch
    from facenet_pytorch import MTCNN, InceptionResnetV1
    from PIL import Image
    FACENET_AVAILABLE = True
except ImportError:
    FACENET_AVAILABLE = False
    print("Warning: facenet-pytorch not available. Install with:")
    print("  pip install facenet-pytorch torch torchvision pillow")

from protect import (
    protect_embedding, match_protected, serialize_template, 
    deserialize_template, generate_seed
)
from kms_sim import LocalKMS, get_or_create_seed


# ============================================================================
# Global state
# ============================================================================

# Initialize models lazily
_mtcnn = None
_resnet = None
_device = None

def get_models():
    """Lazy load face detection and recognition models."""
    global _mtcnn, _resnet, _device
    
    if not FACENET_AVAILABLE:
        raise ImportError("facenet-pytorch not installed")
    
    if _mtcnn is None:
        _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Loading models on {_device}...")
        
        _mtcnn = MTCNN(
            image_size=160,
            margin=20,
            min_face_size=40,
            thresholds=[0.6, 0.7, 0.7],
            device=_device,
            keep_all=False
        )
        
        _resnet = InceptionResnetV1(pretrained='vggface2').eval().to(_device)
        print("Models loaded!")
    
    return _mtcnn, _resnet, _device


def get_kms(passphrase: str) -> LocalKMS:
    """Get KMS instance with passphrase."""
    store_path = Path("kms_store.bin")
    return LocalKMS(store_path=store_path, passphrase=passphrase)


def get_templates_dir() -> Path:
    """Get templates directory."""
    templates_dir = Path("templates")
    templates_dir.mkdir(exist_ok=True)
    return templates_dir


# ============================================================================
# Face processing
# ============================================================================

def extract_embedding(image: np.ndarray) -> Tuple[Optional[np.ndarray], str]:
    """
    Extract face embedding from image.
    
    Args:
        image: RGB image as numpy array
        
    Returns:
        Tuple of (embedding, status_message)
    """
    if not FACENET_AVAILABLE:
        return None, "❌ facenet-pytorch not installed. Run: pip install facenet-pytorch torch torchvision"
    
    try:
        mtcnn, resnet, device = get_models()
        
        # Convert to PIL Image
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        else:
            pil_image = image
        
        # Detect and crop face
        face = mtcnn(pil_image)
        
        if face is None:
            return None, "❌ No face detected in image. Please upload a clear face photo."
        
        # Compute embedding
        face = face.unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = resnet(face)
        
        embedding = embedding.cpu().numpy().flatten()
        
        return embedding, "✅ Face detected and embedding extracted"
        
    except Exception as e:
        return None, f"❌ Error processing image: {str(e)}"


# ============================================================================
# Enrollment
# ============================================================================

def enroll_user(
    image: np.ndarray,
    user_id: str,
    method: str,
    passphrase: str
) -> Tuple[str, str, Optional[str]]:
    """
    Enroll a user from uploaded image.
    
    Returns:
        Tuple of (status, details, template_preview)
    """
    if not user_id or not user_id.strip():
        return "❌ Error", "Please enter a User ID", None
    
    if not passphrase or not passphrase.strip():
        return "❌ Error", "Please enter a KMS passphrase", None
    
    if image is None:
        return "❌ Error", "Please upload an image", None
    
    user_id = user_id.strip()
    passphrase = passphrase.strip()
    
    # Extract embedding
    embedding, status = extract_embedding(image)
    if embedding is None:
        return "❌ Failed", status, None
    
    try:
        # Initialize KMS
        kms = get_kms(passphrase)
        
        # Check if user exists
        existing_seed = kms.get_user_seed(user_id)
        if existing_seed:
            # Use existing seed
            seed = existing_seed
            seed_version = kms.get_user_version(user_id)
            user_status = "existing"
        else:
            # Create new user
            seed = kms.create_user_key(user_id)
            seed_version = 1
            user_status = "new"
        
        # Protect embedding
        method_kwargs = {}
        if method == "permlut":
            method_kwargs = {"group_size": 8, "n_bins": 4, "bits_per_bin": 1}
        
        template_data = protect_embedding(embedding, seed, method=method, **method_kwargs)
        
        # Serialize and save
        template_json = serialize_template(template_data, user_id, seed_version=seed_version)
        
        templates_dir = get_templates_dir()
        template_path = templates_dir / f"{user_id}.json"
        with open(template_path, 'w') as f:
            f.write(template_json)
        
        # Prepare response
        details = f"""
### Enrollment Successful! 🎉

**User ID:** `{user_id}` ({user_status} user)
**Method:** {method}
**Template bits:** {template_data['params']['nbits']}
**Seed version:** {seed_version}
**Saved to:** `{template_path}`
**Timestamp:** {datetime.now().isoformat()}

The protected template has been saved. You can now verify this user.
"""
        
        # Template preview (truncated)
        template_preview = json.dumps(json.loads(template_json), indent=2)
        if len(template_preview) > 500:
            template_preview = template_preview[:500] + "\n..."
        
        return "✅ Enrolled Successfully", details, template_preview
        
    except Exception as e:
        return "❌ Error", f"Enrollment failed: {str(e)}", None


# ============================================================================
# Verification
# ============================================================================

def verify_user(
    image: np.ndarray,
    user_id: str,
    threshold: float,
    passphrase: str
) -> Tuple[str, str, Optional[str]]:
    """
    Verify a user from uploaded image.
    
    Returns:
        Tuple of (result, details, score_display)
    """
    if not user_id or not user_id.strip():
        return "❌ Error", "Please enter a User ID", None
    
    if not passphrase or not passphrase.strip():
        return "❌ Error", "Please enter a KMS passphrase", None
    
    if image is None:
        return "❌ Error", "Please upload an image", None
    
    user_id = user_id.strip()
    passphrase = passphrase.strip()
    
    # Check if template exists
    templates_dir = get_templates_dir()
    template_path = templates_dir / f"{user_id}.json"
    
    if not template_path.exists():
        return "❌ Error", f"No template found for user `{user_id}`. Please enroll first.", None
    
    # Extract embedding
    embedding, status = extract_embedding(image)
    if embedding is None:
        return "❌ Failed", status, None
    
    try:
        # Initialize KMS
        kms = get_kms(passphrase)
        
        # Get seed
        seed = kms.get_user_seed(user_id)
        if seed is None:
            return "❌ Error", f"No seed found for user `{user_id}` in KMS. Check passphrase.", None
        
        # Load enrolled template
        with open(template_path, 'r') as f:
            enrolled_template = deserialize_template(f.read())
        
        # Protect probe embedding
        method = enrolled_template["method"]
        method_kwargs = {}
        if method == "permlut":
            params = enrolled_template["params"]
            method_kwargs = {
                "group_size": params.get("group_size", 8),
                "n_bins": params.get("n_bins", 4),
                "bits_per_bin": params.get("bits_per_bin", 1)
            }
        
        probe_template = protect_embedding(embedding, seed, method=method, **method_kwargs)
        
        # Match
        score = match_protected(enrolled_template, probe_template)
        
        # Determine verdict
        if score >= threshold:
            verdict = "✅ MATCH"
            verdict_color = "green"
            emoji = "🟢"
        else:
            verdict = "❌ NO MATCH"
            verdict_color = "red"
            emoji = "🔴"
        
        # Score visualization
        score_bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        threshold_pos = int(threshold * 20)
        
        details = f"""
### Verification Result: {verdict}

**User ID:** `{user_id}`
**Method:** {method}
**Score:** `{score:.4f}`
**Threshold:** `{threshold:.4f}`

**Score Visualization:**
```
{score_bar} {score:.2%}
{"─" * threshold_pos}▲{"─" * (19 - threshold_pos)}
{" " * threshold_pos}threshold
```

{emoji} {"Identity verified!" if score >= threshold else "Identity not verified."}
"""
        
        score_display = f"{score:.4f}"
        
        return verdict, details, score_display
        
    except Exception as e:
        return "❌ Error", f"Verification failed: {str(e)}", None


# ============================================================================
# Key Management
# ============================================================================

def rotate_key(user_id: str, passphrase: str) -> Tuple[str, str]:
    """Rotate user's key (revoke old template)."""
    if not user_id or not passphrase:
        return "❌ Error", "Please enter User ID and passphrase"
    
    try:
        kms = get_kms(passphrase.strip())
        
        old_version = kms.get_user_version(user_id.strip())
        if old_version is None:
            return "❌ Error", f"User `{user_id}` not found in KMS"
        
        new_seed = kms.rotate_user_key(user_id.strip())
        new_version = kms.get_user_version(user_id.strip())
        
        details = f"""
### Key Rotated Successfully! 🔄

**User ID:** `{user_id}`
**Old version:** {old_version}
**New version:** {new_version}

⚠️ **Important:** The old template is now invalid!

The user must re-enroll to create a new template with the rotated key.
Any verification attempts with the old template will fail.
"""
        
        return "✅ Key Rotated", details
        
    except Exception as e:
        return "❌ Error", f"Key rotation failed: {str(e)}"


def list_users(passphrase: str) -> str:
    """List all enrolled users."""
    templates_dir = get_templates_dir()
    template_files = list(templates_dir.glob("*.json"))
    
    if not template_files:
        return "No users enrolled yet."
    
    output = "### Enrolled Users\n\n"
    output += "| User ID | Method | Bits | Created |\n"
    output += "|---------|--------|------|--------|\n"
    
    for tf in sorted(template_files):
        try:
            with open(tf, 'r') as f:
                data = json.load(f)
            user_id = data.get("user_id", tf.stem)
            method = data.get("method", "unknown")
            nbits = data.get("params", {}).get("nbits", "?")
            created = data.get("created_at", "unknown")[:10]
            output += f"| `{user_id}` | {method} | {nbits} | {created} |\n"
        except:
            output += f"| `{tf.stem}` | error | ? | ? |\n"
    
    return output


def delete_user(user_id: str, passphrase: str) -> Tuple[str, str]:
    """Delete user template and KMS entry."""
    if not user_id or not passphrase:
        return "❌ Error", "Please enter User ID and passphrase"
    
    user_id = user_id.strip()
    
    try:
        # Delete template
        templates_dir = get_templates_dir()
        template_path = templates_dir / f"{user_id}.json"
        
        template_deleted = False
        if template_path.exists():
            template_path.unlink()
            template_deleted = True
        
        # Delete from KMS
        kms_deleted = False
        try:
            kms = get_kms(passphrase.strip())
            kms.delete_user(user_id)
            kms_deleted = True
        except ValueError:
            pass  # User not in KMS
        
        if template_deleted or kms_deleted:
            details = f"""
### User Deleted 🗑️

**User ID:** `{user_id}`
**Template deleted:** {"Yes" if template_deleted else "No (not found)"}
**KMS entry deleted:** {"Yes" if kms_deleted else "No (not found)"}
"""
            return "✅ Deleted", details
        else:
            return "⚠️ Not Found", f"User `{user_id}` not found in templates or KMS"
        
    except Exception as e:
        return "❌ Error", f"Deletion failed: {str(e)}"


# ============================================================================
# Gradio Interface
# ============================================================================

def create_interface():
    """Create Gradio interface."""
    
    with gr.Blocks(
        title="BioProt - Biometric Template Protection"
    ) as app:
        
        gr.Markdown("""
        # 🔐 BioProt - Biometric Template Protection
        
        Secure face recognition with cancelable biometric templates.
        
        **Features:**
        - Two protection methods: **Ortho+Sign** (512-bit) and **Perm+LUT** (configurable)
        - Encrypted local key management
        - Key rotation for template revocation
        - Irreversible templates protect biometric privacy
        
        ---
        """)
        
        with gr.Tabs():
            # ==================== ENROLL TAB ====================
            with gr.TabItem("📝 Enroll"):
                gr.Markdown("### Enroll a New User\nUpload a face image to create a protected template.")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        enroll_image = gr.Image(
                            label="Face Image",
                            type="numpy",
                            sources=["upload", "webcam", "clipboard"]
                        )
                        enroll_user_id = gr.Textbox(
                            label="User ID",
                            placeholder="e.g., alice, bob, user001"
                        )
                        enroll_method = gr.Radio(
                            choices=["ortho", "permlut"],
                            value="ortho",
                            label="Protection Method",
                            info="Ortho: 512-bit binary template | PermLUT: Compact quantized template"
                        )
                        enroll_passphrase = gr.Textbox(
                            label="KMS Passphrase",
                            type="password",
                            placeholder="Enter passphrase for key storage",
                            value="demo_passphrase"
                        )
                        enroll_btn = gr.Button("🔐 Enroll User", variant="primary")
                    
                    with gr.Column(scale=1):
                        enroll_status = gr.Textbox(label="Status", interactive=False)
                        enroll_details = gr.Markdown(label="Details")
                        enroll_template = gr.Code(
                            label="Template Preview (JSON)",
                            language="json",
                            interactive=False
                        )
                
                enroll_btn.click(
                    fn=enroll_user,
                    inputs=[enroll_image, enroll_user_id, enroll_method, enroll_passphrase],
                    outputs=[enroll_status, enroll_details, enroll_template]
                )
            
            # ==================== VERIFY TAB ====================
            with gr.TabItem("✅ Verify"):
                gr.Markdown("### Verify User Identity\nUpload a probe image to verify against enrolled template.")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        verify_image = gr.Image(
                            label="Probe Image",
                            type="numpy",
                            sources=["upload", "webcam", "clipboard"]
                        )
                        verify_user_id = gr.Textbox(
                            label="User ID to Verify",
                            placeholder="Enter enrolled user ID"
                        )
                        verify_threshold = gr.Slider(
                            minimum=0.5,
                            maximum=0.95,
                            value=0.80,
                            step=0.01,
                            label="Threshold",
                            info="Higher = stricter matching"
                        )
                        verify_passphrase = gr.Textbox(
                            label="KMS Passphrase",
                            type="password",
                            placeholder="Same passphrase used during enrollment",
                            value="demo_passphrase"
                        )
                        verify_btn = gr.Button("🔍 Verify Identity", variant="primary")
                    
                    with gr.Column(scale=1):
                        verify_result = gr.Textbox(label="Result", interactive=False)
                        verify_score = gr.Textbox(label="Match Score", interactive=False)
                        verify_details = gr.Markdown(label="Details")
                
                verify_btn.click(
                    fn=verify_user,
                    inputs=[verify_image, verify_user_id, verify_threshold, verify_passphrase],
                    outputs=[verify_result, verify_details, verify_score]
                )
            
            # ==================== MANAGE TAB ====================
            with gr.TabItem("⚙️ Manage"):
                gr.Markdown("### Key Management\nRotate keys, list users, or delete enrollments.")
                
                with gr.Row():
                    with gr.Column():
                        manage_passphrase = gr.Textbox(
                            label="KMS Passphrase",
                            type="password",
                            value="demo_passphrase"
                        )
                        
                        gr.Markdown("#### 📋 List Enrolled Users")
                        list_btn = gr.Button("Refresh User List")
                        users_display = gr.Markdown()
                        
                        list_btn.click(
                            fn=list_users,
                            inputs=[manage_passphrase],
                            outputs=[users_display]
                        )
                    
                    with gr.Column():
                        gr.Markdown("#### 🔄 Rotate Key")
                        rotate_user_id = gr.Textbox(
                            label="User ID",
                            placeholder="User to rotate key for"
                        )
                        rotate_btn = gr.Button("Rotate Key", variant="secondary")
                        rotate_status = gr.Textbox(label="Status", interactive=False)
                        rotate_details = gr.Markdown()
                        
                        rotate_btn.click(
                            fn=rotate_key,
                            inputs=[rotate_user_id, manage_passphrase],
                            outputs=[rotate_status, rotate_details]
                        )
                        
                        gr.Markdown("---")
                        gr.Markdown("#### 🗑️ Delete User")
                        delete_user_id = gr.Textbox(
                            label="User ID to Delete",
                            placeholder="User to remove"
                        )
                        delete_btn = gr.Button("Delete User", variant="stop")
                        delete_status = gr.Textbox(label="Status", interactive=False)
                        delete_details = gr.Markdown()
                        
                        delete_btn.click(
                            fn=delete_user,
                            inputs=[delete_user_id, manage_passphrase],
                            outputs=[delete_status, delete_details]
                        )
            
            # ==================== ABOUT TAB ====================
            with gr.TabItem("ℹ️ About"):
                gr.Markdown("""
                ## About BioProt
                
                BioProt demonstrates **cancelable biometric template protection** for face recognition.
                
                ### Why Template Protection?
                
                Traditional face recognition stores raw embeddings. If compromised:
                - ❌ Biometric data is permanently exposed
                - ❌ Cannot change your face like a password
                - ❌ Same embedding can be used across databases
                
                With BioProt's protected templates:
                - ✅ Templates are **irreversible** - cannot reconstruct face
                - ✅ Templates are **cancelable** - rotate key to revoke
                - ✅ Templates are **unlinkable** - different keys = different templates
                
                ### Protection Methods
                
                | Method | Description | Template Size |
                |--------|-------------|---------------|
                | **Ortho+Sign** | Orthonormal projection + binarization | 512 bits (64 bytes) |
                | **Perm+LUT** | Permutation + lookup table quantization | 128-512 bits |
                
                ### Workflow
                
                ```
                Face Image → FaceNet Embedding → Protection Transform → Binary Template
                                                        ↑
                                                   User Seed (from KMS)
                ```
                
                ### Security Model
                
                - **KMS**: Encrypted local storage (PBKDF2 + Fernet)
                - **Templates**: Stored as JSON with base64-encoded bits
                - **Matching**: Hamming distance on binary templates
                
                ### Source Code
                
                - `protect.py` - Protection algorithms
                - `kms_sim.py` - Key management
                - `cli.py` - Command line interface
                - `app.py` - This Gradio interface
                
                ---
                
                ⚠️ **Note**: This is a research prototype. For production use, implement proper HSM integration.
                """)
        
        gr.Markdown("""
        ---
        *BioProt - Biometric Template Protection Prototype*
        """)
    
    return app


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    # Check dependencies
    if not FACENET_AVAILABLE:
        print("\n" + "="*60)
        print("WARNING: Face recognition models not available!")
        print("Install with: pip install facenet-pytorch torch torchvision pillow")
        print("="*60 + "\n")
    
    # Create and launch app
    app = create_interface()
    
    # Launch with settings suitable for Codespaces
    app.launch(
        server_name="0.0.0.0",  # Allow external connections
        server_port=7860,
        share=True,  # Create public URL (useful for Codespaces)
        show_error=True
    )
