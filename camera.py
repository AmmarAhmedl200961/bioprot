#!/usr/bin/env python3
"""
Live camera enrollment for BioProt.

Uses webcam to capture face images and extract embeddings.
Requires: opencv-python, facenet-pytorch, torch, torchvision
"""

import sys
from typing import Optional, Tuple

import numpy as np

try:
    import cv2
except ImportError:
    print("OpenCV not installed. Run: pip install opencv-python")
    sys.exit(1)

try:
    import torch
    from facenet_pytorch import MTCNN, InceptionResnetV1
    from PIL import Image
except ImportError:
    print("FaceNet not installed. Run: pip install facenet-pytorch torch torchvision pillow")
    sys.exit(1)


class FaceNetEmbedder:
    """Extract face embeddings using FaceNet (InceptionResnetV1)."""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Face detection
        self.mtcnn = MTCNN(
            image_size=160,
            margin=20,
            device=self.device,
            keep_all=False,
            post_process=True
        )
        
        # Face embedding
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
    
    def detect_face(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Detect face and return cropped face tensor and bounding box."""
        # Convert BGR to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)
        
        # Detect face
        face, prob = self.mtcnn(pil_image, return_prob=True)
        
        if face is None:
            return None, None
        
        # Get bounding box
        boxes, _ = self.mtcnn.detect(pil_image)
        box = boxes[0] if boxes is not None else None
        
        return face, box
    
    def get_embedding(self, face_tensor: torch.Tensor) -> np.ndarray:
        """Get embedding from cropped face tensor."""
        face_tensor = face_tensor.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            embedding = self.resnet(face_tensor)
        
        return embedding.cpu().numpy().flatten()
    
    def extract_from_frame(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Extract embedding from frame. Returns (embedding, bounding_box)."""
        face, box = self.detect_face(frame)
        
        if face is None:
            return None, None
        
        embedding = self.get_embedding(face)
        return embedding, box


class CameraEnroller:
    """Interactive camera enrollment."""
    
    def __init__(self, camera_id: int = 0):
        self.camera_id = camera_id
        self.embedder = FaceNetEmbedder()
    
    def capture_embedding(self, window_name: str = "BioProt Enrollment") -> Optional[np.ndarray]:
        """
        Open camera and capture embedding on space key press.
        
        Returns:
            Embedding array or None if cancelled
        """
        cap = cv2.VideoCapture(self.camera_id)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return None
        
        print("Press SPACE to capture, Q to quit")
        
        captured_embedding = None
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect face
            embedding, box = self.embedder.extract_from_frame(frame)
            
            # Draw bounding box if face detected
            display = frame.copy()
            if box is not None:
                x1, y1, x2, y2 = [int(b) for b in box]
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(display, "Face Detected", (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                cv2.putText(display, "No Face Detected", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.putText(display, "SPACE: Capture | Q: Quit", (10, display.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow(window_name, display)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):  # Space
                if embedding is not None:
                    captured_embedding = embedding
                    print("✓ Embedding captured!")
                    break
                else:
                    print("No face detected, try again")
            
            elif key == ord('q'):  # Q
                print("Cancelled")
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        return captured_embedding
    
    def enroll_user(self, user_id: str, kms_passphrase: str, 
                    method: str = "ortho") -> bool:
        """
        Complete enrollment workflow.
        
        Args:
            user_id: User identifier
            kms_passphrase: KMS encryption passphrase
            method: Protection method
            
        Returns:
            True if successful
        """
        from protect import protect_embedding, serialize_template
        from kms_sim import LocalKMS
        from pathlib import Path
        
        print(f"\n=== Enrolling user: {user_id} ===\n")
        
        # Capture embedding
        embedding = self.capture_embedding(f"Enroll: {user_id}")
        
        if embedding is None:
            return False
        
        # Initialize KMS
        kms = LocalKMS(kms_passphrase)
        
        # Delete if exists
        if user_id in kms.list_users():
            kms.delete_user(user_id)
        
        # Create seed
        seed = kms.create_user_key(user_id)
        
        # Protect
        template_data = protect_embedding(embedding, seed, method=method)
        
        # Save
        template_dir = Path("templates")
        template_dir.mkdir(exist_ok=True)
        
        template_json = serialize_template(template_data, user_id, seed_version=1)
        template_path = template_dir / f"{user_id}.json"
        template_path.write_text(template_json)
        
        print(f"\n✓ Enrolled '{user_id}' successfully!")
        print(f"  Template: {template_path}")
        print(f"  Method: {method}")
        print(f"  Bits: {template_data['params']['nbits']}")
        
        return True


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Camera enrollment for BioProt")
    parser.add_argument("--user", "-u", required=True, help="User ID")
    parser.add_argument("--method", "-m", default="ortho", choices=["ortho", "permlut"])
    parser.add_argument("--camera", "-c", type=int, default=0, help="Camera ID")
    parser.add_argument("--kms-passphrase", default=None, help="KMS passphrase")
    
    args = parser.parse_args()
    
    import os
    passphrase = args.kms_passphrase or os.environ.get("BIOPROT_KMS_PASSPHRASE", "demo")
    
    enroller = CameraEnroller(args.camera)
    success = enroller.enroll_user(args.user, passphrase, args.method)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
