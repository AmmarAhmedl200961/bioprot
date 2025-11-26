"""
Key Management Simulator (KMS) - Secure Key Storage

Provides encrypted storage for user-specific seeds used in template protection.
Uses PBKDF2 for key derivation and Fernet (AES-128-CBC) for encryption.
"""

import os
import json
import base64
import hashlib
import secrets
from pathlib import Path
from typing import Optional, Dict, Any
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


class LocalKMS:
    """Local Key Management Simulator with encrypted storage."""
    
    ITERATIONS = 480_000  # OWASP 2023 recommendation for PBKDF2-SHA256
    
    def __init__(self, store_path: str = "kms_store.bin", 
                 passphrase: Optional[str] = None):
        """
        Initialize KMS with encrypted store.
        
        Args:
            store_path: Path to encrypted key store file
            passphrase: Master passphrase (or set BIOPROT_KMS_PASSPHRASE env var)
        """
        self.store_path = Path(store_path)
        self._passphrase = passphrase or os.environ.get("BIOPROT_KMS_PASSPHRASE")
        
        if not self._passphrase:
            raise ValueError(
                "KMS passphrase required. Set BIOPROT_KMS_PASSPHRASE environment "
                "variable or pass passphrase parameter."
            )
        
        self._salt = self._get_or_create_salt()
        self._fernet = self._derive_fernet()
        self._keys: Dict[str, Dict[str, Any]] = self._load_store()
    
    def _get_or_create_salt(self) -> bytes:
        """Get existing salt or create new one."""
        salt_path = self.store_path.with_suffix(".salt")
        if salt_path.exists():
            return salt_path.read_bytes()
        else:
            salt = secrets.token_bytes(16)
            salt_path.write_bytes(salt)
            return salt
    
    def _derive_fernet(self) -> Fernet:
        """Derive Fernet key from passphrase using PBKDF2."""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=self._salt,
            iterations=self.ITERATIONS,
        )
        key = base64.urlsafe_b64encode(kdf.derive(self._passphrase.encode()))
        return Fernet(key)
    
    def _load_store(self) -> Dict[str, Dict[str, Any]]:
        """Load and decrypt key store."""
        if not self.store_path.exists():
            return {}
        
        try:
            encrypted = self.store_path.read_bytes()
            decrypted = self._fernet.decrypt(encrypted)
            return json.loads(decrypted.decode())
        except Exception:
            # Invalid passphrase or corrupted store
            return {}
    
    def _save_store(self) -> None:
        """Encrypt and save key store."""
        data = json.dumps(self._keys).encode()
        encrypted = self._fernet.encrypt(data)
        self.store_path.write_bytes(encrypted)
    
    def create_user_key(self, user_id: str, method: str = "ortho") -> int:
        """Create new seed for user.
        
        Args:
            user_id: Unique user identifier
            method: Protection method ('ortho' or 'perm')
            
        Returns:
            Generated seed value
        """
        seed = secrets.randbelow(2**31)  # Positive 32-bit integer
        
        self._keys[user_id] = {
            "seed": seed,
            "method": method,
            "version": 1,
            "created_at": self._timestamp()
        }
        self._save_store()
        
        return seed
    
    def get_user_seed(self, user_id: str) -> Optional[int]:
        """Retrieve user's current seed."""
        if user_id not in self._keys:
            return None
        return self._keys[user_id]["seed"]
    
    def get_user_method(self, user_id: str) -> Optional[str]:
        """Retrieve user's protection method."""
        if user_id not in self._keys:
            return None
        return self._keys[user_id].get("method", "ortho")
    
    def rotate_user_key(self, user_id: str) -> Optional[int]:
        """Generate new seed for user (for cancelability).
        
        Args:
            user_id: User to rotate key for
            
        Returns:
            New seed value, or None if user doesn't exist
        """
        if user_id not in self._keys:
            return None
        
        old_version = self._keys[user_id].get("version", 1)
        new_seed = secrets.randbelow(2**31)
        
        self._keys[user_id]["seed"] = new_seed
        self._keys[user_id]["version"] = old_version + 1
        self._keys[user_id]["rotated_at"] = self._timestamp()
        self._save_store()
        
        return new_seed
    
    def delete_user(self, user_id: str) -> bool:
        """Delete user's key."""
        if user_id not in self._keys:
            return False
        
        del self._keys[user_id]
        self._save_store()
        return True
    
    def list_users(self) -> list:
        """List all user IDs."""
        return list(self._keys.keys())
    
    def get_user_info(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user key metadata (without exposing seed)."""
        if user_id not in self._keys:
            return None
        
        info = self._keys[user_id].copy()
        info.pop("seed", None)  # Don't expose seed
        return info
    
    def _timestamp(self) -> str:
        """Get current ISO timestamp."""
        from datetime import datetime
        return datetime.utcnow().isoformat() + "Z"
    
    def export_public_info(self) -> Dict[str, Any]:
        """Export non-sensitive KMS info."""
        return {
            "user_count": len(self._keys),
            "users": {
                uid: self.get_user_info(uid) 
                for uid in self._keys
            }
        }
