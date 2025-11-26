"""
Local Key Management System (KMS) simulator.

Provides file-backed, encrypted storage for user-specific seeds.
Uses PBKDF2 + Fernet for encryption.

WARNING: This is a prototype KMS for demonstration purposes only.
Production systems should use Hardware Security Modules (HSM) or
cloud-based KMS solutions.
"""

import base64
import hashlib
import json
import os
import secrets
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.fernet import Fernet


class LocalKMS:
    """
    Local file-backed Key Management System.
    
    Stores user seeds in an encrypted JSON file using passphrase-derived key.
    """
    
    DEFAULT_STORE_PATH = Path.home() / ".local" / "share" / "bioprot" / "kms_store.bin"
    
    def __init__(self, store_path: Optional[Path] = None, passphrase: Optional[str] = None):
        """
        Initialize LocalKMS.
        
        Args:
            store_path: Path to encrypted key store file (default: ~/.local/share/bioprot/kms_store.bin)
            passphrase: Passphrase for encryption (if None, will prompt or use env var)
        """
        self.store_path = Path(store_path) if store_path else self.DEFAULT_STORE_PATH
        
        # Get passphrase
        if passphrase is None:
            passphrase = os.environ.get("BIOPROT_KMS_PASSPHRASE")
            if passphrase is None:
                import getpass
                passphrase = getpass.getpass("Enter KMS passphrase: ")
        
        self.passphrase = passphrase
        self._cipher = self._derive_cipher(passphrase)
        
        # Create store directory if needed
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load or initialize store
        self._store: Dict[str, Dict[str, Any]] = self._load_store()
    
    def _derive_cipher(self, passphrase: str) -> Fernet:
        """
        Derive Fernet cipher from passphrase using PBKDF2.
        
        Args:
            passphrase: User passphrase
            
        Returns:
            Fernet cipher instance
        """
        # Use a fixed salt for the KMS instance (stored with the file)
        # In a real system, salt should be stored separately
        salt = b"bioprot_kms_salt_v1"  # Fixed for prototype
        
        # Derive 32-byte key using PBKDF2
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=480000,  # OWASP recommended (2023)
        )
        key = kdf.derive(passphrase.encode('utf-8'))
        
        # Fernet requires base64-encoded key
        key_b64 = base64.urlsafe_b64encode(key)
        return Fernet(key_b64)
    
    def _load_store(self) -> Dict[str, Dict[str, Any]]:
        """
        Load encrypted store from disk.
        
        Returns:
            Decrypted store dictionary
        """
        if not self.store_path.exists():
            return {}
        
        try:
            with open(self.store_path, 'rb') as f:
                encrypted_data = f.read()
            
            decrypted_data = self._cipher.decrypt(encrypted_data)
            store = json.loads(decrypted_data.decode('utf-8'))
            return store
        except Exception as e:
            raise ValueError(f"Failed to load KMS store (wrong passphrase?): {e}")
    
    def _save_store(self):
        """Save encrypted store to disk."""
        # Serialize store to JSON
        json_data = json.dumps(self._store, indent=2)
        
        # Encrypt
        encrypted_data = self._cipher.encrypt(json_data.encode('utf-8'))
        
        # Write atomically (write to temp, then rename)
        temp_path = self.store_path.with_suffix('.tmp')
        with open(temp_path, 'wb') as f:
            f.write(encrypted_data)
        
        temp_path.replace(self.store_path)
    
    def create_user_key(self, user_id: str) -> bytes:
        """
        Create a new seed for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            Generated seed bytes
            
        Raises:
            ValueError: If user already exists
        """
        if user_id in self._store:
            raise ValueError(f"User '{user_id}' already exists. Use rotate_user_key to change.")
        
        # Generate cryptographically secure random seed
        seed = secrets.token_bytes(32)
        
        # Store
        self._store[user_id] = {
            "seed_hex": seed.hex(),
            "created_at": datetime.utcnow().isoformat() + "Z",
            "version": 1,
            "metadata": {}
        }
        
        self._save_store()
        return seed
    
    def get_user_seed(self, user_id: str) -> Optional[bytes]:
        """
        Retrieve seed for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            Seed bytes, or None if user doesn't exist
        """
        if user_id not in self._store:
            return None
        
        seed_hex = self._store[user_id]["seed_hex"]
        return bytes.fromhex(seed_hex)
    
    def rotate_user_key(self, user_id: str) -> bytes:
        """
        Rotate (regenerate) seed for a user.
        
        Old templates will become invalid.
        
        Args:
            user_id: User identifier
            
        Returns:
            New seed bytes
            
        Raises:
            ValueError: If user doesn't exist
        """
        if user_id not in self._store:
            raise ValueError(f"User '{user_id}' does not exist. Use create_user_key first.")
        
        # Generate new seed
        new_seed = secrets.token_bytes(32)
        
        # Update store (increment version)
        old_version = self._store[user_id].get("version", 1)
        self._store[user_id]["seed_hex"] = new_seed.hex()
        self._store[user_id]["version"] = old_version + 1
        self._store[user_id]["rotated_at"] = datetime.utcnow().isoformat() + "Z"
        
        self._save_store()
        return new_seed
    
    def store_user_metadata(self, user_id: str, metadata: Dict[str, Any]):
        """
        Store arbitrary metadata for a user.
        
        Args:
            user_id: User identifier
            metadata: Metadata dictionary
            
        Raises:
            ValueError: If user doesn't exist
        """
        if user_id not in self._store:
            raise ValueError(f"User '{user_id}' does not exist.")
        
        self._store[user_id]["metadata"] = metadata
        self._save_store()
    
    def get_user_metadata(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve metadata for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            Metadata dictionary, or None if user doesn't exist
        """
        if user_id not in self._store:
            return None
        
        return self._store[user_id].get("metadata", {})
    
    def get_user_version(self, user_id: str) -> Optional[int]:
        """
        Get the current seed version for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            Seed version number, or None if user doesn't exist
        """
        if user_id not in self._store:
            return None
        
        return self._store[user_id].get("version", 1)
    
    def delete_user(self, user_id: str):
        """
        Delete a user and their seed.
        
        Args:
            user_id: User identifier
            
        Raises:
            ValueError: If user doesn't exist
        """
        if user_id not in self._store:
            raise ValueError(f"User '{user_id}' does not exist.")
        
        del self._store[user_id]
        self._save_store()
    
    def list_users(self) -> list:
        """
        List all user IDs in the store.
        
        Returns:
            List of user IDs
        """
        return list(self._store.keys())
    
    def export_store(self, output_path: Path, include_seeds: bool = False):
        """
        Export store metadata to JSON file (optionally without seeds).
        
        Args:
            output_path: Path to write JSON file
            include_seeds: If True, include seed_hex in export (SENSITIVE!)
        """
        export_data = {}
        for user_id, data in self._store.items():
            user_export = {
                "created_at": data.get("created_at"),
                "version": data.get("version"),
                "metadata": data.get("metadata", {})
            }
            if include_seeds:
                user_export["seed_hex"] = data["seed_hex"]
            
            export_data[user_id] = user_export
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)


def get_or_create_seed(kms: LocalKMS, user_id: str) -> bytes:
    """
    Helper function to get existing seed or create new one.
    
    Args:
        kms: LocalKMS instance
        user_id: User identifier
        
    Returns:
        User seed bytes
    """
    seed = kms.get_user_seed(user_id)
    if seed is None:
        seed = kms.create_user_key(user_id)
    return seed
