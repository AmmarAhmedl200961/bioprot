"""
Unit tests for kms_sim.py module.
"""

import tempfile
from pathlib import Path

import pytest

from kms_sim import LocalKMS, get_or_create_seed


class TestLocalKMS:
    """Test LocalKMS functionality."""
    
    @pytest.fixture
    def temp_kms(self):
        """Create temporary KMS instance."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "test_kms.bin"
            kms = LocalKMS(store_path=store_path, passphrase="test_passphrase")
            yield kms
    
    def test_create_user_key(self, temp_kms):
        """Test creating a new user key."""
        seed = temp_kms.create_user_key("alice")
        
        assert seed is not None
        assert len(seed) == 32
        assert isinstance(seed, bytes)
    
    def test_get_user_seed(self, temp_kms):
        """Test retrieving user seed."""
        original_seed = temp_kms.create_user_key("alice")
        retrieved_seed = temp_kms.get_user_seed("alice")
        
        assert original_seed == retrieved_seed
    
    def test_get_nonexistent_user(self, temp_kms):
        """Test retrieving seed for non-existent user."""
        seed = temp_kms.get_user_seed("nonexistent")
        
        assert seed is None
    
    def test_create_duplicate_user(self, temp_kms):
        """Test that creating duplicate user raises error."""
        temp_kms.create_user_key("alice")
        
        with pytest.raises(ValueError, match="already exists"):
            temp_kms.create_user_key("alice")
    
    def test_rotate_user_key(self, temp_kms):
        """Test rotating user key."""
        old_seed = temp_kms.create_user_key("alice")
        new_seed = temp_kms.rotate_user_key("alice")
        
        assert new_seed != old_seed
        assert len(new_seed) == 32
        
        # Verify new seed is now stored
        current_seed = temp_kms.get_user_seed("alice")
        assert current_seed == new_seed
    
    def test_rotate_nonexistent_user(self, temp_kms):
        """Test rotating key for non-existent user raises error."""
        with pytest.raises(ValueError, match="does not exist"):
            temp_kms.rotate_user_key("nonexistent")
    
    def test_get_user_version(self, temp_kms):
        """Test getting user seed version."""
        temp_kms.create_user_key("alice")
        
        version = temp_kms.get_user_version("alice")
        assert version == 1
        
        # After rotation
        temp_kms.rotate_user_key("alice")
        version = temp_kms.get_user_version("alice")
        assert version == 2
    
    def test_store_and_get_metadata(self, temp_kms):
        """Test storing and retrieving user metadata."""
        temp_kms.create_user_key("alice")
        
        metadata = {"department": "engineering", "enrolled": "2025-11-26"}
        temp_kms.store_user_metadata("alice", metadata)
        
        retrieved = temp_kms.get_user_metadata("alice")
        assert retrieved == metadata
    
    def test_metadata_nonexistent_user(self, temp_kms):
        """Test metadata operations on non-existent user."""
        # Get metadata
        metadata = temp_kms.get_user_metadata("nonexistent")
        assert metadata is None
        
        # Store metadata
        with pytest.raises(ValueError, match="does not exist"):
            temp_kms.store_user_metadata("nonexistent", {})
    
    def test_delete_user(self, temp_kms):
        """Test deleting a user."""
        temp_kms.create_user_key("alice")
        
        temp_kms.delete_user("alice")
        
        seed = temp_kms.get_user_seed("alice")
        assert seed is None
    
    def test_delete_nonexistent_user(self, temp_kms):
        """Test deleting non-existent user raises error."""
        with pytest.raises(ValueError, match="does not exist"):
            temp_kms.delete_user("nonexistent")
    
    def test_list_users(self, temp_kms):
        """Test listing all users."""
        temp_kms.create_user_key("alice")
        temp_kms.create_user_key("bob")
        temp_kms.create_user_key("charlie")
        
        users = temp_kms.list_users()
        
        assert set(users) == {"alice", "bob", "charlie"}
    
    def test_persistence(self):
        """Test that KMS state persists across instances."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "test_kms.bin"
            
            # Create KMS and add user
            kms1 = LocalKMS(store_path=store_path, passphrase="test_pass")
            seed1 = kms1.create_user_key("alice")
            
            # Create new KMS instance with same store
            kms2 = LocalKMS(store_path=store_path, passphrase="test_pass")
            seed2 = kms2.get_user_seed("alice")
            
            assert seed1 == seed2
    
    def test_wrong_passphrase(self):
        """Test that wrong passphrase fails to decrypt."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "test_kms.bin"
            
            # Create KMS with one passphrase
            kms1 = LocalKMS(store_path=store_path, passphrase="correct_pass")
            kms1.create_user_key("alice")
            
            # Try to open with wrong passphrase
            with pytest.raises(ValueError, match="wrong passphrase"):
                kms2 = LocalKMS(store_path=store_path, passphrase="wrong_pass")


class TestHelpers:
    """Test helper functions."""
    
    @pytest.fixture
    def temp_kms(self):
        """Create temporary KMS instance."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "test_kms.bin"
            kms = LocalKMS(store_path=store_path, passphrase="test_passphrase")
            yield kms
    
    def test_get_or_create_seed_new_user(self, temp_kms):
        """Test get_or_create_seed for new user."""
        seed = get_or_create_seed(temp_kms, "alice")
        
        assert seed is not None
        assert len(seed) == 32
        
        # Verify it was stored
        retrieved = temp_kms.get_user_seed("alice")
        assert retrieved == seed
    
    def test_get_or_create_seed_existing_user(self, temp_kms):
        """Test get_or_create_seed for existing user."""
        original_seed = temp_kms.create_user_key("alice")
        
        seed = get_or_create_seed(temp_kms, "alice")
        
        assert seed == original_seed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
