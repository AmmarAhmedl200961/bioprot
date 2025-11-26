"""
Unit tests for protect.py module.
"""

import numpy as np
import pytest

from protect import (
    seed_to_rng, generate_seed, pack_bits, unpack_bits,
    bits_to_base64, base64_to_bits, generate_ortho_matrix,
    protect_ortho_sign, match_ortho_sign, generate_permutation,
    generate_lut, protect_perm_lut, match_perm_lut,
    protect_embedding, match_protected, serialize_template,
    deserialize_template
)


class TestRNGUtilities:
    """Test RNG and seed utilities."""
    
    def test_seed_to_rng_deterministic(self):
        """Test that same seed produces same RNG output."""
        seed = b"test_seed_12345678901234567890"
        
        rng1 = seed_to_rng(seed, "test")
        rng2 = seed_to_rng(seed, "test")
        
        # Generate some random numbers
        vals1 = rng1.normal(size=10)
        vals2 = rng2.normal(size=10)
        
        assert np.allclose(vals1, vals2)
    
    def test_seed_to_rng_different_labels(self):
        """Test that different labels produce different RNG outputs."""
        seed = b"test_seed_12345678901234567890"
        
        rng1 = seed_to_rng(seed, "label1")
        rng2 = seed_to_rng(seed, "label2")
        
        vals1 = rng1.normal(size=10)
        vals2 = rng2.normal(size=10)
        
        assert not np.allclose(vals1, vals2)
    
    def test_generate_seed(self):
        """Test seed generation."""
        seed1 = generate_seed(32)
        seed2 = generate_seed(32)
        
        assert len(seed1) == 32
        assert len(seed2) == 32
        assert seed1 != seed2  # Should be different


class TestBitPacking:
    """Test bit packing utilities."""
    
    def test_pack_unpack_bits(self):
        """Test packing and unpacking bits."""
        bits = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0], dtype=np.uint8)
        
        packed = pack_bits(bits)
        unpacked = unpack_bits(packed, len(bits))
        
        assert np.array_equal(bits, unpacked)
    
    def test_base64_roundtrip(self):
        """Test base64 encoding and decoding."""
        bits = np.array([1, 0, 1, 1, 0, 0, 1, 0] * 8, dtype=np.uint8)
        
        b64 = bits_to_base64(bits)
        decoded = base64_to_bits(b64, len(bits))
        
        assert np.array_equal(bits, decoded)


class TestOrthoSign:
    """Test Ortho+Sign protection method."""
    
    def test_generate_ortho_matrix(self):
        """Test orthonormal matrix generation."""
        seed = b"test_seed_12345678901234567890"
        dim = 128
        
        Q = generate_ortho_matrix(seed, dim)
        
        # Check shape
        assert Q.shape == (dim, dim)
        
        # Check orthonormality: Q @ Q.T should be identity
        I = Q @ Q.T
        assert np.allclose(I, np.eye(dim), atol=1e-10)
    
    def test_ortho_matrix_deterministic(self):
        """Test that same seed produces same matrix."""
        seed = b"test_seed_12345678901234567890"
        dim = 64
        
        Q1 = generate_ortho_matrix(seed, dim)
        Q2 = generate_ortho_matrix(seed, dim)
        
        assert np.allclose(Q1, Q2)
    
    def test_protect_ortho_sign_deterministic(self):
        """Test that same embedding and seed produce same template."""
        embedding = np.random.randn(128)
        seed = b"test_seed_12345678901234567890"
        
        template1 = protect_ortho_sign(embedding, seed)
        template2 = protect_ortho_sign(embedding, seed)
        
        assert np.array_equal(template1, template2)
    
    def test_protect_ortho_sign_output_binary(self):
        """Test that output is binary."""
        embedding = np.random.randn(128)
        seed = b"test_seed_12345678901234567890"
        
        template = protect_ortho_sign(embedding, seed)
        
        assert template.dtype == np.uint8
        assert np.all((template == 0) | (template == 1))
    
    def test_match_ortho_sign_identical(self):
        """Test matching identical templates."""
        embedding = np.random.randn(128)
        seed = b"test_seed_12345678901234567890"
        
        template = protect_ortho_sign(embedding, seed)
        score = match_ortho_sign(template, template)
        
        assert score == 1.0
    
    def test_match_ortho_sign_different_seeds(self):
        """Test matching templates from different seeds."""
        embedding = np.random.randn(128)
        seed1 = b"seed1_1234567890123456789012345"
        seed2 = b"seed2_1234567890123456789012345"
        
        template1 = protect_ortho_sign(embedding, seed1)
        template2 = protect_ortho_sign(embedding, seed2)
        
        score = match_ortho_sign(template1, template2)
        
        # Should be around 0.5 for random bits
        assert 0.3 < score < 0.7


class TestPermLUT:
    """Test Perm+LUT protection method."""
    
    def test_generate_permutation_deterministic(self):
        """Test that same seed produces same permutation."""
        seed = b"test_seed_12345678901234567890"
        size = 100
        
        perm1 = generate_permutation(seed, size)
        perm2 = generate_permutation(seed, size)
        
        assert np.array_equal(perm1, perm2)
    
    def test_generate_permutation_valid(self):
        """Test that permutation is valid."""
        seed = b"test_seed_12345678901234567890"
        size = 100
        
        perm = generate_permutation(seed, size)
        
        # Should contain all indices 0 to size-1
        assert set(perm) == set(range(size))
    
    def test_generate_lut_deterministic(self):
        """Test that same seed produces same LUT."""
        seed = b"test_seed_12345678901234567890"
        
        lut1 = generate_lut(seed, n_groups=10, n_bins=4, bits_per_bin=2)
        lut2 = generate_lut(seed, n_groups=10, n_bins=4, bits_per_bin=2)
        
        assert np.array_equal(lut1, lut2)
    
    def test_protect_perm_lut_deterministic(self):
        """Test that same embedding and seed produce same template."""
        embedding = np.random.randn(128)
        seed = b"test_seed_12345678901234567890"
        
        template1 = protect_perm_lut(embedding, seed, group_size=8, n_bins=4)
        template2 = protect_perm_lut(embedding, seed, group_size=8, n_bins=4)
        
        assert np.array_equal(template1, template2)
    
    def test_protect_perm_lut_output_binary(self):
        """Test that output is binary."""
        embedding = np.random.randn(128)
        seed = b"test_seed_12345678901234567890"
        
        template = protect_perm_lut(embedding, seed)
        
        assert template.dtype == np.uint8
        assert np.all((template == 0) | (template == 1))
    
    def test_match_perm_lut_identical(self):
        """Test matching identical templates."""
        embedding = np.random.randn(128)
        seed = b"test_seed_12345678901234567890"
        
        template = protect_perm_lut(embedding, seed)
        score = match_perm_lut(template, template)
        
        assert score == 1.0


class TestUnifiedAPI:
    """Test unified protection API."""
    
    def test_protect_embedding_ortho(self):
        """Test protect_embedding with ortho method."""
        embedding = np.random.randn(128)
        seed = b"test_seed_12345678901234567890"
        
        result = protect_embedding(embedding, seed, method="ortho")
        
        assert result["method"] == "ortho"
        assert "template" in result
        assert "template_b64" in result
        assert "params" in result
        assert result["params"]["nbits"] == 128
    
    def test_protect_embedding_permlut(self):
        """Test protect_embedding with permlut method."""
        embedding = np.random.randn(128)
        seed = b"test_seed_12345678901234567890"
        
        result = protect_embedding(embedding, seed, method="permlut",
                                  group_size=8, n_bins=4, bits_per_bin=1)
        
        assert result["method"] == "permlut"
        assert "template" in result
        assert "template_b64" in result
    
    def test_match_protected(self):
        """Test matching protected templates."""
        embedding = np.random.randn(128)
        seed = b"test_seed_12345678901234567890"
        
        template1 = protect_embedding(embedding, seed, method="ortho")
        template2 = protect_embedding(embedding, seed, method="ortho")
        
        score = match_protected(template1, template2)
        
        assert score == 1.0
    
    def test_serialize_deserialize_template(self):
        """Test template serialization and deserialization."""
        embedding = np.random.randn(128)
        seed = b"test_seed_12345678901234567890"
        
        template_data = protect_embedding(embedding, seed, method="ortho")
        
        # Serialize
        json_str = serialize_template(template_data, "alice", seed_version=1)
        
        # Deserialize
        restored = deserialize_template(json_str)
        
        assert restored["user_id"] == "alice"
        assert restored["method"] == "ortho"
        assert restored["params"]["seed_version"] == 1
        assert restored["template_b64"] == template_data["template_b64"]
    
    def test_match_after_deserialization(self):
        """Test matching templates after serialization."""
        embedding = np.random.randn(128)
        seed = b"test_seed_12345678901234567890"
        
        template1 = protect_embedding(embedding, seed, method="ortho")
        
        # Serialize and deserialize
        json_str = serialize_template(template1, "alice")
        template2 = deserialize_template(json_str)
        
        # Match
        score = match_protected(template1, template2)
        
        assert score == 1.0


class TestRevocation:
    """Test key rotation and revocation."""
    
    def test_different_seeds_different_templates(self):
        """Test that different seeds produce different templates."""
        embedding = np.random.randn(128)
        seed1 = b"seed1_1234567890123456789012345"
        seed2 = b"seed2_1234567890123456789012345"
        
        template1 = protect_embedding(embedding, seed1, method="ortho")
        template2 = protect_embedding(embedding, seed2, method="ortho")
        
        score = match_protected(template1, template2)
        
        # Should be around 0.5 for independent templates
        assert score < 0.7


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
