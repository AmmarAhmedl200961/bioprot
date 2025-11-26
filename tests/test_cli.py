"""
Unit tests for cli.py module.
"""

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from cli import (
    load_embedding_from_file,
    get_embedding,
    get_template_path,
    save_template,
    load_template,
    cmd_enroll,
    cmd_verify,
    cmd_rotate,
    cmd_inspect,
    main
)
from protect import protect_embedding, serialize_template
from kms_sim import LocalKMS


class TestEmbeddingUtilities:
    """Test embedding loading utilities."""
    
    def test_load_embedding_from_file(self):
        """Test loading embedding from .npy file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a sample embedding file
            embedding = np.random.randn(128).astype(np.float32)
            filepath = Path(tmpdir) / "test_embedding.npy"
            np.save(filepath, embedding)
            
            # Load it back
            loaded = load_embedding_from_file(filepath)
            
            assert np.allclose(embedding, loaded)
    
    def test_get_embedding_from_npy(self):
        """Test get_embedding with .npy file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a sample embedding file
            embedding = np.random.randn(128).astype(np.float32)
            filepath = Path(tmpdir) / "test_embedding.npy"
            np.save(filepath, embedding)
            
            # Get embedding
            loaded = get_embedding(embedding_path=filepath)
            
            assert np.allclose(embedding, loaded)
    
    def test_get_embedding_requires_input(self):
        """Test that get_embedding raises error without inputs."""
        with pytest.raises(ValueError, match="Must provide either"):
            get_embedding()


class TestTemplateStorage:
    """Test template storage utilities."""
    
    def test_get_template_path(self):
        """Test template path generation."""
        templates_dir = Path("/some/path/templates")
        path = get_template_path("alice", templates_dir)
        
        assert path == Path("/some/path/templates/alice.json")
    
    def test_save_and_load_template(self):
        """Test saving and loading template."""
        with tempfile.TemporaryDirectory() as tmpdir:
            templates_dir = Path(tmpdir)
            
            # Create a sample template
            seed = b"test_seed_12345678901234567890"
            embedding = np.random.randn(128)
            template_data = protect_embedding(embedding, seed, method="ortho")
            template_json = serialize_template(template_data, "alice", seed_version=1)
            
            # Save template
            save_template(template_json, "alice", templates_dir)
            
            # Check file exists
            template_path = get_template_path("alice", templates_dir)
            assert template_path.exists()
            
            # Load template
            loaded = load_template("alice", templates_dir)
            
            assert loaded["user_id"] == "alice"
            assert loaded["method"] == "ortho"
            assert loaded["params"]["seed_version"] == 1
    
    def test_load_template_not_found(self):
        """Test loading non-existent template raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            templates_dir = Path(tmpdir)
            
            with pytest.raises(FileNotFoundError, match="No template found"):
                load_template("nonexistent", templates_dir)


class TestCmdEnroll:
    """Test enroll command."""
    
    @pytest.fixture
    def enroll_setup(self):
        """Setup for enroll tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Create KMS
            kms_path = tmpdir / "test_kms.bin"
            
            # Create sample embedding
            embedding = np.random.randn(128).astype(np.float32)
            embedding_path = tmpdir / "test_embedding.npy"
            np.save(embedding_path, embedding)
            
            templates_dir = tmpdir / "templates"
            
            yield {
                "tmpdir": tmpdir,
                "kms_path": str(kms_path),
                "embedding_path": str(embedding_path),
                "templates_dir": str(templates_dir)
            }
    
    def test_cmd_enroll_basic(self, enroll_setup):
        """Test basic enrollment."""
        # Create mock args
        args = MagicMock()
        args.user = "alice"
        args.image = None
        args.embedding = enroll_setup["embedding_path"]
        args.no_gpu = True
        args.method = "ortho"
        args.templates_dir = enroll_setup["templates_dir"]
        args.kms_path = enroll_setup["kms_path"]
        args.kms_passphrase = "test_pass"
        args.output = None
        args.group_size = 8
        args.n_bins = 4
        args.bits_per_bin = 1
        
        result = cmd_enroll(args)
        
        assert result == 0
        
        # Check template was created
        template_path = Path(enroll_setup["templates_dir"]) / "alice.json"
        assert template_path.exists()
    
    def test_cmd_enroll_with_output_file(self, enroll_setup):
        """Test enrollment with specific output file."""
        output_file = Path(enroll_setup["tmpdir"]) / "custom_output.json"
        
        args = MagicMock()
        args.user = "bob"
        args.image = None
        args.embedding = enroll_setup["embedding_path"]
        args.no_gpu = True
        args.method = "ortho"
        args.templates_dir = enroll_setup["templates_dir"]
        args.kms_path = enroll_setup["kms_path"]
        args.kms_passphrase = "test_pass"
        args.output = str(output_file)
        args.group_size = 8
        args.n_bins = 4
        args.bits_per_bin = 1
        
        result = cmd_enroll(args)
        
        assert result == 0
        assert output_file.exists()
    
    def test_cmd_enroll_permlut(self, enroll_setup):
        """Test enrollment with permlut method."""
        args = MagicMock()
        args.user = "charlie"
        args.image = None
        args.embedding = enroll_setup["embedding_path"]
        args.no_gpu = True
        args.method = "permlut"
        args.templates_dir = enroll_setup["templates_dir"]
        args.kms_path = enroll_setup["kms_path"]
        args.kms_passphrase = "test_pass"
        args.output = None
        args.group_size = 8
        args.n_bins = 4
        args.bits_per_bin = 1
        
        result = cmd_enroll(args)
        
        assert result == 0
        
        # Verify template uses permlut
        template = load_template("charlie", Path(enroll_setup["templates_dir"]))
        assert template["method"] == "permlut"


class TestCmdVerify:
    """Test verify command."""
    
    @pytest.fixture
    def verify_setup(self):
        """Setup for verify tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Create KMS and enroll a user
            kms_path = tmpdir / "test_kms.bin"
            kms = LocalKMS(store_path=kms_path, passphrase="test_pass")
            seed = kms.create_user_key("alice")
            
            # Create sample embedding
            embedding = np.random.randn(128).astype(np.float32)
            embedding = embedding / np.linalg.norm(embedding)
            embedding_path = tmpdir / "test_embedding.npy"
            np.save(embedding_path, embedding)
            
            # Create template
            templates_dir = tmpdir / "templates"
            templates_dir.mkdir()
            template_data = protect_embedding(embedding, seed, method="ortho")
            template_json = serialize_template(template_data, "alice", seed_version=1)
            save_template(template_json, "alice", templates_dir)
            
            yield {
                "tmpdir": tmpdir,
                "kms_path": str(kms_path),
                "embedding_path": str(embedding_path),
                "templates_dir": str(templates_dir),
                "embedding": embedding
            }
    
    def test_cmd_verify_genuine(self, verify_setup):
        """Test verification with same embedding (genuine)."""
        args = MagicMock()
        args.user = "alice"
        args.image = None
        args.embedding = verify_setup["embedding_path"]
        args.no_gpu = True
        args.templates_dir = verify_setup["templates_dir"]
        args.kms_path = verify_setup["kms_path"]
        args.kms_passphrase = "test_pass"
        args.threshold = 0.80
        
        result = cmd_verify(args)
        
        # Should match (return 0)
        assert result == 0
    
    def test_cmd_verify_impostor(self, verify_setup):
        """Test verification with different embedding (impostor)."""
        # Create a different embedding
        impostor_embedding = np.random.randn(128).astype(np.float32)
        impostor_path = Path(verify_setup["tmpdir"]) / "impostor.npy"
        np.save(impostor_path, impostor_embedding)
        
        args = MagicMock()
        args.user = "alice"
        args.image = None
        args.embedding = str(impostor_path)
        args.no_gpu = True
        args.templates_dir = verify_setup["templates_dir"]
        args.kms_path = verify_setup["kms_path"]
        args.kms_passphrase = "test_pass"
        args.threshold = 0.80
        
        result = cmd_verify(args)
        
        # Should not match (return 1) with high probability
        # Note: with random embeddings, score should be ~0.5
        assert result == 1
    
    def test_cmd_verify_user_not_found(self, verify_setup):
        """Test verification with non-existent user."""
        args = MagicMock()
        args.user = "nonexistent"
        args.image = None
        args.embedding = verify_setup["embedding_path"]
        args.no_gpu = True
        args.templates_dir = verify_setup["templates_dir"]
        args.kms_path = verify_setup["kms_path"]
        args.kms_passphrase = "test_pass"
        args.threshold = 0.80
        
        result = cmd_verify(args)
        
        # Should return error
        assert result == 1


class TestCmdRotate:
    """Test rotate command."""
    
    @pytest.fixture
    def rotate_setup(self):
        """Setup for rotate tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Create KMS and enroll a user
            kms_path = tmpdir / "test_kms.bin"
            kms = LocalKMS(store_path=kms_path, passphrase="test_pass")
            kms.create_user_key("alice")
            
            yield {
                "tmpdir": tmpdir,
                "kms_path": str(kms_path)
            }
    
    def test_cmd_rotate_success(self, rotate_setup):
        """Test successful key rotation."""
        args = MagicMock()
        args.user = "alice"
        args.kms_path = rotate_setup["kms_path"]
        args.kms_passphrase = "test_pass"
        
        result = cmd_rotate(args)
        
        assert result == 0
        
        # Verify version was incremented
        kms = LocalKMS(store_path=rotate_setup["kms_path"], passphrase="test_pass")
        assert kms.get_user_version("alice") == 2
    
    def test_cmd_rotate_nonexistent_user(self, rotate_setup):
        """Test rotation for non-existent user."""
        args = MagicMock()
        args.user = "nonexistent"
        args.kms_path = rotate_setup["kms_path"]
        args.kms_passphrase = "test_pass"
        
        result = cmd_rotate(args)
        
        assert result == 1


class TestCmdInspect:
    """Test inspect command."""
    
    @pytest.fixture
    def inspect_setup(self):
        """Setup for inspect tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Create a sample template
            seed = b"test_seed_12345678901234567890"
            embedding = np.random.randn(128)
            template_data = protect_embedding(embedding, seed, method="ortho")
            template_json = serialize_template(template_data, "alice", seed_version=1)
            
            templates_dir = tmpdir / "templates"
            templates_dir.mkdir()
            save_template(template_json, "alice", templates_dir)
            
            # Also save to a specific file
            template_file = tmpdir / "custom_template.json"
            with open(template_file, 'w') as f:
                f.write(template_json)
            
            yield {
                "tmpdir": tmpdir,
                "templates_dir": str(templates_dir),
                "template_file": str(template_file)
            }
    
    def test_cmd_inspect_by_user(self, inspect_setup):
        """Test inspecting template by user ID."""
        args = MagicMock()
        args.user = "alice"
        args.template = None
        args.templates_dir = inspect_setup["templates_dir"]
        
        result = cmd_inspect(args)
        
        assert result == 0
    
    def test_cmd_inspect_by_file(self, inspect_setup):
        """Test inspecting template by file path."""
        args = MagicMock()
        args.user = None
        args.template = inspect_setup["template_file"]
        args.templates_dir = inspect_setup["templates_dir"]
        
        result = cmd_inspect(args)
        
        assert result == 0
    
    def test_cmd_inspect_user_not_found(self, inspect_setup):
        """Test inspecting non-existent user."""
        args = MagicMock()
        args.user = "nonexistent"
        args.template = None
        args.templates_dir = inspect_setup["templates_dir"]
        
        result = cmd_inspect(args)
        
        assert result == 1


class TestMainCLI:
    """Test main CLI entry point."""
    
    def test_main_no_command(self):
        """Test main with no command shows help."""
        with patch('sys.argv', ['cli.py']):
            result = main()
            assert result == 1
    
    def test_main_unknown_command(self):
        """Test main with unknown command raises SystemExit."""
        with patch('sys.argv', ['cli.py', 'unknown']):
            # argparse exits with status 2 for invalid choices
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 2
    
    def test_main_enroll_command(self):
        """Test main with enroll command."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Create sample embedding
            embedding = np.random.randn(128).astype(np.float32)
            embedding_path = tmpdir / "test_embedding.npy"
            np.save(embedding_path, embedding)
            
            kms_path = tmpdir / "test_kms.bin"
            templates_dir = tmpdir / "templates"
            
            argv = [
                'cli.py', 'enroll',
                '--user', 'testuser',
                '--embedding', str(embedding_path),
                '--kms-path', str(kms_path),
                '--kms-passphrase', 'test_pass',
                '--templates-dir', str(templates_dir),
                '--method', 'ortho'
            ]
            
            with patch('sys.argv', argv):
                result = main()
                assert result == 0
    
    def test_main_inspect_command(self):
        """Test main with inspect command."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Create a sample template file
            seed = b"test_seed_12345678901234567890"
            embedding = np.random.randn(128)
            template_data = protect_embedding(embedding, seed, method="ortho")
            template_json = serialize_template(template_data, "alice", seed_version=1)
            
            template_file = tmpdir / "template.json"
            with open(template_file, 'w') as f:
                f.write(template_json)
            
            argv = [
                'cli.py', 'inspect',
                '--template', str(template_file)
            ]
            
            with patch('sys.argv', argv):
                result = main()
                assert result == 0


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_enroll_invalid_embedding_path(self):
        """Test enrollment with invalid embedding path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            args = MagicMock()
            args.user = "alice"
            args.image = None
            args.embedding = "/nonexistent/path.npy"
            args.no_gpu = True
            args.method = "ortho"
            args.templates_dir = tmpdir
            args.kms_path = str(Path(tmpdir) / "kms.bin")
            args.kms_passphrase = "test_pass"
            args.output = None
            args.group_size = 8
            args.n_bins = 4
            args.bits_per_bin = 1
            
            result = cmd_enroll(args)
            
            # Should return error
            assert result == 1
    
    def test_verify_template_not_found(self):
        """Test verification when template doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Create KMS with user
            kms_path = tmpdir / "test_kms.bin"
            kms = LocalKMS(store_path=kms_path, passphrase="test_pass")
            kms.create_user_key("alice")
            
            # Create embedding but no template
            embedding = np.random.randn(128).astype(np.float32)
            embedding_path = tmpdir / "test_embedding.npy"
            np.save(embedding_path, embedding)
            
            templates_dir = tmpdir / "templates"
            templates_dir.mkdir()
            
            args = MagicMock()
            args.user = "alice"
            args.image = None
            args.embedding = str(embedding_path)
            args.no_gpu = True
            args.templates_dir = str(templates_dir)
            args.kms_path = str(kms_path)
            args.kms_passphrase = "test_pass"
            args.threshold = 0.80
            
            result = cmd_verify(args)
            
            # Should return error
            assert result == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
