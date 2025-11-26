"""
Unit tests for evaluate.py module.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from evaluate import (
    compute_genuine_impostor_scores,
    compute_roc_metrics,
    test_ortho_naive_inversion,
    test_regressor_inversion,
    test_revocation
)
from protect import protect_embedding
from kms_sim import LocalKMS


class TestScoreComputation:
    """Test genuine/impostor score computation."""
    
    @pytest.fixture
    def temp_kms(self):
        """Create temporary KMS instance."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "test_kms.bin"
            kms = LocalKMS(store_path=store_path, passphrase="test_passphrase")
            yield kms
    
    @pytest.fixture
    def sample_embeddings_dir(self, temp_kms):
        """Create temporary directory with sample embeddings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            embeddings_dir = Path(tmpdir) / "embeddings"
            embeddings_dir.mkdir()
            
            # Create sample embeddings for 3 users, 2 samples each
            np.random.seed(42)
            users = ["user1", "user2", "user3"]
            
            for user_id in users:
                temp_kms.create_user_key(user_id)
                for sample_num in range(1, 3):
                    # Create random embedding
                    embedding = np.random.randn(128)
                    embedding = embedding / np.linalg.norm(embedding)
                    
                    # Save to file
                    filename = f"{user_id}_{sample_num:02d}.npy"
                    np.save(embeddings_dir / filename, embedding)
            
            templates_dir = Path(tmpdir) / "templates"
            templates_dir.mkdir()
            
            yield embeddings_dir, templates_dir, temp_kms
    
    def test_compute_genuine_impostor_scores(self, sample_embeddings_dir):
        """Test computing genuine and impostor scores."""
        embeddings_dir, templates_dir, kms = sample_embeddings_dir
        
        genuine_scores, impostor_scores = compute_genuine_impostor_scores(
            embeddings_dir=embeddings_dir,
            templates_dir=templates_dir,
            kms=kms,
            method="ortho"
        )
        
        # Should have genuine scores (pairs from same user)
        # 3 users, 2 samples each: 3 * C(2,1) = 3 genuine pairs
        assert len(genuine_scores) == 3
        
        # Should have impostor scores (pairs from different users)
        assert len(impostor_scores) > 0
        
        # Genuine scores should be relatively high (different samples from same user)
        # Not 1.0 because different samples are being compared
        mean_genuine = np.mean(genuine_scores)
        assert 0.3 < mean_genuine <= 1.0
        
        # Impostor scores should be around 0.5 (random)
        mean_impostor = np.mean(impostor_scores)
        assert 0.3 < mean_impostor < 0.7
    
    def test_compute_genuine_impostor_scores_permlut(self, sample_embeddings_dir):
        """Test computing scores with permlut method."""
        embeddings_dir, templates_dir, kms = sample_embeddings_dir
        
        genuine_scores, impostor_scores = compute_genuine_impostor_scores(
            embeddings_dir=embeddings_dir,
            templates_dir=templates_dir,
            kms=kms,
            method="permlut",
            group_size=8,
            n_bins=4,
            bits_per_bin=1
        )
        
        # Should have scores
        assert len(genuine_scores) > 0
        assert len(impostor_scores) > 0


class TestROCMetrics:
    """Test ROC metrics computation."""
    
    def test_compute_roc_metrics_creates_files(self):
        """Test that ROC metrics computation creates expected output files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            
            # Create sample scores
            genuine_scores = [0.95, 0.90, 0.92, 0.88, 0.93, 0.91]
            impostor_scores = [0.45, 0.50, 0.42, 0.55, 0.48, 0.51]
            
            compute_roc_metrics(genuine_scores, impostor_scores, output_dir)
            
            # Check that output files were created
            assert (output_dir / "roc_metrics.csv").exists()
            assert (output_dir / "roc_curve.csv").exists()
            assert (output_dir / "roc_curve.png").exists()
            assert (output_dir / "score_distributions.png").exists()
    
    def test_compute_roc_metrics_csv_content(self):
        """Test ROC metrics CSV file content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            
            # Create clearly separable scores for high AUC
            genuine_scores = [0.95, 0.90, 0.92, 0.88, 0.93, 0.91]
            impostor_scores = [0.15, 0.20, 0.12, 0.25, 0.18, 0.21]
            
            compute_roc_metrics(genuine_scores, impostor_scores, output_dir)
            
            # Read metrics file
            with open(output_dir / "roc_metrics.csv", 'r') as f:
                content = f.read()
            
            # Should contain AUC metric
            assert "AUC" in content
            
            # Parse AUC value
            for line in content.strip().split('\n'):
                if line.startswith('AUC'):
                    auc_value = float(line.split(',')[1])
                    # With clearly separable scores, AUC should be high
                    assert auc_value > 0.9
                    break
    
    def test_compute_roc_metrics_with_overlapping_scores(self):
        """Test ROC metrics with overlapping score distributions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            
            # Create overlapping scores (more realistic)
            np.random.seed(42)
            genuine_scores = list(np.random.normal(0.8, 0.1, 50))
            impostor_scores = list(np.random.normal(0.5, 0.1, 50))
            
            compute_roc_metrics(genuine_scores, impostor_scores, output_dir)
            
            # Should complete without error
            assert (output_dir / "roc_metrics.csv").exists()


class TestIrreversibilityTests:
    """Test irreversibility testing functions."""
    
    @pytest.fixture
    def sample_embeddings_and_seeds(self):
        """Create sample embeddings and seeds for testing."""
        np.random.seed(42)
        n_samples = 15
        
        embeddings = [np.random.randn(128) for _ in range(n_samples)]
        # Normalize embeddings
        embeddings = [e / np.linalg.norm(e) for e in embeddings]
        
        # Generate seeds with proper 32-byte length (standard for this project)
        # The seed format follows the pattern used in test_protect.py
        seed_length = 32
        seeds = [f"seed_{i:02d}".ljust(seed_length, '_').encode('utf-8') for i in range(n_samples)]
        
        return embeddings, seeds
    
    def test_ortho_naive_inversion(self, sample_embeddings_and_seeds):
        """Test naive inversion attack results."""
        embeddings, seeds = sample_embeddings_and_seeds
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            
            test_ortho_naive_inversion(embeddings, seeds, output_dir)
            
            # Check that results file was created
            results_file = output_dir / "naive_inversion_results.json"
            assert results_file.exists()
            
            # Check results content
            with open(results_file, 'r') as f:
                results = json.load(f)
            
            assert "method" in results
            assert results["method"] == "ortho_naive_inversion"
            assert "mean_cosine" in results
            assert "n_samples" in results
            assert results["n_samples"] == len(embeddings)
    
    def test_regressor_inversion(self, sample_embeddings_and_seeds):
        """Test regressor inversion attack results."""
        embeddings, seeds = sample_embeddings_and_seeds
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            
            test_regressor_inversion(embeddings, seeds, "ortho", output_dir)
            
            # Check that results file was created
            results_file = output_dir / "regressor_inversion_results.json"
            assert results_file.exists()
            
            # Check results content
            with open(results_file, 'r') as f:
                results = json.load(f)
            
            assert "method" in results
            assert "mse" in results
            assert "mean_cosine_similarity" in results
            assert "n_train" in results
            assert "n_test" in results
    
    def test_regressor_inversion_permlut(self, sample_embeddings_and_seeds):
        """Test regressor inversion with permlut method."""
        embeddings, seeds = sample_embeddings_and_seeds
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            
            test_regressor_inversion(
                embeddings, seeds, "permlut", output_dir,
                group_size=8, n_bins=4, bits_per_bin=1
            )
            
            # Check that results file was created
            results_file = output_dir / "regressor_inversion_results.json"
            assert results_file.exists()


class TestRevocationTest:
    """Test revocation testing function."""
    
    @pytest.fixture
    def temp_kms(self):
        """Create temporary KMS instance."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "test_kms.bin"
            kms = LocalKMS(store_path=store_path, passphrase="test_passphrase")
            yield kms
    
    def test_revocation_creates_results(self, temp_kms):
        """Test that revocation test creates results file."""
        np.random.seed(42)
        embedding = np.random.randn(128)
        user_id = "test_user"
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            
            test_revocation(user_id, embedding, temp_kms, "ortho", output_dir)
            
            # Check results file
            results_file = output_dir / "revocation_test_results.json"
            assert results_file.exists()
            
            with open(results_file, 'r') as f:
                results = json.load(f)
            
            assert results["user_id"] == user_id
            assert results["method"] == "ortho"
            assert "score_old_old" in results
            assert "score_new_new" in results
            assert "score_old_new" in results
            assert "revocation_successful" in results
    
    def test_revocation_scores_validity(self, temp_kms):
        """Test that revocation test produces valid scores."""
        np.random.seed(42)
        embedding = np.random.randn(128)
        user_id = "test_user"
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            
            test_revocation(user_id, embedding, temp_kms, "ortho", output_dir)
            
            with open(output_dir / "revocation_test_results.json", 'r') as f:
                results = json.load(f)
            
            # Self-matches should be 1.0
            assert results["score_old_old"] == 1.0
            assert results["score_new_new"] == 1.0
            
            # Cross-match (old vs new seed) should be low (around 0.5 for random)
            assert results["score_old_new"] < 0.7
    
    def test_revocation_permlut(self, temp_kms):
        """Test revocation with permlut method."""
        np.random.seed(42)
        embedding = np.random.randn(128)
        user_id = "test_user_permlut"
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            
            test_revocation(
                user_id, embedding, temp_kms, "permlut", output_dir,
                group_size=8, n_bins=4, bits_per_bin=1
            )
            
            results_file = output_dir / "revocation_test_results.json"
            assert results_file.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
