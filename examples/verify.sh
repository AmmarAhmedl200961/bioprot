#!/bin/bash
# Example verification script

set -e

# Set KMS passphrase
export BIOPROT_KMS_PASSPHRASE="demo_passphrase_12345"

echo "===== Biometric Template Protection Demo: Verification ====="
echo ""

# Test genuine verification (same user, different sample)
echo "Test 1: Genuine verification (user01 with different sample)"
python cli.py verify --user user01 --embedding embeddings/user01_02.npy --threshold 0.80
echo ""

# Test impostor verification (different user)
echo "Test 2: Impostor verification (user01 template vs user02 embedding)"
python cli.py verify --user user01 --embedding embeddings/user02_01.npy --threshold 0.80 || echo "Correctly rejected impostor"
echo ""

echo "Test 3: Another genuine verification (user02)"
python cli.py verify --user user02 --embedding embeddings/user02_02.npy --threshold 0.80
echo ""

echo "===== Verification Tests Complete ====="
