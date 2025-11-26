#!/bin/bash
# Example verification script
set -e
export BIOPROT_KMS_PASSPHRASE="demo_passphrase_12345"
echo "===== BioProt Verification Demo ====="
echo "Test 1: Genuine verification"
python cli.py verify --user user01 --embedding embeddings/user01_02.npy --threshold 0.80
echo "Test 2: Impostor verification"
python cli.py verify --user user01 --embedding embeddings/user02_01.npy --threshold 0.80 || echo "Correctly rejected"
echo "===== Verification Complete ====="
