#!/bin/bash
# Example enrollment script
set -e
export BIOPROT_KMS_PASSPHRASE="demo_passphrase_12345"
echo "===== BioProt Enrollment Demo ====="
python cli.py enroll --user user01 --embedding embeddings/user01_01.npy --method ortho
python cli.py enroll --user user02 --embedding embeddings/user02_01.npy --method ortho
echo "===== Enrollment Complete ====="
