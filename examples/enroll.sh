#!/bin/bash
# Example enrollment script using Ortho+Sign method

set -e

# Set KMS passphrase (in production, use environment variable or secure vault)
export BIOPROT_KMS_PASSPHRASE="demo_passphrase_12345"

echo "===== Biometric Template Protection Demo: Enrollment ====="
echo ""

# Enroll users with Ortho+Sign method
echo "Enrolling users with Ortho+Sign method..."
echo ""

python cli.py enroll --user user01 --embedding embeddings/user01_01.npy --method ortho
python cli.py enroll --user user02 --embedding embeddings/user02_01.npy --method ortho
python cli.py enroll --user user03 --embedding embeddings/user03_01.npy --method ortho

echo ""
echo "===== Enrollment Complete ====="
echo ""
echo "Templates stored in: templates/"
echo ""
echo "Next steps:"
echo "  - Run ./verify.sh to test verification"
echo "  - Run 'python cli.py rotate --user user01' to rotate keys"
