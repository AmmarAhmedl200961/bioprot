# Contributing to BioProt

Thank you for your interest in contributing to BioProt! This document provides guidelines for contributing to the project.

## Code of Conduct

Please be respectful and constructive in all interactions. We welcome contributions from everyone.

## How to Contribute

### Reporting Bugs

1. Check existing issues to avoid duplicates
2. Create a new issue with:
   - Clear title describing the bug
   - Steps to reproduce
   - Expected vs actual behavior
   - Python version and OS
   - Relevant error messages

### Suggesting Features

1. Open an issue with the `enhancement` label
2. Describe the feature and its use case
3. Discuss implementation approach

### Submitting Code

1. **Fork** the repository
2. **Create a branch** for your feature:
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Make changes** following our coding standards
4. **Write tests** for new functionality
5. **Run tests** to ensure nothing breaks:
   ```bash
   pytest tests/ -v
   ```
6. **Commit** with clear messages:
   ```bash
   git commit -m "Add amazing feature for X"
   ```
7. **Push** to your fork:
   ```bash
   git push origin feature/amazing-feature
   ```
8. **Open a Pull Request**

## Development Setup

```bash
# Clone your fork
git clone https://github.com/AmmarAhmedl200961/bioprot.git
cd bioprot

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install dev dependencies
pip install pytest pytest-cov black flake8

# Run tests
pytest tests/ -v
```

## Coding Standards

### Python Style

- Follow PEP 8
- Use type hints where practical
- Maximum line length: 100 characters
- Use descriptive variable names

### Documentation

- Add docstrings to all public functions
- Follow Google-style docstrings:

```python
def protect_embedding(embedding: np.ndarray, seed: bytes) -> dict:
    """
    Protect a face embedding using cancelable transform.
    
    Args:
        embedding: 512-dimensional face embedding.
        seed: 32-byte random seed from KMS.
    
    Returns:
        dict: Protected template with 'method', 'bits', 'packed' keys.
    
    Raises:
        ValueError: If embedding dimension is not 512.
    
    Example:
        >>> template = protect_embedding(emb, seed)
    """
```

### Testing

- Write tests for all new functionality
- Aim for >90% code coverage
- Use descriptive test names:
  ```python
  def test_ortho_sign_produces_deterministic_output():
      ...
  ```

### Commit Messages

- Use present tense: "Add feature" not "Added feature"
- Be descriptive but concise
- Reference issues when applicable: "Fix #123"

## Project Structure

When adding new features, follow the existing structure:

```
bioprot/
├── protect.py      # Core algorithms → add new transforms here
├── kms_sim.py      # Key management → add new backends here
├── cli.py          # CLI commands → add new commands here
├── evaluate.py     # Metrics → add new evaluations here
└── tests/
    ├── test_protect.py    # Tests for protect.py
    ├── test_kms.py        # Tests for kms_sim.py
    └── ...
```

## Review Process

1. All PRs require at least one review
2. CI must pass (tests, linting)
3. Documentation must be updated if needed
4. Breaking changes require discussion

## Questions?

Open an issue with the `question` label or start a discussion.

Thank you for contributing! 🙏
