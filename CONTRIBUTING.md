# Contributing to RLFusion Orchestrator

Thank you for your interest in contributing. This document provides guidelines for contributing to RLFusion Orchestrator.

## Reporting Bugs

1. **Search existing issues** first to avoid duplicates.
2. Open a new issue with:
   - A clear, descriptive title.
   - Steps to reproduce the problem.
   - Expected vs. actual behavior.
   - Your environment: OS, Python version, GPU model (if applicable), Ollama version.
3. Include relevant logs from `training/logs/` or terminal output.

## Security Vulnerabilities

**Do not open a public issue for security vulnerabilities.** See [SECURITY.md](SECURITY.md) for responsible disclosure instructions.

## Development Setup

Development setup differs slightly from end-user setup:

```bash
# Clone and enter the repo
git clone https://github.com/moonrunnerkc/rlfusion-orchestrator.git
cd rlfusion-orchestrator

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate

# Install backend dependencies
pip install -r backend/requirements.txt

# Install dev/test dependencies
pip install pytest black flake8 isort

# Copy environment config
cp .env.example .env

# Initialize the database
./scripts/init_db.sh

# Install frontend dependencies
cd frontend && npm install && cd ..
```

### Running the Backend

```bash
uvicorn backend.main:app --port 8000 --reload
```

### Running the Frontend

```bash
cd frontend
npm run dev
```

### Running Tests

```bash
# All tests (CPU-only, no GPU required)
pytest tests/ -v --tb=short -m "not gpu"

# GPU-specific tests (requires CUDA)
pytest tests/ -v --tb=short -m "gpu"
```

## Coding Standards

- **Python formatter:** [Black](https://github.com/psf/black) with default settings (line length 88).
- **Import sorting:** [isort](https://pycqa.github.io/isort/) with Black-compatible profile.
- **Linting:** [flake8](https://flake8.pycqa.org/). Ignore E501 (line length is handled by Black).
- **Type hints:** Encouraged for all new functions and public APIs.
- **Docstrings:** Required for all public functions and classes. Use Google-style docstrings.

Before submitting, run:

```bash
black backend/ tests/
isort backend/ tests/ --profile black
flake8 backend/ tests/ --ignore=E501,W503
```

## Pull Request Process

1. Fork the repository and create a feature branch from `main`.
2. Make your changes in focused, atomic commits.
3. Ensure all tests pass and the linter is clean.
4. Update documentation if your change affects the public API or user-facing behavior.
5. Open a PR with:
   - A clear description of what the change does and why.
   - References to any related issues.
6. A maintainer will review your PR. Expect feedback. This is collaborative, not adversarial.

## Commit Messages

Use clear, imperative-mood commit messages:

```
Add CSWR stability threshold configuration
Fix OOD detection false positives on short queries
Remove unused CAG cache invalidation logic
```

## Project Structure

```
backend/
  main.py          # FastAPI entry point
  config.py        # Configuration loader
  config.yaml      # Default configuration
  core/            # Core retrieval, fusion, critique logic
  rl/              # Reinforcement learning training scripts
frontend/
  src/             # React UI components
models/            # Trained RL policy artifacts
scripts/           # Utility scripts (DB init, compatibility fixes)
tests/             # Test suites (API, GPU, load testing)
```

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
