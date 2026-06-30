# Contributing to 3D Tongue Tip Tracking

Thank you for your interest in contributing! This is a research project for medical applications, so correctness and reproducibility are top priorities.

## Getting Started

```bash
git clone https://github.com/dronefreak/3D-tongue-tip-tracking.git
cd 3D-tongue-tip-tracking
pip install -r requirements.txt -r requirements-dev.txt
pre-commit install        # set up pre-commit hooks
pytest                    # verify everything passes
```

## How to Contribute

### Reporting bugs
Open a [GitHub Issue](https://github.com/dronefreak/3D-tongue-tip-tracking/issues) using the **Bug Report** template. Include:
- Python version and OS
- Full error traceback
- Minimal reproduction steps

### Suggesting features
Open an issue using the **Feature Request** template before writing code — it's easier to discuss direction first.

### Submitting a pull request
1. Fork the repo and create a branch: `git checkout -b fix/my-fix`
2. Make your changes in `src/`
3. Add or update tests in `tests/` — new code needs test coverage
4. Ensure all checks pass:
   ```bash
   pytest
   pre-commit run --all-files
   ```
5. Open a PR against `master` with a clear description of *what* and *why*

## Code Style

This project uses **ruff** for linting and formatting (configured in `pyproject.toml`).
Pre-commit hooks run automatically on `git commit`. To run manually:

```bash
pre-commit run --all-files
```

Key conventions:
- Line length: 100 characters
- Docstrings: required on all public functions, classes, and modules
- Type hints: encouraged for new code
- No bare `except:` — always catch specific exceptions

## Testing

```bash
pytest                          # full suite
pytest -m "not integration"    # fast unit tests only
pytest tests/test_tracking_tongue.py  # specific file
pytest --cov=src                # with coverage
```

Tests live in `tests/`. Each source file in `src/` should have a corresponding `test_*.py`.

## Project Structure

```
src/           Python source scripts
tests/         pytest test suite
legacy/        Archived MATLAB scripts (read-only reference)
examples/      Usage examples
```

## Questions?

Open an issue or email kumaar324@gmail.com.
