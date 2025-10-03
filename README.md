# All-22 Atlas (A22A)

A22A is a world-model hyperensemble for NFL outcomes: odds-agnostic core signals, a fast drive-level simulator, and calibrated probabilities with selective precision and abstention.

## Quickstart
```bash
# Python 3.11 recommended
pip install -e .
make doctor     # smoke checks
make ingest     # stage data (no-op until sources wired)
make features   # build feature snapshots (stubs)
make train      # baseline (stub)
make sim        # simulator (stub)
```

Design
	•	Phases 1–16 with acceptance checklists in /docs (or Drive).
	•	No betting odds in core features/models (odds only for CLV/timing later).
	•	Doctor enforces seeds, SLO placeholders, and license/odds guardrails.

### `pyproject.toml`
```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "a22a"
version = "0.0.1"
requires-python = ">=3.11"
dependencies = [
  "numpy>=1.26",
  "pandas>=2.2",
  "polars>=1.6",
  "duckdb>=1.0",
  "scikit-learn>=1.5",
  "lightgbm>=4.3",
  "catboost>=1.2",
  "pyyaml>=6.0",
  "python-dotenv>=1.0",
  "meteostat>=1.6",
  "requests>=2.32",
  "tqdm>=4.66",
  "pytest>=8",
  "pytest-cov>=5",
]

[tool.pytest.ini_options]
addopts = "-q"
```
