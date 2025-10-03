#!/usr/bin/env bash
set -euo pipefail

python -m pip install --upgrade pip wheel

# Core runtime (fast ETL + modeling + sim)
pip install "polars>=1.6" duckdb "lightgbm>=4.3" "catboost>=1.2" \
            "scikit-learn>=1.5" numpy scipy pandas \
            "pyyaml>=6.0" "pydantic>=2" python-dotenv \
            "meteostat>=1.6" requests \
            "numba>=0.59" "tqdm>=4.66" \
            "pytest>=8" "pytest-cov>=5"

# Optional for faster math kernels (pure CPU ok)
pip install "torch>=2.3; platform_system!='Windows'"

# If repo has pyproject/requirements, install the package itself
if [ -f "pyproject.toml" ]; then
  pip install -e .
elif [ -f "requirements.txt" ]; then
  pip install -r requirements.txt
fi

# Smoke import to catch wheel issues early
python - <<'PY'
import polars, duckdb, lightgbm, catboost, sklearn, numpy, meteostat
print("Deps OK")
PY
