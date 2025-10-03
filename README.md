# All-22 Atlas (A22A)

All-22 Atlas (A22A) is a research playground for exploring American football analytics with discipline.
This bootstrap focuses on the first six phases of the platform so contributors can extend data, features,
and models without sacrificing governance or reproducibility.

## Phase Overview

| Phase | Focus | Key Artefacts |
| --- | --- | --- |
| 1 | Scaffolding & Governance | Doctor checks, run registry, seed policy, CI harness |
| 2 | Data Spine | `staged/` contract stubs, ingestion harness, schema enforcement |
| 3 | Feature Store v1 | Polars lazy pipelines, leakage guard, PSI drift hook |
| 4 | Baseline Model | Forward-chaining CV stub, calibration artefact placeholders |
| 5 | Bayesian Team Strength | Recency decay configuration, dummy posterior export |
| 6 | Simulation Engine v1 | QMC/CI hooks, slate orchestration scaffolding |

## Quickstart

1. **Environment** – Install Python 3.11 and create a virtual environment (e.g. `python -m venv .venv && source .venv/bin/activate`).
2. **Dependencies** – Install the project in editable mode: `pip install -e .[dev]`.
3. **Secrets** – Copy `.env.sample` to `.env` and populate the required API keys.
4. **Doctor** – Run `make doctor` to validate governance, seeds, and placeholder SLOs.
5. **Workflow Targets** – Use the remaining Make targets (`make ingest`, `make features`, `make train`, `make sim`, `make report`) as you flesh out each phase.
6. **CI** – Push changes and rely on the GitHub Actions workflow to run the doctor and unit tests.

## Repository Layout

```
a22a/                  Python package with tools, data, features, models, sim, and reports
configs/defaults.yaml  Centralised configuration (seeds, SLOs, knobs)
staged/                Scratch area for intermediate artefacts (tracked empty)
Makefile               Workflow entry points
```

## Governance Checklist

- Seeds and configuration are centrally managed in YAML.
- Secrets are injected via environment variables only.
- Odds provider domains are explicitly banned from feature/model code.
- SLOs for ETL, feature engineering, modelling, and simulation are asserted in the doctor.
- Run registry entries ensure every execution is traceable.

## Development Tips

- Keep modules pure and side-effect free until explicitly required by later phases.
- Extend tests alongside the doctor to guard contracts as they evolve.
- Update `configs/defaults.yaml` when introducing new knobs; never hardcode values in code.

