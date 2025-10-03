# All-22 Atlas (A22A)

All-22 Atlas is a modular analytics platform for orchestrating data ingestion, feature engineering, baseline modelling, and simulation workflows for American football research while maintaining strict governance over seeds, configs, and operational SLOs. This bootstrap repo ships lightweight scaffolding for the first six phases so teams can iterate safely before data arrives.

## Quickstart
1. Create a virtual environment with Python 3.11 and install dependencies via `pip install -e .`.
2. Copy `.env.sample` to `.env` and populate API keys.
3. Run `make doctor` to verify the environment and governance scaffolding.
4. Use the remaining make targets (`ingest`, `features`, `train`, `sim`, `report`) as you flesh out each phase.
