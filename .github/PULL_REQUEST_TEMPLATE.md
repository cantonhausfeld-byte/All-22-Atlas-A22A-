## A22A — PR Checklist (Phases 1–6)
- [ ] Phase 1 — Scaffolding & Governance: doctor < 30s; run registry & seed policy present
- [ ] Phase 2 — Data Spine: staged Parquet + schema/join checks pass
- [ ] Phase 3 — Feature Store v1: lazy Polars features; leakage guard; PSI hook
- [ ] Phase 4 — Baseline Model: purged forward CV; calibrated probs; no odds
- [ ] Phase 5 — Team Strength: θ (mean±CI) produced; integrated as features
- [ ] Phase 6 — Sim v1: QMC + early-stop by CI width; fair ladders emitted

### What changed
-

### Why
-

### Evidence (paste logs/plots)
- `make doctor` output
- `make train` metrics (Brier/LogLoss/ECE)
- `make sim` sample distributions

