"""Bootstrap stub for Phase 9 coach adaptation modeling."""

from __future__ import annotations

import pathlib
import time
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import yaml

CONFIG_PATH = pathlib.Path("configs/defaults.yaml")


def _load_config() -> Dict[str, Any]:
    with CONFIG_PATH.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _generate_stub_rows(strategy_cfg: Dict[str, Any]) -> pd.DataFrame:
    teams = [f"TEAM{i:02d}" for i in range(10)]
    rng = np.random.default_rng(90210)
    recency = strategy_cfg.get("recency_half_life_weeks", 6)
    min_samples = strategy_cfg.get("min_samples_per_coach", 200)
    tags: List[str] = list(strategy_cfg.get("aggressiveness_targets", [])) or ["PROE"]

    rows = []
    for idx, team in enumerate(teams):
        base = rng.uniform(-0.5, 0.5)
        adjustment = (recency / (min_samples or 1)) * 0.1
        agg_index = float(base + adjustment)
        rows.append(
            {
                "team_id": team,
                "coach_id": f"C_{team}",
                "agg_index": agg_index,
                "tempo_delta": float(rng.normal(0, 0.1)),
                "samples_seen": int(min_samples + idx * 5),
                "tags": ",".join(tags),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    start = time.time()
    cfg = _load_config()
    strategy_cfg = cfg.get("strategy", {})

    outdir = pathlib.Path("artifacts/strategy")
    outdir.mkdir(parents=True, exist_ok=True)

    df = _generate_stub_rows(strategy_cfg)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_path = outdir / f"coach_adapt_{stamp}.parquet"

    try:
        df.to_parquet(out_path, index=False)
    except Exception:  # pragma: no cover - fallback for environments without parquet deps
        out_path = out_path.with_suffix(".csv")
        df.to_csv(out_path, index=False)

    duration = time.time() - start
    print(
        f"[strategy] wrote {out_path} with {len(df)} rows in {duration:.2f}s "
        f"(targets={strategy_cfg.get('aggressiveness_targets', [])})"
    )


if __name__ == "__main__":
    main()
