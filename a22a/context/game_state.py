"""Bootstrap stub for Phase 10 dynamic game context engine."""

from __future__ import annotations

import pathlib
import time
from typing import Any, Dict

import numpy as np
import pandas as pd
import yaml

CONFIG_PATH = pathlib.Path("configs/defaults.yaml")


def _load_config() -> Dict[str, Any]:
    with CONFIG_PATH.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def incremental_update_stub(n_drives: int = 24) -> pd.DataFrame:
    rng = np.random.default_rng(2024)
    drives = np.arange(1, n_drives + 1)
    lead = np.cumsum(rng.integers(-3, 4, size=n_drives))
    pace = np.clip(28 + rng.normal(0, 3, size=n_drives), 20, 40)
    fatigue = np.clip(np.cumsum(np.abs(rng.normal(0, 0.6, size=n_drives))), 0, None)

    df = pd.DataFrame(
        {
            "drive": drives,
            "lead": lead,
            "lag": -lead,
            "pace_s_per_play": pace,
            "fatigue_proxy": fatigue,
        }
    )
    return df


def main() -> None:
    start = time.time()
    cfg = _load_config()
    context_cfg = cfg.get("context", {})

    outdir = pathlib.Path("artifacts/context")
    outdir.mkdir(parents=True, exist_ok=True)

    df = incremental_update_stub()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_path = outdir / f"state_{stamp}.parquet"

    try:
        df.to_parquet(out_path, index=False)
    except Exception:  # pragma: no cover - fallback when parquet deps absent
        out_path = out_path.with_suffix(".csv")
        df.to_csv(out_path, index=False)

    latency = time.time() - start
    budget = float(context_cfg.get("update_latency_budget_s", 1.0))
    partial_budget_ms = float(context_cfg.get("partial_sim_max_ms", 200))

    print(f"[context] wrote {out_path} in {latency:.3f}s (budget {budget}s)")
    print(
        f"[context] partial-sim hook: OK (budget {partial_budget_ms}ms, stubbed update)"
    )


if __name__ == "__main__":
    main()
