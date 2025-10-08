"""Bootstrap logic for Phase 13 player impact modeling."""

from __future__ import annotations

import pathlib
import time
from typing import Any

import numpy as np
import pandas as pd
import yaml

from a22a.metrics import summarize_player_metric

CONFIG_PATH = pathlib.Path("configs/defaults.yaml")
ARTIFACT_DIR = pathlib.Path("artifacts/impact")


def _load_config(path: pathlib.Path = CONFIG_PATH) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    cfg = yaml.safe_load(path.read_text()) or {}
    return cfg.get("impact", {})


def _simulate_player(player_id: str, samples: int, ci_level: float, rng: np.random.Generator) -> dict[str, Any]:
    win_samples = rng.normal(0.0, 0.01, size=samples)
    margin_samples = rng.normal(0.0, 0.5, size=samples)
    total_samples = rng.normal(0.0, 0.6, size=samples)

    summary = summarize_player_metric(win_samples, margin_samples, total_samples, ci_level)
    summary.update({
        "player_id": player_id,
        "samples": samples,
    })
    return summary


def _write_artifact(df: pd.DataFrame) -> pathlib.Path:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_path = ARTIFACT_DIR / f"player_impact_{stamp}.parquet"
    try:
        df.to_parquet(out_path, index=False)
        return out_path
    except Exception:
        fallback = out_path.with_suffix(".csv")
        df.to_csv(fallback, index=False)
        return fallback


def main() -> None:
    start = time.time()
    cfg = _load_config()

    samples = int(cfg.get("samples_per_player", 200))
    ci_level = float(cfg.get("ci_level", 0.90))
    seed = int(cfg.get("seed", 13))

    rng = np.random.default_rng(seed)
    players = [f"P{i:05d}" for i in range(50)]

    records = [_simulate_player(pid, samples, ci_level, rng) for pid in players]
    df = pd.DataFrame.from_records(records)

    artifact_path = _write_artifact(df)
    duration = time.time() - start
    print(
        f"[impact] wrote {artifact_path.name} with {len(df)} players in {duration:.2f}s"
    )


if __name__ == "__main__":
    main()
