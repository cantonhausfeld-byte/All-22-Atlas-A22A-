"""Bootstrap scaffolding for injury availability and exit risk modeling."""

from __future__ import annotations

import pathlib
import time
from typing import Dict, Iterable

import numpy as np
import pandas as pd
import yaml

CONFIG_PATH = pathlib.Path("configs/defaults.yaml")
ARTIFACT_DIR = pathlib.Path("artifacts/health")


def load_health_config() -> Dict[str, object]:
    """Load the health configuration block from the defaults file."""
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"missing config file: {CONFIG_PATH}")
    config = yaml.safe_load(CONFIG_PATH.read_text()) or {}
    return config.get("health", {})


def build_player_pool(num_players: int = 256) -> Iterable[str]:
    """Return a deterministic list of player identifiers for stub generation."""
    return [f"P{i:05d}" for i in range(num_players)]


def simulate_availability(players: Iterable[str], rng: np.random.Generator) -> pd.DataFrame:
    """Produce a table of player availability probabilities bounded to [0, 1]."""
    players = list(players)
    availability = rng.uniform(0.72, 0.97, size=len(players))
    df = pd.DataFrame(
        {
            "player_id": players,
            "season": 2024,
            "week": 1,
            "avail_prob": np.clip(availability, 0.0, 1.0),
            "source": "bootstrap_stub",
        }
    )
    return df


def simulate_exit_hazards(players: Iterable[str], rng: np.random.Generator) -> pd.DataFrame:
    """Generate in-game exit hazard percentages that are guaranteed to be non-negative."""
    players = list(players)
    baseline = rng.uniform(0.3, 4.2, size=len(players))
    df = pd.DataFrame(
        {
            "player_id": players,
            "season": 2024,
            "week": 1,
            "exit_hazard_pct": np.maximum(baseline, 0.0),
            "events_considered": rng.integers(10, 50, size=len(players)),
        }
    )
    return df


def write_artifact(df: pd.DataFrame, prefix: str, timestamp: str) -> pathlib.Path:
    """Persist the dataframe to parquet with a CSV fallback."""
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = ARTIFACT_DIR / f"{prefix}_{timestamp}.parquet"
    try:
        df.to_parquet(out_path, index=False)
        return out_path
    except Exception:
        csv_path = out_path.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        return csv_path


def main() -> None:
    """Entry point that builds stub availability and exit hazard tables."""
    start = time.time()
    cfg = load_health_config()

    recency_half_life = cfg.get("recency_half_life_weeks", 8)
    min_events = max(int(cfg.get("min_events", 50)), 1)

    rng_seed = min_events + int(recency_half_life)
    rng = np.random.default_rng(rng_seed)

    players = build_player_pool()
    availability = simulate_availability(players, rng)
    exit_hazards = simulate_exit_hazards(players, rng)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    availability_path = write_artifact(availability, "availability", timestamp)
    exit_path = write_artifact(exit_hazards, "exit_hazards", timestamp)

    duration = time.time() - start
    print(
        "[injuries] wrote %s, %s (frailty=%s, half_life=%s, min_events=%s) in %.2fs"
        % (
            availability_path.name,
            exit_path.name,
            bool(cfg.get("frailty", True)),
            recency_half_life,
            min_events,
            duration,
        )
    )


if __name__ == "__main__":
    main()
