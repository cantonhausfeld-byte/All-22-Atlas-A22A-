"""Bootstrap scaffolding for roster depth and replacement planning."""

from __future__ import annotations

import pathlib
import time
from typing import Dict, Iterable, List, Sequence

import pandas as pd
import yaml

CONFIG_PATH = pathlib.Path("configs/defaults.yaml")
ARTIFACT_DIR = pathlib.Path("artifacts/roster")
DEFAULT_POSITIONS: Sequence[str] = (
    "QB",
    "RB",
    "WR",
    "TE",
    "LT",
    "RT",
    "CB",
    "S",
    "DL",
    "LB",
)


def load_roster_config() -> Dict[str, object]:
    """Load the roster configuration block from the defaults file."""
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"missing config file: {CONFIG_PATH}")
    config = yaml.safe_load(CONFIG_PATH.read_text()) or {}
    return config.get("roster", {})


def build_depth_chart(
    teams: Iterable[str],
    positions: Sequence[str],
    fallback_by_position: Dict[str, List[str]],
) -> pd.DataFrame:
    """Construct a stub depth chart with deterministic ordering and replacements."""
    rows = []
    for team in teams:
        for pos in positions:
            depth_players = [f"{team}_{pos}_{slot}" for slot in ("1", "2")]
            for idx, player_id in enumerate(depth_players, start=1):
                if idx < len(depth_players):
                    replacement_player = depth_players[idx]
                    replacement_role = "internal"
                else:
                    fallback_roles = fallback_by_position.get(pos, [])
                    replacement_player = (
                        f"{team}_{fallback_roles[0]}_1" if fallback_roles else None
                    )
                    replacement_role = "fallback" if replacement_player else None
                rows.append(
                    {
                        "team_id": team,
                        "position": pos,
                        "depth_role": "starter" if idx == 1 else "primary_backup",
                        "player_id": player_id,
                        "depth_order": idx,
                        "replacement_player_id": replacement_player,
                        "replacement_type": replacement_role,
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    """Entry point that materialises the depth chart stub to disk."""
    start = time.time()
    cfg = load_roster_config()

    fallback_by_position = cfg.get("fallback_by_position", {})
    fallback_by_position = {
        key: list(value) for key, value in fallback_by_position.items()
    }

    teams = [f"TEAM{idx:02d}" for idx in range(6)]
    positions = list(DEFAULT_POSITIONS)
    depth_chart = build_depth_chart(teams, positions, fallback_by_position)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_path = ARTIFACT_DIR / f"lineups_{timestamp}.parquet"
    try:
        depth_chart.to_parquet(output_path, index=False)
    except Exception:
        output_path = output_path.with_suffix(".csv")
        depth_chart.to_csv(output_path, index=False)

    elapsed = time.time() - start
    print(
        "[depth] wrote %s with %d rows (min_snaps=%s, qb_penalty=%s) in %.2fs"
        % (
            output_path.name,
            len(depth_chart),
            cfg.get("min_snaps_for_starter", "n/a"),
            cfg.get("qb_replacement_penalty", "n/a"),
            elapsed,
        )
    )


if __name__ == "__main__":
    main()
