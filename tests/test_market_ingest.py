import os
import pathlib
import subprocess
import sys

import pandas as pd
import pandas.testing as pdt


def _latest_artifact(directory: pathlib.Path, pattern: str) -> pathlib.Path | None:
    candidates = sorted(directory.glob(pattern))
    return candidates[-1] if candidates else None


def _run_ingest(env: dict[str, str]) -> pathlib.Path:
    artifacts_dir = pathlib.Path("artifacts/market")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    before = set(artifacts_dir.glob("snapshots_*.parquet"))
    result = subprocess.run(
        [sys.executable, "-m", "a22a.market.ingest"],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    assert "wrote" in result.stdout
    after = set(artifacts_dir.glob("snapshots_*.parquet"))
    new_files = sorted(after - before)
    path = new_files[-1] if new_files else _latest_artifact(artifacts_dir, "snapshots_*.parquet")
    assert path is not None
    return path


def test_market_ingest_snapshot_offline(tmp_path):
    env = {**os.environ, "ODDS_API_KEY": "", "SPORTSGAMEODDS_API_KEY": ""}
    first = _run_ingest(env)
    second = _run_ingest(env)
    df_first = pd.read_parquet(first).sort_values(["game_id", "book", "market", "selection", "ts"]).reset_index(drop=True)
    df_second = pd.read_parquet(second).sort_values(["game_id", "book", "market", "selection", "ts"]).reset_index(drop=True)
    pdt.assert_frame_equal(df_first, df_second)

    required = {
        "event_id",
        "game_id",
        "provider",
        "book",
        "market",
        "selection",
        "line",
        "price",
        "ts",
        "home_team",
        "away_team",
        "synthetic",
    }
    assert required.issubset(df_first.columns)
    assert len(df_first) > 0
    assert set(df_first["provider"].unique()).issubset({"theodds", "sgo"})
