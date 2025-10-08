import os
import pathlib
import subprocess
import sys

import numpy as np
import pandas as pd


def _latest_artifact(directory: pathlib.Path, pattern: str) -> pathlib.Path | None:
    candidates = sorted(directory.glob(pattern))
    return candidates[-1] if candidates else None


def test_clv_smoke(tmp_path):
    artifacts_dir = pathlib.Path("artifacts/market")
    env = {**os.environ, "ODDS_API_KEY": "", "SPORTSGAMEODDS_API_KEY": ""}
    subprocess.run([sys.executable, "-m", "a22a.market.ingest"], check=True, env=env)
    before = set(artifacts_dir.glob("clv_*.parquet")) if artifacts_dir.exists() else set()
    result = subprocess.run([sys.executable, "-m", "a22a.market.clv"], capture_output=True, text=True, check=True)
    assert "wrote" in result.stdout
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    after = set(artifacts_dir.glob("clv_*.parquet"))
    new_files = sorted(after - before)
    path = new_files[-1] if new_files else _latest_artifact(artifacts_dir, "clv_*.parquet")
    assert path is not None
    df = pd.read_parquet(path)
    required = {
        "game_id",
        "book",
        "market",
        "selection",
        "open_price",
        "close_price",
        "open_line",
        "close_line",
        "clv_bps",
    }
    assert required.issubset(df.columns)
    if not df.empty:
        assert np.isfinite(df["clv_bps"]).all()
        assert np.isfinite(df["open_prob"]).all()
        assert np.isfinite(df["close_prob"]).all()
        assert not np.allclose(df["clv_bps"], 0.0)
