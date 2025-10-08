import pathlib
import subprocess
import sys

import pandas as pd


def _latest_artifact(directory: pathlib.Path, pattern: str) -> pathlib.Path | None:
    candidates = sorted(directory.glob(pattern))
    return candidates[-1] if candidates else None


def test_market_ingest_snapshot(tmp_path):
    artifacts_dir = pathlib.Path("artifacts/market")
    before = set(artifacts_dir.glob("snapshots_*.parquet")) if artifacts_dir.exists() else set()
    result = subprocess.run([sys.executable, "-m", "a22a.market.ingest"], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "wrote" in result.stdout
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    after = set(artifacts_dir.glob("snapshots_*.parquet"))
    new_files = sorted(after - before)
    path = new_files[-1] if new_files else _latest_artifact(artifacts_dir, "snapshots_*.parquet")
    assert path is not None
    df = pd.read_parquet(path)
    required = {"game_id", "book", "market", "selection", "line", "price", "ts", "fair_delta_bps"}
    assert required.issubset(df.columns)
    assert len(df) > 0
