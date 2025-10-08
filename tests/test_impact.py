import pathlib
import subprocess
import sys

import pandas as pd


def _read_artifact(path: pathlib.Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        try:
            return pd.read_parquet(path)
        except Exception:
            path = path.with_suffix(".csv")
    return pd.read_csv(path)


def test_impact_runs(tmp_path):
    outdir = pathlib.Path("artifacts/impact")
    outdir.mkdir(parents=True, exist_ok=True)
    before = set(outdir.glob("player_impact_*"))

    result = subprocess.run(
        [sys.executable, "-m", "a22a.impact.player_value"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "[impact]" in result.stdout

    after = set(outdir.glob("player_impact_*"))
    new_files = sorted(after - before, key=lambda p: p.stat().st_mtime)
    assert new_files, "Expected a new impact artifact to be created"

    artifact = new_files[-1]
    df = _read_artifact(artifact)
    required = {"player_id", "delta_win_pct", "delta_margin", "delta_total"}
    assert required.issubset(df.columns)
    assert len(df) > 0
