import pathlib
import re
import subprocess
import sys

import pandas as pd


OUTPUT_PATTERN = re.compile(
    r"availability_(?P<stamp>\d{8}-\d{6})\.(?P<ext>parquet|csv)"
)


def _read_table(path: pathlib.Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def test_injuries_outputs_are_bounded(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "a22a.health.injury_model"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    match = OUTPUT_PATTERN.search(result.stdout)
    assert match, f"availability artifact missing from stdout: {result.stdout}"
    stamp = match.group("stamp")
    ext = match.group("ext")

    availability_path = pathlib.Path("artifacts/health") / f"availability_{stamp}.{ext}"
    exit_path = availability_path.with_name(f"exit_hazards_{stamp}.{ext}")

    assert availability_path.exists(), "availability artifact missing"
    assert exit_path.exists(), "exit hazard artifact missing"

    availability = _read_table(availability_path)
    exit_hazards = _read_table(exit_path)

    assert {"player_id", "avail_prob"}.issubset(availability.columns)
    assert availability["avail_prob"].between(0.0, 1.0).all()

    assert "exit_hazard_pct" in exit_hazards.columns
    assert (exit_hazards["exit_hazard_pct"] >= 0).all()
