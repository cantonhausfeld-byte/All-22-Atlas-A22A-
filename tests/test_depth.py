import pathlib
import re
import subprocess
import sys

import pandas as pd


OUTPUT_PATTERN = re.compile(
    r"lineups_(?P<stamp>\d{8}-\d{6})\.(?P<ext>parquet|csv)"
)


def _read_table(path: pathlib.Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def test_depth_respects_ordering():
    result = subprocess.run(
        [sys.executable, "-m", "a22a.roster.depth_logic"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    match = OUTPUT_PATTERN.search(result.stdout)
    assert match, f"lineups artifact missing from stdout: {result.stdout}"
    stamp = match.group("stamp")
    ext = match.group("ext")

    output_path = pathlib.Path("artifacts/roster") / f"lineups_{stamp}.{ext}"
    assert output_path.exists()

    depth = _read_table(output_path)

    required_cols = {
        "team_id",
        "position",
        "depth_role",
        "player_id",
        "depth_order",
        "replacement_player_id",
    }
    assert required_cols.issubset(depth.columns)

    ordered = depth.sort_values(["team_id", "position", "depth_order"])
    grouped = ordered.groupby(["team_id", "position"])

    for (_, _), group in grouped:
        assert (group["depth_order"].diff().fillna(1) >= 1).all()
        assert group["depth_order"].iloc[0] == 1
        starters = group[group["depth_role"] == "starter"]
        assert not starters.empty
        backups = group[group["depth_role"] != "starter"]
        if not backups.empty:
            first_backup = backups.iloc[0]
            assert first_backup["depth_order"] > starters.iloc[0]["depth_order"]
            replacement_id = first_backup["replacement_player_id"]
            if pd.notna(replacement_id) and replacement_id:
                assert str(replacement_id).startswith(first_backup["team_id"])
