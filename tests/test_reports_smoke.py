"""Bootstrap smoke coverage for the reports module."""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_reports_smoke(tmp_path: pathlib.Path) -> None:
    (tmp_path / "artifacts").mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH")
    repo_path = str(PROJECT_ROOT)
    env["PYTHONPATH"] = repo_path if not pythonpath else os.pathsep.join([repo_path, pythonpath])

    result = subprocess.run(
        [sys.executable, "-m", "a22a.reports.smoke"],
        capture_output=True,
        text=True,
        check=False,
        cwd=tmp_path,
        env=env,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    reports_dir = tmp_path / "reports"
    assert reports_dir.exists(), "reports/ directory should be created"
    assert (reports_dir / "summary.json").exists(), "summary.json must be written"
    assert any(reports_dir.glob("*.png")), "at least one chart should be written"
