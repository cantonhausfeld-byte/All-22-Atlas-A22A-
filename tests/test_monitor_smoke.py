import json
import pathlib
import shutil
import subprocess
import sys
from typing import Tuple


def _clear(path: pathlib.Path) -> None:
    if path.exists():
        if path.is_file():
            path.unlink()
        else:
            shutil.rmtree(path)


def _run_monitor() -> Tuple[pathlib.Path, dict[str, object], str]:
    outdir = pathlib.Path("artifacts/monitor")
    before = set(outdir.glob("health_*.json")) if outdir.exists() else set()
    result = subprocess.run(
        [sys.executable, "-m", "a22a.monitor.run"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    after = set(outdir.glob("health_*.json"))
    assert after, "monitor should create at least one health artifact"
    new = sorted(after - before, key=lambda p: p.stat().st_mtime)
    path = new[-1] if new else sorted(after, key=lambda p: p.stat().st_mtime)[-1]
    payload = json.loads(path.read_text())
    return path, payload, result.stdout


def test_monitor_handles_missing_artifacts(tmp_path):
    # remove upstream artifacts so the monitor must synthesise metrics
    for directory in [
        pathlib.Path("artifacts/meta"),
        pathlib.Path("artifacts/backtest"),
        pathlib.Path("artifacts/market"),
        pathlib.Path("artifacts/portfolio"),
        pathlib.Path("reports"),
    ]:
        _clear(directory)

    path, payload, stdout = _run_monitor()
    assert path.name.startswith("health_")
    assert "[monitor]" in stdout

    assert payload["status"] in {"ok", "warn", "fail"}
    assert isinstance(payload.get("metrics"), dict)
    for key in ["calibration_ece", "coverage", "clv", "slo"]:
        assert key in payload["metrics"], f"missing metric {key}"
        assert "status" in payload["metrics"][key]


def test_monitor_flags_high_ece(tmp_path):
    meta_dir = pathlib.Path("artifacts/meta")
    meta_dir.mkdir(parents=True, exist_ok=True)
    report_path = meta_dir / "calibration_report_99999999.json"
    report_path.write_text(
        json.dumps(
            {
                "ece": 0.12,
                "brier": 0.5,
                "conformal": {"binary": {"empirical": 0.82, "nominal": 0.9}},
            }
        ),
        encoding="utf-8",
    )

    _, payload, _ = _run_monitor()
    metric = payload["metrics"]["calibration_ece"]
    assert metric["status"] in {"warn", "fail"}
    assert payload["status"] in {"warn", "fail"}
