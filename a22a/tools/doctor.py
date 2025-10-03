"""Governance doctor for All-22 Atlas."""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping

import yaml

REQUIRED_ENV_KEYS = (
    "OPENAI_API_KEY",
    "ODDS_API_KEY",
    "SPORTSGAMEODDS_API_KEY",
    "WEATHER_API_KEY",
)

ODDS_DOMAINS = (
    "theoddsapi.com",
    "oddsapi.com",
    "sportsgameodds.com",
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"
FEATURE_DIR = REPO_ROOT / "a22a" / "features"
PACKAGE_DIR = REPO_ROOT / "a22a"


@dataclass
class SLORecord:
    """Represents a single SLO budget check."""

    name: str
    budget_seconds: float


class SLOTimer:
    """Context manager enforcing placeholder SLOs."""

    def __init__(self, record: SLORecord) -> None:
        self.record = record
        self._start = 0.0

    def __enter__(self) -> "SLOTimer":
        self._start = time.perf_counter()
        print(f"[doctor] starting SLO window for {self.record.name} (budget={self.record.budget_seconds}s)")
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        elapsed = time.perf_counter() - self._start
        status = "OK" if elapsed <= self.record.budget_seconds else "VIOLATION"
        print(f"[doctor] completed {self.record.name} in {elapsed:.4f}s -> {status}")
        if elapsed > self.record.budget_seconds:
            raise RuntimeError(f"SLO violation for {self.record.name}: {elapsed:.4f}s > {self.record.budget_seconds}s")
        return False


def _load_yaml(path: Path) -> Dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Expected config at {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config {path} must decode to a mapping")
    return data


def compute_seed_hashes(seeds: Mapping[str, str]) -> Dict[str, str]:
    """Return short SHA256 digests for configured seeds."""
    hashes: Dict[str, str] = {}
    for name, value in seeds.items():
        digest = hashlib.sha256(str(value).encode("utf-8")).hexdigest()
        hashes[name] = digest[:12]
    return hashes


def validate_env(env: Mapping[str, str] | None = None) -> None:
    """Ensure required secrets are pulled from the environment."""
    if env is None:
        env = os.environ
    missing = [key for key in REQUIRED_ENV_KEYS if not env.get(key)]
    if missing:
        raise EnvironmentError(f"Missing required environment variables: {', '.join(sorted(missing))}")


_SECRET_ASSIGNMENT_PATTERNS = [re.compile(rf"{re.escape(key)}\s*=\s*['\"]") for key in REQUIRED_ENV_KEYS]


def check_secrets_not_hardcoded(base_path: Path = PACKAGE_DIR) -> None:
    """Fail if any known secret appears to be hardcoded in Python modules."""
    for path in base_path.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for pattern in _SECRET_ASSIGNMENT_PATTERNS:
            if pattern.search(text):
                raise RuntimeError(f"Potential hardcoded secret detected in {path}")


def scan_feature_code_for_odds(feature_path: Path = FEATURE_DIR) -> None:
    """Ensure odds provider domains are not referenced in feature engineering code."""
    for path in feature_path.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for domain in ODDS_DOMAINS:
            if domain in text:
                raise RuntimeError(f"Odds API domain '{domain}' found in {path}")


def ensure_run_registry() -> None:
    registry_path = REPO_ROOT / "staged" / "run_registry.log"
    if not registry_path.exists():
        registry_path.touch()
    print(f"[doctor] run registry available at {registry_path.relative_to(REPO_ROOT)}")


def render_report(ci: bool, seed_hashes: Mapping[str, str], slo_config: Mapping[str, object]) -> None:
    print("[doctor] seed digests:")
    for name, digest in seed_hashes.items():
        print(f"    - {name}: {digest}")
    slo_records = [
        SLORecord("etl", float(slo_config.get("etl_budget_seconds", 0))),
        SLORecord("feature_build", float(slo_config.get("feature_budget_seconds", 0))),
        SLORecord("training", float(slo_config.get("training_budget_seconds", 0))),
        SLORecord("simulation", float(slo_config.get("simulation_budget_seconds", 0))),
        SLORecord("reporting", float(slo_config.get("report_budget_seconds", 0))),
    ]
    for record in slo_records:
        with SLOTimer(record):
            time.sleep(0.001)
    if ci:
        print("[doctor] running in CI mode: reduced logging enabled")


def run(ci: bool = False) -> None:
    seeds = _load_yaml(CONFIG_DIR / "seeds.yaml")
    slo_config = _load_yaml(CONFIG_DIR / "slo.yaml")
    validate_env()
    check_secrets_not_hardcoded()
    scan_feature_code_for_odds()
    ensure_run_registry()
    hashes = compute_seed_hashes(seeds)
    render_report(ci, hashes, slo_config)
    print("[doctor] all checks passed")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run All-22 Atlas governance doctor")
    parser.add_argument("--ci", action="store_true", help="Enable CI mode")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        run(ci=args.ci)
    except Exception as exc:  # noqa: BLE001
        print(f"[doctor] failure: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
