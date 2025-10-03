"""Governance doctor for All-22 Atlas (A22A)."""
from __future__ import annotations

import argparse
import hashlib
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

import numpy as np
import yaml

REQUIRED_ENV_KEYS: Sequence[str] = (
    "OPENAI_API_KEY",
    "ODDS_API_KEY",
    "SPORTSGAMEODDS_API_KEY",
    "WEATHER_API_KEY",
)

ODDS_DOMAINS: Sequence[str] = (
    "theoddsapi.com",
    "oddsapi.com",
    "sportsgameodds.com",
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs" / "defaults.yaml"
PACKAGE_DIR = REPO_ROOT / "a22a"
FEATURE_DIR = PACKAGE_DIR / "features"
MODEL_DIR = PACKAGE_DIR / "models"


@dataclass
class SLORecord:
    """Represents a single SLO budget check."""

    name: str
    budget_seconds: float
    description: str


class SLOTimer:
    """Context manager enforcing placeholder SLOs."""

    def __init__(self, record: SLORecord) -> None:
        self.record = record
        self._start = 0.0

    def __enter__(self) -> "SLOTimer":
        self._start = time.perf_counter()
        print(
            f"[doctor] starting SLO window for {self.record.name} "
            f"(budget={self.record.budget_seconds}s :: {self.record.description})"
        )
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        elapsed = time.perf_counter() - self._start
        status = "OK" if elapsed <= self.record.budget_seconds else "VIOLATION"
        print(
            f"[doctor] completed {self.record.name} in {elapsed:.4f}s -> {status}"
        )
        if elapsed > self.record.budget_seconds:
            raise RuntimeError(
                f"SLO violation for {self.record.name}: {elapsed:.4f}s > "
                f"{self.record.budget_seconds}s"
            )
        return False


def load_defaults(path: Path = CONFIG_PATH) -> Dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(
            f"Configuration file missing: {path.relative_to(REPO_ROOT)}"
        )
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError("defaults.yaml must decode to a mapping")
    return data


def compute_seed_hashes(seeds: Mapping[str, object]) -> Dict[str, str]:
    hashes: Dict[str, str] = {}
    for name, value in seeds.items():
        digest = hashlib.sha256(str(value).encode("utf-8")).hexdigest()
        hashes[name] = digest[:12]
    return hashes


def set_global_seeds(seeds: Mapping[str, object]) -> None:
    code_seed = int(seeds.get("code", 0))
    data_seed = int(seeds.get("data", code_seed))
    config_seed = int(seeds.get("config", code_seed))
    random.seed(code_seed)
    np.random.seed(data_seed)
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    run_id = f"{config_seed}-{timestamp}"
    print(
        f"[doctor] deterministic seeds set (code={code_seed}, data={data_seed}, config={config_seed})"
    )
    print(f"[doctor] run-id template active -> {run_id}")


def validate_env(env: Mapping[str, str] | None = None) -> None:
    if env is None:
        env = os.environ
    missing = [key for key in REQUIRED_ENV_KEYS if not env.get(key)]
    if missing:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(sorted(missing))}"
        )


_SECRET_ASSIGNMENT_PATTERNS = [
    re.compile(rf"{re.escape(key)}\s*=\s*['\"]") for key in REQUIRED_ENV_KEYS
]


def check_secrets_not_hardcoded(base_path: Path = PACKAGE_DIR) -> None:
    for path in base_path.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for pattern in _SECRET_ASSIGNMENT_PATTERNS:
            if pattern.search(text):
                raise RuntimeError(f"Potential hardcoded secret detected in {path}")


def scan_code_for_odds(paths: Sequence[Path]) -> None:
    for base_path in paths:
        for path in base_path.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            for domain in ODDS_DOMAINS:
                if domain in text:
                    try:
                        location = path.relative_to(REPO_ROOT)
                    except ValueError:
                        location = path
                    raise RuntimeError(
                        f"Odds API domain '{domain}' found in {location}"
                    )


def ensure_run_registry(staged_root: Path) -> None:
    staged_root.mkdir(parents=True, exist_ok=True)
    registry_path = staged_root / "run_registry.log"
    if not registry_path.exists():
        registry_path.touch()
    print(
        f"[doctor] run registry available at {registry_path.relative_to(REPO_ROOT)}"
    )


def ensure_seed_policy(seeds: Mapping[str, object]) -> None:
    required = {"code", "data", "config"}
    missing = required - set(seeds)
    if missing:
        raise RuntimeError(
            f"Seed policy incomplete; expected keys: {', '.join(sorted(required))}"
        )
    print(
        "[doctor] seed policy confirmed with keys -> "
        + ", ".join(sorted(seeds.keys()))
    )


def enforce_slos(slo_config: Mapping[str, object]) -> None:
    records = [
        SLORecord(
            "etl",
            float(slo_config.get("etl_seconds_per_season", 0)),
            "seasonal ETL budget",
        ),
        SLORecord(
            "feature_build",
            float(slo_config.get("feature_build_seconds", 0)),
            "feature engineering budget",
        ),
        SLORecord(
            "baseline_train",
            float(slo_config.get("baseline_train_seconds_per_season", 0)),
            "forward-chaining baseline training budget",
        ),
        SLORecord(
            "team_strength",
            float(slo_config.get("team_strength_seconds", 0)),
            "bayesian team strength budget",
        ),
        SLORecord(
            "simulation_full_slate",
            float(slo_config.get("simulation_full_slate_seconds", 0)),
            "256-1024 sims per game budget",
        ),
    ]
    for record in records:
        with SLOTimer(record):
            time.sleep(0.001)


def render_seed_hashes(hashes: Mapping[str, str]) -> None:
    print("[doctor] seed digests:")
    for name, digest in sorted(hashes.items()):
        print(f"    - {name}: {digest}")


def render_phase_matrix(phases: Sequence[object]) -> None:
    items = ", ".join(str(phase) for phase in phases)
    print(f"[doctor] phases prepared -> [{items}]")


def run(ci: bool = False) -> None:
    defaults = load_defaults()
    seeds = defaults.get("seeds", {})
    slo_config = defaults.get("slo", {})
    project_meta = defaults.get("project", {})

    ensure_seed_policy(seeds)
    set_global_seeds(seeds)
    validate_env()
    check_secrets_not_hardcoded()
    scan_code_for_odds([FEATURE_DIR, MODEL_DIR])

    staged_root = REPO_ROOT / str(defaults.get("contracts", {}).get("staged_root", "staged"))
    ensure_run_registry(staged_root)

    hashes = compute_seed_hashes(seeds)
    render_seed_hashes(hashes)
    enforce_slos(slo_config)
    render_phase_matrix(project_meta.get("phases", []))
    template = project_meta.get("run_id_template")
    if template:
        print(f"[doctor] run-id structure -> {template}")

    if ci:
        print("[doctor] running in CI mode -> concise logging enabled")

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
