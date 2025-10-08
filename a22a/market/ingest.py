"""Market ingestion entrypoint (Phase 16)."""

from __future__ import annotations

import pathlib
import time
from typing import Any, Dict, Iterable, Sequence

import pandas as pd
import yaml

from . import clients as market_clients
from .clients.base import ClientResponse
from .normalize import NORMALIZED_COLUMNS, to_rows

CONFIG_PATH = pathlib.Path("configs/defaults.yaml")
ARTIFACT_DIR = pathlib.Path("artifacts/market")


CLIENTS: Dict[str, Any] = {
    "theodds": market_clients.theodds_client.get_events,
    "the_odds": market_clients.theodds_client.get_events,
    "the-odds": market_clients.theodds_client.get_events,
    "sgo": market_clients.sgo_client.get_events,
    "sports_game_odds": market_clients.sgo_client.get_events,
    "sports-game-odds": market_clients.sgo_client.get_events,
}


def _load_config(path: pathlib.Path = CONFIG_PATH) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _resolve_market_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    market_cfg = dict(cfg.get("market", {}))
    providers_cfg = market_cfg.get("providers", {})
    providers: list[str] = [name for name, enabled in providers_cfg.items() if bool(enabled)]
    if not providers:
        providers = ["theodds", "sgo"]
    leagues = market_cfg.get("leagues", ["NFL"])
    markets = market_cfg.get("markets", ["h2h", "spreads", "totals"])
    books = market_cfg.get(
        "books_allowlist",
        ["pinnacle", "circa", "betmgm", "fanduel", "draftkings"],
    )
    poll_window_hours = int(market_cfg.get("poll_window_hours", 168))
    rate_limit = int(market_cfg.get("rate_limit_per_min", 30))
    return {
        "providers": providers,
        "leagues": leagues,
        "markets": markets,
        "books": books,
        "poll_window_hours": poll_window_hours,
        "rate_limit_per_min": rate_limit,
    }


def _collect_provider_payloads(
    providers: Iterable[str],
    *,
    leagues: Sequence[str],
    markets: Sequence[str],
    books: Sequence[str],
    poll_window_hours: int,
    rate_limit_per_min: int,
) -> list[ClientResponse]:
    results: list[ClientResponse] = []
    for provider in providers:
        handler = CLIENTS.get(provider.lower())
        if handler is None:
            continue
        response = handler(
            leagues=leagues,
            markets=markets,
            books=books,
            poll_window_hours=poll_window_hours,
            rate_limit_per_min=rate_limit_per_min,
        )
        results.append(response)
    return results


def _build_snapshot(results: Iterable[ClientResponse]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for result in results:
        frame = to_rows(result.events, result.provider)
        if frame.empty:
            continue
        frame = frame.copy()
        frame["synthetic"] = bool(result.synthetic)
        frames.append(frame)
    if not frames:
        columns = NORMALIZED_COLUMNS + ["synthetic"]
        return pd.DataFrame(columns=columns)
    combined = pd.concat(frames, ignore_index=True)
    if "synthetic" not in combined.columns:
        combined["synthetic"] = True
    return combined


def _filter_books(df: pd.DataFrame, books: Sequence[str]) -> pd.DataFrame:
    if df.empty:
        return df
    allow = {str(book).lower() for book in books}
    mask = df["book"].str.lower().isin(allow)
    return df.loc[mask].reset_index(drop=True)


def _deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    dedup_cols = ["game_id", "book", "market", "selection", "ts"]
    if "synthetic" not in df.columns:
        df["synthetic"] = True
    df = df.copy()
    df["_synthetic_rank"] = df["synthetic"].astype(int)
    df = df.sort_values(["_synthetic_rank", "provider", "ts"])
    df = df.drop_duplicates(subset=dedup_cols, keep="first")
    df = df.sort_values(["game_id", "book", "market", "selection", "ts"]).reset_index(drop=True)
    df = df.drop(columns=["_synthetic_rank"])
    return df


def _enforce_schema(df: pd.DataFrame) -> pd.DataFrame:
    expected = NORMALIZED_COLUMNS + ["synthetic"]
    for column in expected:
        if column not in df.columns:
            df[column] = pd.Series(dtype="object")
    df = df[expected]
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    return df


def main() -> None:
    start = time.time()
    cfg = _load_config()
    market_cfg = _resolve_market_config(cfg)
    results = _collect_provider_payloads(
        market_cfg["providers"],
        leagues=market_cfg["leagues"],
        markets=market_cfg["markets"],
        books=market_cfg["books"],
        poll_window_hours=market_cfg["poll_window_hours"],
        rate_limit_per_min=market_cfg["rate_limit_per_min"],
    )
    snapshot = _build_snapshot(results)
    snapshot = _filter_books(snapshot, market_cfg["books"])
    snapshot = _deduplicate(snapshot)
    snapshot = _enforce_schema(snapshot)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = ARTIFACT_DIR / f"snapshots_{stamp}.parquet"
    snapshot.to_parquet(path, index=False)

    providers = ",".join(sorted({result.provider for result in results})) if results else "none"
    synthetic = sum(1 for result in results if result.synthetic)
    elapsed = time.time() - start
    print(
        "[market] wrote",
        path.name,
        f"rows={len(snapshot)}",
        f"providers={providers}",
        f"synthetic_sources={synthetic}",
        f"elapsed={elapsed:.2f}s",
    )


if __name__ == "__main__":
    main()

