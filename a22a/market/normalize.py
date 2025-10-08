"""Normalization utilities for provider payloads."""

from __future__ import annotations

import functools
import pathlib
import re
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

STAGED_GAMES_DIR = pathlib.Path("data/staged/games")

NORMALIZED_COLUMNS = [
    "event_id",
    "game_id",
    "provider",
    "book",
    "market",
    "selection",
    "line",
    "price",
    "ts",
    "home_team",
    "away_team",
]


def _normalize_team(team: str | None) -> str:
    if not team:
        return ""
    return re.sub(r"[^a-z0-9]", "", str(team).lower())


@functools.lru_cache(maxsize=1)
def _load_schedule_index() -> dict[tuple[str, str], str]:
    frames: list[pd.DataFrame] = []
    if STAGED_GAMES_DIR.exists():
        for path in sorted(STAGED_GAMES_DIR.rglob("*.parquet")):
            try:
                df = pd.read_parquet(path, columns=["game_id", "home_team", "away_team"])
            except Exception:  # pragma: no cover - corrupt partition
                continue
            if not df.empty:
                frames.append(df)
    if not frames:
        try:  # fallback to bundled sample schedule
            from a22a.data import sample_data

            sample = sample_data.sample_games().to_pandas()
            frames.append(sample[["game_id", "home_team", "away_team"]])
        except Exception:  # pragma: no cover - defensive
            return {}
    combined = pd.concat(frames, ignore_index=True)
    index: dict[tuple[str, str], str] = {}
    for row in combined.itertuples(index=False):
        home = _normalize_team(getattr(row, "home_team", None))
        away = _normalize_team(getattr(row, "away_team", None))
        game_id = getattr(row, "game_id", "")
        if not game_id or not home or not away:
            continue
        index[(home, away)] = str(game_id)
    return index


def _resolve_game_id(home: str | None, away: str | None, event_id: str) -> str:
    lookup = _load_schedule_index()
    home_norm = _normalize_team(home)
    away_norm = _normalize_team(away)
    if home_norm and away_norm:
        direct = lookup.get((home_norm, away_norm))
        if direct:
            return direct
        flipped = lookup.get((away_norm, home_norm))
        if flipped:
            return flipped
    return event_id


def _canonical_selection(
    selection: str | None,
    *,
    market: str,
    home_team: str | None,
    away_team: str | None,
) -> str:
    raw = (selection or "").strip().lower()
    if raw in {"home", "away", "over", "under"}:
        return raw
    home_norm = _normalize_team(home_team)
    away_norm = _normalize_team(away_team)
    sel_norm = _normalize_team(raw)
    if market == "h2h":
        if sel_norm and home_norm and sel_norm == home_norm:
            return "home"
        if sel_norm and away_norm and sel_norm == away_norm:
            return "away"
    if market == "spreads":
        if raw.endswith("home"):
            return "home"
        if raw.endswith("away"):
            return "away"
    if market == "totals":
        if "over" in raw:
            return "over"
        if "under" in raw:
            return "under"
    return raw or "unknown"


def _coerce_price(price: Any) -> int | None:
    try:
        return int(float(price))
    except (TypeError, ValueError):
        return None


def _coerce_line(line: Any) -> float:
    try:
        return float(line)
    except (TypeError, ValueError):
        return 0.0


def _iter_books(event: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    books = event.get("books") or event.get("bookmakers") or []
    for book in books:
        if isinstance(book, Mapping):
            yield book


def _iter_markets(book: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    markets = book.get("markets") or []
    for market in markets:
        if isinstance(market, Mapping):
            yield market


def _iter_outcomes(market: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    outcomes = market.get("outcomes") or market.get("offers") or []
    for outcome in outcomes:
        if isinstance(outcome, Mapping):
            yield outcome


def to_rows(payload: Sequence[Mapping[str, Any]] | Mapping[str, Any], provider: str) -> pd.DataFrame:
    if isinstance(payload, Mapping):
        events_iter = payload.get("events") or payload.get("data") or []
    else:
        events_iter = payload

    rows: list[dict[str, Any]] = []
    for event in events_iter or []:
        if not isinstance(event, Mapping):
            continue
        event_id = str(event.get("event_id") or event.get("id") or event.get("game_id") or "").strip()
        if not event_id:
            continue
        home_team = event.get("home_team") or event.get("home")
        away_team = event.get("away_team") or event.get("away")
        game_id = str(event.get("game_id") or _resolve_game_id(home_team, away_team, event_id))
        commence = event.get("commence_time") or event.get("start_time")
        base_ts = pd.to_datetime(commence, utc=True, errors="coerce")
        for book in _iter_books(event):
            book_key = str(book.get("book") or book.get("key") or book.get("title") or "").lower()
            if not book_key:
                book_key = "unknown"
            for market in _iter_markets(book):
                market_key = str(market.get("market") or market.get("key") or "").lower()
                if not market_key:
                    continue
                for outcome in _iter_outcomes(market):
                    selection = _canonical_selection(
                        outcome.get("selection") or outcome.get("name") or outcome.get("type"),
                        market=market_key,
                        home_team=home_team,
                        away_team=away_team,
                    )
                    price = _coerce_price(outcome.get("price") or outcome.get("odds") or outcome.get("american"))
                    if price is None:
                        continue
                    line = _coerce_line(outcome.get("line") or outcome.get("point") or outcome.get("total"))
                    ts = outcome.get("timestamp") or outcome.get("last_update") or commence
                    ts_parsed = pd.to_datetime(ts, utc=True, errors="coerce")
                    if pd.isna(ts_parsed):
                        ts_parsed = base_ts
                    rows.append(
                        {
                            "event_id": event_id,
                            "game_id": game_id,
                            "provider": provider,
                            "book": book_key,
                            "market": market_key,
                            "selection": selection,
                            "line": line,
                            "price": price,
                            "ts": ts_parsed,
                            "home_team": home_team,
                            "away_team": away_team,
                        }
                    )

    if not rows:
        return pd.DataFrame(columns=NORMALIZED_COLUMNS)

    df = pd.DataFrame(rows, columns=NORMALIZED_COLUMNS)
    df = df.dropna(subset=["ts"]).sort_values(["game_id", "book", "market", "selection", "ts"]).reset_index(drop=True)
    return df


__all__ = ["NORMALIZED_COLUMNS", "to_rows"]

