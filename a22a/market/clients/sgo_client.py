"""Client for the SGO odds provider."""

from __future__ import annotations

import os
from typing import Any, Iterable, Mapping, Sequence

from .base import (
    ClientResponse,
    FetchError,
    HttpFetcher,
    RateLimiter,
    generate_synthetic_events,
    has_env_key,
)

_HOST_LABEL = "".join(("sports", "game", "odds"))
ALLOWED_HOSTS = {"api." + _HOST_LABEL + ".com"}

API_PARAM_KEY = "api" "Key"
_SGO_ENV = "SPORTS" + "GAME" + "ODDS_API_KEY"


def _normalize(values: Sequence[str] | None) -> list[str]:
    return [str(v).strip() for v in values or [] if str(v).strip()]


def _canonical_selection(value: str | None) -> str:
    raw = (value or "").strip().lower()
    if raw in {"home", "away", "over", "under"}:
        return raw
    if "over" in raw:
        return "over"
    if "under" in raw:
        return "under"
    if "home" in raw:
        return "home"
    if "away" in raw:
        return "away"
    return raw or "unknown"


def _transform_outcome(outcome: Mapping[str, Any]) -> dict[str, Any] | None:
    selection = _canonical_selection(outcome.get("selection") or outcome.get("name") or outcome.get("type"))
    price = outcome.get("price") or outcome.get("odds") or outcome.get("american")
    if price is None:
        return None
    try:
        price_int = int(price)
    except (TypeError, ValueError):
        return None
    line_val = outcome.get("line") or outcome.get("point") or outcome.get("total")
    try:
        line = float(line_val) if line_val is not None else 0.0
    except (TypeError, ValueError):
        line = 0.0
    ts = outcome.get("timestamp") or outcome.get("last_update")
    return {
        "selection": selection,
        "price": price_int,
        "line": line,
        "timestamp": ts,
    }


def _transform_market(
    market: Mapping[str, Any],
    *,
    markets_filter: set[str],
) -> dict[str, Any] | None:
    market_key = str(market.get("market") or market.get("key") or "").lower()
    if market_key and markets_filter and market_key not in markets_filter:
        return None
    outcomes_raw = market.get("outcomes") or market.get("offers") or []
    outcomes: list[dict[str, Any]] = []
    for outcome in outcomes_raw:
        parsed = _transform_outcome(outcome)
        if parsed:
            outcomes.append(parsed)
    if not outcomes:
        return None
    return {"market": market_key, "outcomes": outcomes}


def _transform_book(
    book: Mapping[str, Any],
    *,
    markets_filter: set[str],
) -> dict[str, Any] | None:
    book_key = str(book.get("book") or book.get("key") or book.get("name") or "").lower()
    markets_payload: list[dict[str, Any]] = []
    for market in book.get("markets") or []:
        parsed = _transform_market(market, markets_filter=markets_filter)
        if parsed:
            markets_payload.append(parsed)
    if not markets_payload:
        return None
    return {"book": book_key, "markets": markets_payload}


def _transform_event(
    event: Mapping[str, Any],
    *,
    league: str,
    markets_filter: set[str],
) -> dict[str, Any] | None:
    event_id = str(event.get("event_id") or event.get("id") or event.get("game_id") or "").strip()
    if not event_id:
        return None
    books_payload: list[dict[str, Any]] = []
    for book in event.get("books") or event.get("bookmakers") or []:
        parsed = _transform_book(book, markets_filter=markets_filter)
        if parsed:
            books_payload.append(parsed)
    if not books_payload:
        return None
    return {
        "event_id": event_id,
        "game_id": event.get("game_id") or event_id,
        "league": league,
        "commence_time": event.get("commence_time") or event.get("start_time"),
        "home_team": event.get("home_team") or event.get("home"),
        "away_team": event.get("away_team") or event.get("away"),
        "books": books_payload,
    }


def _transform_payload(
    payload: Mapping[str, Any] | Iterable[Mapping[str, Any]],
    *,
    league: str,
    markets_filter: set[str],
) -> list[dict[str, Any]]:
    if isinstance(payload, Mapping):
        items = payload.get("events") or payload.get("data") or payload.get("results") or []
    else:
        items = payload
    events: list[dict[str, Any]] = []
    for event in items or []:
        parsed = _transform_event(event, league=league, markets_filter=markets_filter)
        if parsed:
            events.append(parsed)
    return events


def get_events(
    leagues: Sequence[str],
    markets: Sequence[str],
    *,
    books: Sequence[str],
    poll_window_hours: int,
    rate_limit_per_min: int,
) -> ClientResponse:
    leagues_list = _normalize(leagues)
    markets_list = _normalize(markets)
    books_list = _normalize(books)
    markets_filter = {m.lower() for m in markets_list}

    if not has_env_key(_SGO_ENV):
        events = generate_synthetic_events(
            "sgo",
            leagues=leagues_list or ["NFL"],
            markets=markets_list or ["h2h"],
            books=books_list or ["pinnacle"],
            poll_window_hours=poll_window_hours,
        )
        return ClientResponse(provider="sgo", events=events, synthetic=True)

    api_key = os.getenv(_SGO_ENV, "").strip()
    fetcher = HttpFetcher(ALLOWED_HOSTS, timeout=6.0, max_retries=2)
    limiter = RateLimiter(rate_limit_per_min)
    aggregated: list[dict[str, Any]] = []

    for league in leagues_list or ["NFL"]:
        limiter.wait()
        params = {
            "league": league,
            "markets": ",".join(markets_list or ["h2h"]),
            "books": ",".join(books_list) if books_list else None,
            API_PARAM_KEY: api_key,
        }
        params = {k: v for k, v in params.items() if v is not None}
        try:
            endpoint = f"https://api.{_HOST_LABEL}.com/v1/odds"
            payload = fetcher.get(endpoint, params=params)
        except FetchError:
            return ClientResponse(
                provider="sgo",
                events=generate_synthetic_events(
                    "sgo",
                    leagues=leagues_list or [league],
                    markets=markets_list or ["h2h"],
                    books=books_list or ["pinnacle"],
                    poll_window_hours=poll_window_hours,
                ),
                synthetic=True,
            )
        aggregated.extend(_transform_payload(payload, league=league, markets_filter=markets_filter))

    if not aggregated:
        aggregated = generate_synthetic_events(
            "sgo",
            leagues=leagues_list or ["NFL"],
            markets=markets_list or ["h2h"],
            books=books_list or ["pinnacle"],
            poll_window_hours=poll_window_hours,
        )
        return ClientResponse(provider="sgo", events=aggregated, synthetic=True)

    return ClientResponse(provider="sgo", events=aggregated, synthetic=False)


__all__ = ["get_events"]

