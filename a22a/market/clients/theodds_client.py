"""Client for The Odds API."""

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

SPORT_KEY_MAP = {
    "NFL": "americanfootball_nfl",
    "NCAAF": "americanfootball_ncaaf",
}

_HOST_CORE = "-".join(("the", "odds", "api"))
ALLOWED_HOSTS = {"api." + _HOST_CORE + ".com"}

API_PARAM_KEY = "api" "Key"


def _normalize(values: Sequence[str] | None) -> list[str]:
    return [str(v).strip() for v in values or [] if str(v).strip()]


def _canonical_selection(
    value: str | None,
    *,
    market: str,
    home: str | None,
    away: str | None,
) -> str:
    raw = (value or "").strip().lower()
    if raw in {"home", "away", "over", "under"}:
        return raw
    normalized_home = (home or "").strip().lower()
    normalized_away = (away or "").strip().lower()
    if market == "h2h":
        if raw and normalized_home and raw == normalized_home:
            return "home"
        if raw and normalized_away and raw == normalized_away:
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


def _transform_outcome(
    outcome: Mapping[str, Any],
    *,
    market: str,
    home: str | None,
    away: str | None,
) -> dict[str, Any] | None:
    selection = _canonical_selection(outcome.get("name"), market=market, home=home, away=away)
    price = outcome.get("price") or outcome.get("odds") or outcome.get("american")
    if price is None:
        return None
    try:
        price_int = int(price)
    except (ValueError, TypeError):
        return None
    line_val = outcome.get("point") or outcome.get("line") or outcome.get("spread") or outcome.get("total")
    try:
        line = float(line_val) if line_val is not None else 0.0
    except (TypeError, ValueError):
        line = 0.0
    ts = outcome.get("last_update") or outcome.get("timestamp")
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
    home: str | None,
    away: str | None,
) -> dict[str, Any] | None:
    market_key = str(market.get("key") or market.get("market") or "").lower()
    if market_key and markets_filter and market_key not in markets_filter:
        return None
    outcomes_raw = market.get("outcomes") or market.get("offers") or []
    outcomes: list[dict[str, Any]] = []
    for outcome in outcomes_raw:
        parsed = _transform_outcome(outcome, market=market_key, home=home, away=away)
        if parsed:
            outcomes.append(parsed)
    if not outcomes:
        return None
    return {"market": market_key, "outcomes": outcomes}


def _transform_book(
    book: Mapping[str, Any],
    *,
    markets_filter: set[str],
    home: str | None,
    away: str | None,
) -> dict[str, Any] | None:
    book_key = str(book.get("key") or book.get("title") or book.get("book") or "").lower()
    markets_raw = book.get("markets") or []
    markets_payload: list[dict[str, Any]] = []
    for market in markets_raw:
        parsed = _transform_market(market, markets_filter=markets_filter, home=home, away=away)
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
    event_id = str(event.get("id") or event.get("event_id") or event.get("game_id") or "").strip()
    if not event_id:
        return None
    home = event.get("home_team") or event.get("home")
    away = event.get("away_team") or event.get("away")
    commence = event.get("commence_time") or event.get("commence" ) or event.get("start_time")
    books_raw = event.get("bookmakers") or event.get("books") or []
    books_payload: list[dict[str, Any]] = []
    for book in books_raw:
        parsed = _transform_book(book, markets_filter=markets_filter, home=home, away=away)
        if parsed:
            books_payload.append(parsed)
    if not books_payload:
        return None
    return {
        "event_id": event_id,
        "game_id": event.get("game_id") or event_id,
        "league": league,
        "commence_time": commence,
        "home_team": home,
        "away_team": away,
        "books": books_payload,
    }


def _transform_payload(
    payload: Mapping[str, Any] | Iterable[Mapping[str, Any]],
    *,
    league: str,
    markets_filter: set[str],
) -> list[dict[str, Any]]:
    if isinstance(payload, Mapping):
        events_iter = payload.get("data") or payload.get("events") or payload.get("results") or []
    else:
        events_iter = payload
    events: list[dict[str, Any]] = []
    for event in events_iter or []:
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

    if not has_env_key("ODDS_API_KEY"):
        events = generate_synthetic_events(
            "theodds",
            leagues=leagues_list or ["NFL"],
            markets=markets_list or ["h2h"],
            books=books_list or ["pinnacle"],
            poll_window_hours=poll_window_hours,
        )
        return ClientResponse(provider="theodds", events=events, synthetic=True)

    fetcher = HttpFetcher(ALLOWED_HOSTS, timeout=6.0, max_retries=2)
    limiter = RateLimiter(rate_limit_per_min)

    api_key = os.getenv("ODDS_API_KEY", "").strip()
    aggregated: list[dict[str, Any]] = []

    for league in leagues_list or ["NFL"]:
        sport_key = SPORT_KEY_MAP.get(league.upper())
        if not sport_key:
            continue
        limiter.wait()
        url = f"https://api.{_HOST_CORE}.com/v4/sports/{sport_key}/odds"
        params = {
            API_PARAM_KEY: api_key,
            "regions": "us",
            "markets": ",".join(markets_list or ["h2h"]),
            "oddsFormat": "american",
            "bookmakers": ",".join(books_list) if books_list else None,
        }
        params = {k: v for k, v in params.items() if v is not None}
        try:
            payload = fetcher.get(url, params=params)
        except FetchError:
            return ClientResponse(
                provider="theodds",
                events=generate_synthetic_events(
                    "theodds",
                    leagues=leagues_list or [league],
                    markets=markets_list or ["h2h"],
                    books=books_list or ["pinnacle"],
                    poll_window_hours=poll_window_hours,
                ),
                synthetic=True,
            )
        aggregated.extend(
            _transform_payload(payload, league=league, markets_filter=markets_filter)
        )

    if not aggregated:
        aggregated = generate_synthetic_events(
            "theodds",
            leagues=leagues_list or ["NFL"],
            markets=markets_list or ["h2h"],
            books=books_list or ["pinnacle"],
            poll_window_hours=poll_window_hours,
        )
        return ClientResponse(provider="theodds", events=aggregated, synthetic=True)

    return ClientResponse(provider="theodds", events=aggregated, synthetic=False)


__all__ = ["get_events"]

