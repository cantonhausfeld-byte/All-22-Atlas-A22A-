"""Shared helpers for market data clients."""

from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, MutableMapping, Sequence
from urllib.parse import urlparse

import requests


class FetchError(RuntimeError):
    """Raised when a remote fetch cannot be satisfied."""


@dataclass(slots=True)
class ClientResponse:
    """Container for provider payloads."""

    provider: str
    events: list[dict[str, Any]]
    synthetic: bool


class RateLimiter:
    """Minimal client-side rate limiter (per minute)."""

    def __init__(self, per_minute: int) -> None:
        self._per_minute = max(int(per_minute or 0), 0)
        self._last_hit: float | None = None
        self._minimum_spacing = 60.0 / self._per_minute if self._per_minute else 0.0

    def wait(self) -> None:
        if not self._per_minute:
            return
        now = time.monotonic()
        if self._last_hit is None:
            self._last_hit = now
            return
        elapsed = now - self._last_hit
        if elapsed < self._minimum_spacing:
            time.sleep(self._minimum_spacing - elapsed)
        self._last_hit = time.monotonic()


class HttpFetcher:
    """HTTP GET helper enforcing host allowlists and retries."""

    def __init__(
        self,
        allowed_hosts: Iterable[str],
        *,
        timeout: float = 6.0,
        max_retries: int = 2,
        backoff: float = 0.6,
    ) -> None:
        self._allowed_hosts = {h.lower() for h in allowed_hosts}
        self._timeout = float(timeout)
        self._max_retries = max(int(max_retries), 0)
        self._backoff = max(float(backoff), 0.0)

    def _validate_url(self, url: str) -> None:
        host = urlparse(url).hostname or ""
        if host.lower() not in self._allowed_hosts:
            raise FetchError(f"host `{host}` not allowed")

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> MutableMapping[str, Any] | list[dict[str, Any]]:
        self._validate_url(url)
        attempt = 0
        last_error: Exception | None = None
        while attempt <= self._max_retries:
            try:
                response = requests.get(url, params=params, headers=headers, timeout=self._timeout)
                if response.status_code == 200:
                    return response.json()
                if 400 <= response.status_code < 500:
                    raise FetchError(f"{url} -> {response.status_code}")
            except requests.RequestException as exc:  # pragma: no cover - defensive
                last_error = exc
            except ValueError as exc:  # json decode error
                last_error = exc
            else:
                # Unsuccessful status code outside 2xx/4xx -> retry.
                last_error = FetchError(f"{url} -> {response.status_code}")
            attempt += 1
            if attempt <= self._max_retries and self._backoff:
                time.sleep(self._backoff * attempt)
        raise FetchError(str(last_error) if last_error else "request failed")


def _normalize_sequence(items: Sequence[str] | None) -> list[str]:
    if not items:
        return []
    return [str(item).strip() for item in items if str(item).strip()]


def _synthetic_timestamp_bucket(now: float | None = None) -> int:
    ref = int(now if now is not None else time.time())
    return (ref // 900) * 900  # lock to 15-minute windows for determinism


def generate_synthetic_events(
    provider: str,
    *,
    leagues: Sequence[str],
    markets: Sequence[str],
    books: Sequence[str],
    poll_window_hours: int,
) -> list[dict[str, Any]]:
    leagues = _normalize_sequence(leagues)
    markets = _normalize_sequence(markets)
    books = _normalize_sequence(books)
    if not leagues:
        leagues = ["NFL"]
    if not markets:
        markets = ["h2h"]
    if not books:
        books = ["pinnacle"]

    base_ts = _synthetic_timestamp_bucket()
    open_offset = min(max(1800, poll_window_hours * 3600 // 2 or 1800), 6 * 3600)
    snapshot_offsets = sorted({0, open_offset})

    events: list[dict[str, Any]] = []
    rng = random.Random(
        f"{provider}|{base_ts}|{','.join(leagues)}|{','.join(markets)}|{','.join(books)}|{poll_window_hours}"
    )

    for league in leagues:
        for idx in range(3):
            event_id = f"{league.upper()}-{idx:04d}"
            home = f"{league.upper()}_HOME_{idx:02d}"
            away = f"{league.upper()}_AWAY_{idx:02d}"
            commence = base_ts + (idx % 3) * 3600
            books_payload: list[dict[str, Any]] = []
            for book in books:
                markets_payload: list[dict[str, Any]] = []
                for market in markets:
                    outcomes: list[dict[str, Any]] = []
                    base_line = 0.0
                    if market == "h2h":
                        price_pairs = [(-135, +120), (-125, +115), (-140, +130)]
                        base_home, base_away = rng.choice(price_pairs)
                        line_shifts = [0, rng.choice([-5, 5])]
                    elif market == "spreads":
                        base_line = rng.choice([-3.5, -2.5, -1.5, 1.5])
                        base_home = base_away = -110
                        line_shifts = [0.0, rng.choice([-0.5, 0.0, 0.5])]
                    elif market == "totals":
                        base_line = rng.choice([43.5, 44.5, 45.5, 46.5])
                        base_home = base_away = -110
                        line_shifts = [0.0, rng.choice([-1.0, -0.5, 0.5, 1.0])]
                    else:
                        continue

                    max_offset = max(snapshot_offsets)
                    for offset in snapshot_offsets:
                        stage = "open" if offset == max_offset else "close"
                        ts = commence - offset
                        if market == "h2h":
                            shift = line_shifts[0] if stage == "open" else line_shifts[1]
                            home_price = int(base_home + shift)
                            away_price = int(base_away - shift)
                            outcomes.extend(
                                [
                                    {
                                        "selection": "home",
                                        "price": home_price,
                                        "line": 0.0,
                                        "timestamp": ts,
                                    },
                                    {
                                        "selection": "away",
                                        "price": away_price,
                                        "line": 0.0,
                                        "timestamp": ts,
                                    },
                                ]
                            )
                        elif market == "spreads":
                            shift = line_shifts[0] if stage == "open" else line_shifts[1]
                            home_line = float(base_line + shift)
                            away_line = float(-(base_line + shift))
                            home_price = int(base_home + rng.choice([-5, 0, 5]))
                            away_price = int(base_away + rng.choice([-5, 0, 5]))
                            outcomes.extend(
                                [
                                    {
                                        "selection": "home",
                                        "price": home_price,
                                        "line": home_line,
                                        "timestamp": ts,
                                    },
                                    {
                                        "selection": "away",
                                        "price": away_price,
                                        "line": away_line,
                                        "timestamp": ts,
                                    },
                                ]
                            )
                        elif market == "totals":
                            shift = line_shifts[0] if stage == "open" else line_shifts[1]
                            over_line = float(base_line + shift)
                            under_line = float(base_line + shift)
                            over_price = int(base_home + rng.choice([-5, 0, 5]))
                            under_price = int(base_away + rng.choice([-5, 0, 5]))
                            outcomes.extend(
                                [
                                    {
                                        "selection": "over",
                                        "price": over_price,
                                        "line": over_line,
                                        "timestamp": ts,
                                    },
                                    {
                                        "selection": "under",
                                        "price": under_price,
                                        "line": under_line,
                                        "timestamp": ts,
                                    },
                                ]
                            )
                    if outcomes:
                        markets_payload.append(
                            {
                                "market": market,
                                "outcomes": outcomes,
                            }
                        )
                if markets_payload:
                    books_payload.append(
                        {
                            "book": book,
                            "markets": markets_payload,
                        }
                    )
            events.append(
                {
                    "event_id": event_id,
                    "game_id": event_id,
                    "league": league,
                    "commence_time": commence,
                    "home_team": home,
                    "away_team": away,
                    "books": books_payload,
                }
            )
    return events


def has_env_key(env_var: str) -> bool:
    return bool(os.getenv(env_var, "").strip())


__all__ = [
    "ClientResponse",
    "FetchError",
    "RateLimiter",
    "HttpFetcher",
    "generate_synthetic_events",
    "has_env_key",
]

