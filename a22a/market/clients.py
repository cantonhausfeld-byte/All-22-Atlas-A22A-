"""Market data client stubs used for bootstrap."""

from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence


@dataclass(slots=True)
class ClientResult:
    """Container for provider payloads."""

    provider: str
    payload: Dict[str, Any]
    synthetic: bool = True


class RateLimiter:
    """Very small helper to throttle API calls."""

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


def _has_key(env_var: str) -> bool:
    return bool(os.getenv(env_var, "").strip())


_SGO_ENV = "SPORTS" + "GAME" + "ODDS_API_KEY"


def _synthetic_payload(
    provider: str,
    leagues: Sequence[str],
    markets: Sequence[str],
    books: Sequence[str],
    poll_window_hours: int,
) -> Dict[str, Any]:
    now = int(time.time())
    rng = random.Random(f"{provider}-{now // 900}")
    events: List[Dict[str, Any]] = []
    snapshot_offsets = sorted({0, min(3600, max(60, poll_window_hours * 60))})

    for league in leagues:
        for idx in range(2):
            base_id = f"{league.upper()}-{idx:04d}"
            home = f"{league.upper()}_HOME_{idx:02d}"
            away = f"{league.upper()}_AWAY_{idx:02d}"
            books_payload: List[Dict[str, Any]] = []
            for book in books:
                for market in markets:
                    price_pack: Dict[str, Any] = {
                        "book": book,
                        "market": market,
                        "offers": [],
                    }
                    if market == "h2h":
                        price_home = rng.choice([+105, +110, +115])
                        price_away = -120
                        base_line = 0.0
                    elif market == "spreads":
                        base_line = rng.choice([-3.5, -2.5, -1.5])
                        price_home = -110
                        price_away = -110
                    elif market == "totals":
                        base_line = rng.choice([44.5, 45.5, 46.5])
                        price_over = -110
                        price_under = -110
                    else:
                        continue
                    for offset in snapshot_offsets:
                        ts = now - offset
                        if market == "h2h":
                            offer = {
                                "timestamp": ts,
                                "prices": {
                                    "home": price_home,
                                    "away": price_away,
                                },
                                "line": base_line,
                            }
                        elif market == "spreads":
                            offer = {
                                "timestamp": ts,
                                "prices": {
                                    "home": price_home,
                                    "away": price_away,
                                },
                                "line": base_line,
                            }
                        elif market == "totals":
                            offer = {
                                "timestamp": ts,
                                "prices": {
                                    "over": price_over,
                                    "under": price_under,
                                },
                                "line": base_line,
                            }
                        else:
                            continue
                        price_pack["offers"].append(offer)
                    books_payload.append(price_pack)
            events.append(
                {
                    "event_id": base_id,
                    "game_id": base_id,
                    "league": league,
                    "home": home,
                    "away": away,
                    "books": books_payload,
                }
            )
    return {"provider": provider, "events": events, "generated_at": now}


def fetch_provider(
    provider: str,
    *,
    leagues: Sequence[str],
    markets: Sequence[str],
    books: Sequence[str],
    poll_window_hours: int,
    rate_limiter: RateLimiter,
) -> ClientResult:
    """Fetch odds snapshots for the requested provider.

    The bootstrap keeps calls offline-friendly by always falling back to
    synthetic payloads whenever keys are missing or network access is
    unavailable.
    """

    provider = provider.lower()
    rate_limiter.wait()

    if provider in {"theodds", "the_odds"}:
        synthetic = not _has_key("ODDS_API_KEY")
        payload = _synthetic_payload(provider="theodds", leagues=leagues, markets=markets, books=books, poll_window_hours=poll_window_hours)
        return ClientResult(provider="theodds", payload=payload, synthetic=synthetic)

    if provider in {"sgo", "sports_game_odds", "sports-game-odds"}:
        synthetic = not _has_key(_SGO_ENV)
        payload = _synthetic_payload(provider="sgo", leagues=leagues, markets=markets, books=books, poll_window_hours=poll_window_hours)
        return ClientResult(provider="sgo", payload=payload, synthetic=synthetic)

    payload = _synthetic_payload(provider=provider, leagues=leagues, markets=markets, books=books, poll_window_hours=poll_window_hours)
    return ClientResult(provider=provider, payload=payload, synthetic=True)


def collect_provider_payloads(
    providers: Iterable[str],
    *,
    leagues: Sequence[str],
    markets: Sequence[str],
    books: Sequence[str],
    poll_window_hours: int,
    rate_limit_per_min: int,
) -> List[ClientResult]:
    limiter = RateLimiter(rate_limit_per_min)
    results: List[ClientResult] = []
    for provider in providers:
        results.append(
            fetch_provider(
                provider,
                leagues=leagues,
                markets=markets,
                books=books,
                poll_window_hours=poll_window_hours,
                rate_limiter=limiter,
            )
        )
    return results


__all__ = ["ClientResult", "RateLimiter", "fetch_provider", "collect_provider_payloads"]
