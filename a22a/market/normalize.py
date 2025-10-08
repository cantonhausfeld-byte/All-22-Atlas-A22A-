"""Normalization utilities for provider payloads."""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd

NORMALIZED_COLUMNS = [
    "provider",
    "game_id",
    "event_id",
    "league",
    "book",
    "market",
    "selection",
    "line",
    "price",
    "ts",
    "synthetic",
]


def normalize_payload(provider: str, payload: Dict[str, Any], *, synthetic: bool) -> pd.DataFrame:
    events = payload.get("events", []) if isinstance(payload, dict) else []
    rows: list[dict[str, Any]] = []
    generated_at = payload.get("generated_at") if isinstance(payload, dict) else None

    for event in events:
        event_id = str(event.get("event_id"))
        game_id = str(event.get("game_id", event_id))
        league = event.get("league")
        books = event.get("books") or []
        for book_entry in books:
            book = book_entry.get("book") or "unknown"
            market = (book_entry.get("market") or "").lower()
            offers = book_entry.get("offers") or []
            for offer in offers:
                ts_raw = offer.get("timestamp", generated_at)
                ts = pd.to_datetime(ts_raw, unit="s", utc=True, errors="coerce")
                line = offer.get("line", 0.0)
                prices = offer.get("prices") or {}
                for selection, price in prices.items():
                    rows.append(
                        {
                            "provider": provider,
                            "game_id": game_id,
                            "event_id": event_id,
                            "league": league,
                            "book": str(book).lower(),
                            "market": market,
                            "selection": str(selection).lower(),
                            "line": float(line) if line is not None else 0.0,
                            "price": int(price),
                            "ts": ts,
                            "synthetic": bool(synthetic),
                        }
                    )

    if not rows:
        return pd.DataFrame(columns=NORMALIZED_COLUMNS)

    df = pd.DataFrame(rows, columns=NORMALIZED_COLUMNS)
    df = df.sort_values(["game_id", "book", "market", "selection", "ts"]).reset_index(drop=True)
    return df


__all__ = ["NORMALIZED_COLUMNS", "normalize_payload"]
