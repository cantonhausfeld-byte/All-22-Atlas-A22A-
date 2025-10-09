"""Alerting adapters for the monitoring subsystem."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(slots=True)
class AlertChannelConfig:
    """Minimal configuration required to emit alerts."""

    slack_webhook_env: str | None = None
    email_from: str | None = None
    email_to: Iterable[str] | None = None


class AlertsClient:
    """Fan-out router for monitoring alerts.

    The bootstrap implementation prints rich messages instead of performing
    network calls. Secrets remain environment-only and we log when falling back
    to a dry-run so CI visibility remains high.
    """

    def __init__(self, config: dict[str, Any] | None, channels: Iterable[str] | None):
        cfg = config or {}
        self.channels = tuple(channels or ())
        self.channel_config = AlertChannelConfig(
            slack_webhook_env=cfg.get("slack_webhook_env"),
            email_from=cfg.get("email_from"),
            email_to=tuple(cfg.get("email_to", [])),
        )

    def notify(self, payload: dict[str, Any]) -> None:
        message = json.dumps(payload, indent=2, sort_keys=True)
        for channel in self.channels:
            if channel == "slack":
                self._emit_slack(message)
            elif channel == "email":
                self._emit_email(message)
            else:
                print(f"[alerts] unsupported channel '{channel}' — skipping")

    # Internal helpers -----------------------------------------------------------------

    def _emit_slack(self, message: str) -> None:
        env_name = self.channel_config.slack_webhook_env
        webhook = os.getenv(env_name or "", "") if env_name else ""
        if webhook:
            print(
                f"[alerts] slack webhook present in env '{env_name}' — bootstrap mode, message not sent"
            )
        else:
            env_hint = env_name or "SLACK_WEBHOOK_URL"
            print(f"[alerts] slack dry-run — missing env '{env_hint}'\n{message}")

    def _emit_email(self, message: str) -> None:
        sender = self.channel_config.email_from or ""
        recipients = list(self.channel_config.email_to or ())
        if sender and recipients:
            print(
                "[alerts] email config present — bootstrap mode, message not sent",
                f"from={sender}",
                f"to={recipients}",
            )
        else:
            print(
                "[alerts] email dry-run — missing sender/recipients",
                f"from={sender or '<unset>'}",
                f"to={recipients}",
                message,
            )


__all__ = ["AlertsClient", "AlertChannelConfig"]
