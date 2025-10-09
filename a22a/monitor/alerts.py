"""Bootstrap alert adapters used by the monitoring stub."""

from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class AlertPayload:
    title: str
    status: str
    body: dict[str, Any]

    def as_json(self) -> str:
        return json.dumps(
            {
                "title": self.title,
                "status": self.status,
                "body": self.body,
            },
            indent=2,
            sort_keys=True,
        )


class AlertsClient:
    """Small helper that performs dry-run alert fan-out when secrets are missing."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or {}
        self.slack_env = cfg.get("slack_webhook_env", "SLACK_WEBHOOK_URL")
        self.email_from = cfg.get("email_from", "")
        self.email_to = tuple(cfg.get("email_to", []))

    def send(self, channel: str, payload: AlertPayload) -> str:
        if channel == "slack":
            return self._send_slack(payload)
        if channel == "email":
            return self._send_email(payload)
        message = f"[alerts] unsupported channel '{channel}'"
        print(message)
        return message

    # Internal helpers -------------------------------------------------------------

    def _send_slack(self, payload: AlertPayload) -> str:
        webhook = os.getenv(self.slack_env)
        if not webhook:
            message = f"[alerts] slack dry-run — missing env '{self.slack_env}'"
            print(message)
            return message

        data = payload.as_json().encode("utf-8")
        request = urllib.request.Request(
            webhook,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=10) as response:  # pragma: no cover - network
                message = f"[alerts] slack {response.status}"
        except Exception as exc:  # pragma: no cover - network
            message = f"[alerts] slack error: {exc}"
        print(message)
        return message

    def _send_email(self, payload: AlertPayload) -> str:
        if not self.email_from or not self.email_to:
            message = "[alerts] email dry-run — sender or recipients unset"
            print(message)
            return message

        message = (
            "[alerts] email dry-run — message not sent\\n"
            f"from={self.email_from} to={list(self.email_to)}\\n{payload.as_json()}"
        )
        print(message)
        return message


__all__ = ["AlertPayload", "AlertsClient"]
