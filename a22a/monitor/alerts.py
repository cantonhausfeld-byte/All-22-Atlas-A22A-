"""Alert helpers for Phase 18 monitoring."""

from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict

__all__ = ["AlertPayload", "AlertsClient", "send_slack"]


@dataclass(slots=True)
class AlertPayload:
    """Structured alert payload used for Slack/email notifications."""

    title: str
    status: str
    body: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {"title": self.title, "status": self.status, "body": self.body}

    def as_json(self, *, compact: bool = False) -> str:
        if compact:
            return json.dumps(self.as_dict(), separators=(",", ":"), sort_keys=True)
        return json.dumps(self.as_dict(), indent=2, sort_keys=True)


def send_slack(payload: Dict[str, Any], *, webhook_env: str = "SLACK_WEBHOOK_URL") -> str:
    """Send a Slack webhook payload or fall back to a dry-run message."""

    webhook_url = os.getenv(webhook_env)
    if not webhook_url:
        message = f"[alerts] slack dry-run — missing env '{webhook_env}'"
        print(message)
        return message

    data = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    request = urllib.request.Request(
        webhook_url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:  # pragma: no cover - network call
            message = f"[alerts] slack {response.status}"
    except Exception as exc:  # pragma: no cover - network call
        message = f"[alerts] slack error: {exc}"
    print(message)
    return message


class AlertsClient:
    """Simple fan-out client that keeps integrations environment-only."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or {}
        self.slack_env = cfg.get("slack_webhook_env", "SLACK_WEBHOOK_URL")
        self.email_from = cfg.get("email_from", "")
        self.email_to = tuple(cfg.get("email_to", []))

    def send(self, channel: str, payload: AlertPayload) -> str:
        if channel == "slack":
            return send_slack(payload.as_dict(), webhook_env=self.slack_env)
        if channel == "email":
            return self._send_email(payload)
        message = f"[alerts] unsupported channel '{channel}'"
        print(message)
        return message

    def _send_email(self, payload: AlertPayload) -> str:
        if not self.email_from or not self.email_to:
            message = "[alerts] email dry-run — sender or recipients unset"
            print(message)
            return message

        message = (
            "[alerts] email dry-run — message not sent\n"
            f"from={self.email_from} to={list(self.email_to)}\n{payload.as_json(compact=False)}"
        )
        print(message)
        return message
