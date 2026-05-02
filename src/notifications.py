"""
Envoi de notifications (e-mail SMTP, Slack Incoming Webhook).

Configuration par variables d'environnement (aucune clé en dur).
"""

from __future__ import annotations

import json
import os
import smtplib
import ssl
from dataclasses import dataclass
from email.message import EmailMessage
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class NotificationSettings:
    """Paramètres lus une fois depuis l'environnement."""

    smtp_host: str | None
    smtp_port: int
    smtp_user: str | None
    smtp_password: str | None
    smtp_use_tls: bool
    email_from: str | None
    email_to: list[str]
    slack_webhook_url: str | None

    @classmethod
    def from_env(cls) -> NotificationSettings:
        to_raw = os.environ.get("ALERT_EMAIL_TO", "").strip()
        recipients = [x.strip() for x in to_raw.split(",") if x.strip()]
        use_tls = os.environ.get("SMTP_USE_TLS", "true").lower() in ("1", "true", "yes")
        port = int(os.environ.get("SMTP_PORT", "587") or "587")
        return cls(
            smtp_host=os.environ.get("SMTP_HOST", "").strip() or None,
            smtp_port=port,
            smtp_user=os.environ.get("SMTP_USER", "").strip() or None,
            smtp_password=os.environ.get("SMTP_PASSWORD", "").strip() or None,
            smtp_use_tls=use_tls,
            email_from=os.environ.get("ALERT_EMAIL_FROM", "").strip() or None,
            email_to=recipients,
            slack_webhook_url=os.environ.get("SLACK_WEBHOOK_URL", "").strip() or None,
        )

    def email_ready(self) -> bool:
        return bool(
            self.smtp_host
            and self.email_from
            and self.email_to
            and self.smtp_user
            and self.smtp_password
        )

    def slack_ready(self) -> bool:
        return bool(self.slack_webhook_url)


def send_email_smtp(
    settings: NotificationSettings,
    subject: str,
    body_text: str,
) -> None:
    """Envoie un e-mail texte brut UTF-8 (STARTTLS si ``smtp_use_tls``)."""
    if not settings.email_ready():
        raise ValueError("Configuration e-mail incomplète (SMTP_HOST, utilisateur, mot de passe, expéditeur, destinataires).")

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = settings.email_from
    msg["To"] = ", ".join(settings.email_to)
    msg.set_content(body_text, charset="utf-8")

    context = ssl.create_default_context()
    with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=60) as server:
        if settings.smtp_use_tls:
            server.starttls(context=context)
        server.login(settings.smtp_user, settings.smtp_password)
        server.send_message(msg)


def send_slack_webhook(settings: NotificationSettings, text: str) -> None:
    """POST JSON sur l'URL Incoming Webhook Slack."""
    if not settings.slack_ready():
        raise ValueError("SLACK_WEBHOOK_URL manquant.")

    payload: dict[str, Any] = {"text": text}
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = Request(
        settings.slack_webhook_url,
        data=data,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    try:
        with urlopen(req, timeout=30) as resp:
            _ = resp.read()
    except HTTPError as e:
        raise RuntimeError(f"Slack HTTP {e.code}: {e.reason}") from e
    except URLError as e:
        raise RuntimeError(f"Slack erreur réseau: {e.reason}") from e


def format_multipart_message(
    title: str,
    lines: list[str],
    max_body_chars: int = 12000,
) -> str:
    """Assemble un corps unique pour e-mail et Slack (tronqué si trop long)."""
    body = title + "\n\n" + "\n".join(lines)
    if len(body) <= max_body_chars:
        return body
    return body[: max_body_chars - 80] + "\n\n[… message tronqué — voir l’application Centre d’Alertes]"
