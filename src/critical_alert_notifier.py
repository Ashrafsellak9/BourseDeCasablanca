"""
Notifications pour alertes **Critique** (anomalies statistiques + scores instruments).

- Cible la **dernière séance** présente dans les données (``max(Jour)``).
- Déduplication persistante via fichier JSON (évite renvois à chaque chargement Streamlit).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.notifications import (
    NotificationSettings,
    format_multipart_message,
    send_email_smtp,
    send_slack_webhook,
)

CRITIQUE = "Critique"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_STATE_PATH = PROJECT_ROOT / "data" / "runtime" / "critical_alerts_sent.json"
MAX_STORED_FINGERPRINTS = 8000
MAX_ROWS_IN_MESSAGE = 25


def _norm_jour(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s).dt.normalize()


def fingerprint_anomaly_row(row: pd.Series) -> str:
    j = pd.to_datetime(row["Jour"]).strftime("%Y-%m-%d")
    parts = [
        "anom",
        j,
        str(row.get("Ticker", "")),
        str(row.get("Niveau", "")),
        str(row.get("Indicateur", "")),
        str(row.get("Description", ""))[:400],
    ]
    v = row.get("Valeur")
    if pd.notna(v) and v is not None:
        try:
            parts.append(f"{float(v):.8g}")
        except (TypeError, ValueError):
            parts.append(str(v))
    raw = "|".join(parts).encode("utf-8", errors="replace")
    return "a:" + hashlib.sha256(raw).hexdigest()[:40]


def fingerprint_alert_score_row(row: pd.Series) -> str:
    j = pd.to_datetime(row["Jour"]).strftime("%Y-%m-%d")
    parts = [
        "score",
        j,
        str(row.get("Ticker", "")),
        str(row.get("Libelle", ""))[:120],
        f"{float(row.get('Score_Alerte', 0) or 0):.4f}",
    ]
    raw = "|".join(parts).encode("utf-8", errors="replace")
    return "s:" + hashlib.sha256(raw).hexdigest()[:40]


def _load_state(path: Path) -> set[str]:
    if not path.exists():
        return set()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        fps = data.get("fingerprints", [])
        if isinstance(fps, list):
            return {str(x) for x in fps}
    except (json.JSONDecodeError, OSError):
        pass
    return set()


def _save_state(path: Path, fingerprints: set[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lst = sorted(fingerprints)
    if len(lst) > MAX_STORED_FINGERPRINTS:
        lst = lst[-MAX_STORED_FINGERPRINTS:]
    path.write_text(
        json.dumps({"fingerprints": lst}, ensure_ascii=False, indent=0),
        encoding="utf-8",
    )


def collect_critical_rows_last_session(
    df_anom: pd.DataFrame,
    df_alerts: pd.DataFrame | None = None,
) -> tuple[pd.Timestamp, pd.DataFrame, pd.DataFrame]:
    """
    Retourne (jour_de_référence, anomalies_critiques, alertes_scores_critiques)
    pour la dernière date présente dans ``df_anom`` (normalisée à minuit).
    """
    if df_anom is None or df_anom.empty or "Jour" not in df_anom.columns:
        return pd.NaT, pd.DataFrame(), pd.DataFrame()

    df_a = df_anom.copy()
    df_a["Jour"] = pd.to_datetime(df_a["Jour"])
    ref = df_a["Jour"].max()
    if pd.isna(ref):
        return pd.NaT, pd.DataFrame(), pd.DataFrame()
    ref_norm = pd.Timestamp(ref).normalize()

    mask_date = _norm_jour(df_a["Jour"]) == ref_norm
    mask_sev = df_a["Severite_Finale"].astype(str) == CRITIQUE
    crit_a = df_a[mask_date & mask_sev].copy()

    crit_s = pd.DataFrame()
    if df_alerts is not None and not df_alerts.empty and "Jour" in df_alerts.columns:
        df_s = df_alerts.copy()
        df_s["Jour"] = pd.to_datetime(df_s["Jour"])
        m_d = _norm_jour(df_s["Jour"]) == ref_norm
        if "Severite" in df_s.columns:
            m_s = df_s["Severite"].astype(str) == CRITIQUE
            crit_s = df_s[m_d & m_s].copy()

    return ref_norm, crit_a, crit_s


def build_notification_lines(
    crit_a: pd.DataFrame,
    crit_s: pd.DataFrame,
) -> list[tuple[str, str]]:
    """Liste de (fingerprint, ligne_texte) pour les lignes à notifier."""
    out: list[tuple[str, str]] = []

    for _, row in crit_a.iterrows():
        fp = fingerprint_anomaly_row(row)
        lib = row.get("Libelle", "") or ""
        desc = (row.get("Description", "") or "")[:200]
        line = (
            f"• [{row.get('Niveau','')}] {row.get('Ticker','')} {lib} — {row.get('Indicateur','')}\n"
            f"  Sévérité: Critique | Valeur: {row.get('Valeur','')} | {desc}"
        )
        out.append((fp, line))

    for _, row in crit_s.iterrows():
        fp = fingerprint_alert_score_row(row)
        line = (
            f"• [Score alerte] {row.get('Ticker','')} {row.get('Libelle','')}\n"
            f"  Score: {float(row.get('Score_Alerte', 0) or 0):.1f}/100"
        )
        out.append((fp, line))

    return out


def notify_new_critical_alerts(
    df_anom: pd.DataFrame,
    df_alerts: pd.DataFrame | None,
    settings: NotificationSettings | None = None,
    state_path: Path | None = None,
    subject_prefix: str = "[BVC] Alertes critiques",
) -> dict[str, Any]:
    """
    Envoie e-mail et/ou Slack pour les alertes critiques de la **dernière séance**
    dont l'empreinte n'a pas encore été enregistrée.

    Retourne un dict ``ok``, ``skipped``, ``message``, ``errors``, etc.
    """
    settings = settings or NotificationSettings.from_env()
    state_path = state_path or DEFAULT_STATE_PATH

    result: dict[str, Any] = {
        "ok": False,
        "skipped": False,
        "reason": "",
        "ref_jour": None,
        "new_count": 0,
        "email_sent": False,
        "slack_sent": False,
        "errors": [],
    }

    if not settings.email_ready() and not settings.slack_ready():
        result["reason"] = "Aucune chaîne configurée (e-mail et/ou SLACK_WEBHOOK_URL)."
        result["no_channels"] = True
        return result

    ref, crit_a, crit_s = collect_critical_rows_last_session(df_anom, df_alerts)
    if pd.isna(ref):
        result["skipped"] = True
        result["reason"] = "Pas de données d'anomalies."
        return result

    result["ref_jour"] = ref.strftime("%Y-%m-%d")
    pairs = build_notification_lines(crit_a, crit_s)
    if not pairs:
        result["skipped"] = True
        result["reason"] = f"Aucune alerte critique le {result['ref_jour']}."
        return result

    known = _load_state(state_path)
    new_pairs = [(fp, line) for fp, line in pairs if fp not in known]
    if not new_pairs:
        result["skipped"] = True
        result["reason"] = "Toutes les alertes critiques de cette séance ont déjà été notifiées."
        return result

    new_pairs = new_pairs[:500]
    title = f"{subject_prefix} — {result['ref_jour']} ({len(new_pairs)} nouvelle(s) entrée(s))"
    lines = [line for _, line in new_pairs[:MAX_ROWS_IN_MESSAGE]]
    if len(new_pairs) > MAX_ROWS_IN_MESSAGE:
        lines.append(f"\n… et {len(new_pairs) - MAX_ROWS_IN_MESSAGE} autre(s) ligne(s).")
    body = format_multipart_message(title, lines)

    errors: list[str] = []
    email_ok = True
    slack_ok = True

    if settings.email_ready():
        try:
            send_email_smtp(settings, subject=title, body_text=body)
            result["email_sent"] = True
        except Exception as e:
            email_ok = False
            errors.append(f"E-mail: {e}")

    if settings.slack_ready():
        try:
            send_slack_webhook(settings, body)
            result["slack_sent"] = True
        except Exception as e:
            slack_ok = False
            errors.append(f"Slack: {e}")

    result["errors"] = errors
    need_email = settings.email_ready()
    need_slack = settings.slack_ready()
    all_ok = (not need_email or email_ok) and (not need_slack or slack_ok)

    if all_ok:
        new_fps = {fp for fp, _ in new_pairs}
        updated = known | new_fps
        try:
            _save_state(state_path, updated)
        except OSError as e:
            errors.append(f"État persistant: {e}")
            result["errors"] = errors
            return result
        result["ok"] = True
        result["new_count"] = len(new_pairs)
    else:
        result["reason"] = "Envoi partiel ou en échec — empreintes non enregistrées (nouvelle tentative possible)."

    return result


def notification_status_summary(settings: NotificationSettings | None = None) -> dict[str, Any]:
    settings = settings or NotificationSettings.from_env()
    return {
        "email_configured": settings.email_ready(),
        "slack_configured": settings.slack_ready(),
        "state_path": str(DEFAULT_STATE_PATH),
    }
