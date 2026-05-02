#!/usr/bin/env python3
"""
Tâche planifiée (cron / Task Scheduler) : envoie les nouvelles alertes critiques
(e-mail + Slack) pour la dernière séance, sans ouvrir Streamlit.

Variables d'environnement : voir ``src/notifications.NotificationSettings.from_env``.
Optionnel : ``BVC_PARQUET_DIR`` pour surcharger le dossier des Parquet (défaut : ./data).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.critical_alert_notifier import notify_new_critical_alerts
from src.notifications import NotificationSettings


def main() -> int:
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
    except ImportError:
        pass

    data_dir = Path(os.environ.get("BVC_PARQUET_DIR", str(ROOT / "data"))).resolve()
    anom_path = data_dir / "all_anomalies.parquet"
    alerts_path = data_dir / "alert_scores.parquet"

    if not anom_path.exists():
        print(json.dumps({"ok": False, "error": f"Fichier manquant: {anom_path}"}, ensure_ascii=False))
        return 1

    df_anom = pd.read_parquet(anom_path)
    df_alerts = pd.read_parquet(alerts_path) if alerts_path.exists() else None

    settings = NotificationSettings.from_env()
    rep = notify_new_critical_alerts(df_anom, df_alerts, settings=settings)
    print(json.dumps(rep, ensure_ascii=False, default=str))
    if rep.get("ok") or rep.get("skipped") or rep.get("no_channels"):
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
