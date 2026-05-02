"""
Événements macroéconomiques marquants superposables aux graphiques marché.

Source : ``data/ref/macro_events.csv`` (éditable sans toucher au code).
Colonnes attendues : ``Jour``, ``Titre``, ``Type`` (optionnel : ``Source``).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DEFAULT_COLUMNS = ("Jour", "Titre", "Type", "Source")


def load_macro_events(data_dir: Path | None = None) -> pd.DataFrame:
    """
    Charge le calendrier macro depuis ``data/ref/macro_events.csv``.

    Retourne un DataFrame vide (schéma minimal) si le fichier est absent ou vide.
    """
    base = data_dir or Path(__file__).resolve().parent.parent / "data"
    path = base / "ref" / "macro_events.csv"
    if not path.exists():
        return pd.DataFrame(columns=list(DEFAULT_COLUMNS))

    try:
        df = pd.read_csv(path, dtype=str, encoding="utf-8")
    except (OSError, pd.errors.EmptyDataError, UnicodeDecodeError):
        return pd.DataFrame(columns=list(DEFAULT_COLUMNS))

    if df.empty:
        return pd.DataFrame(columns=list(DEFAULT_COLUMNS))

    df.columns = [str(c).strip() for c in df.columns]
    if "Jour" not in df.columns or "Titre" not in df.columns:
        return pd.DataFrame(columns=list(DEFAULT_COLUMNS))

    df = df.dropna(subset=["Jour", "Titre"])
    df["Jour"] = pd.to_datetime(df["Jour"], errors="coerce", dayfirst=False)
    df = df.dropna(subset=["Jour"])
    df["Titre"] = df["Titre"].astype(str).str.strip()
    if "Type" not in df.columns:
        df["Type"] = ""
    else:
        df["Type"] = df["Type"].fillna("").astype(str).str.strip()
    if "Source" not in df.columns:
        df["Source"] = ""
    else:
        df["Source"] = df["Source"].fillna("").astype(str).str.strip()

    return df.sort_values("Jour").reset_index(drop=True)


def filter_macro_events_for_period(
    df: pd.DataFrame,
    d_start,
    d_end,
) -> pd.DataFrame:
    """Filtre les événements dont la date est dans ``[d_start, d_end]`` (inclus)."""
    if df is None or df.empty:
        return df.iloc[0:0].copy() if df is not None else pd.DataFrame(columns=list(DEFAULT_COLUMNS))

    out = df.copy()
    out["Jour"] = pd.to_datetime(out["Jour"]).dt.normalize()
    j0 = pd.Timestamp(d_start).normalize()
    j1 = pd.Timestamp(d_end).normalize()
    m = (out["Jour"] >= j0) & (out["Jour"] <= j1)
    return out.loc[m].sort_values("Jour").reset_index(drop=True)
