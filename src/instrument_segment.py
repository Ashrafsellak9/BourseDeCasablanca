"""
Segmentation des titres (Actions, OPCVM, Obligations, Autre) pour analyses de volume.

Priorité :
  1. Fichier optionnel ``data/ref/ticker_segment.csv`` (colonnes ``Ticker``, ``Segment``)
  2. Colonnes explicites dans les données : ``Segment``, ``Type_Valeur``, ``Famille_Valeur``
  3. Heuristiques sur ``Libelle`` / ``Ticker`` (mots-clés OPCVM, obligations, etc.)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SEG_ACTIONS = "Actions"
SEG_OPCVM = "OPCVM"
SEG_OBLIGATIONS = "Obligations"
SEG_AUTRE = "Autre"

ALL_SEGMENTS = (SEG_ACTIONS, SEG_OPCVM, SEG_OBLIGATIONS, SEG_AUTRE)

VOLUME_COL_ACTIONS = "Volume_MAD_Actions"
VOLUME_COL_OPCVM = "Volume_MAD_OPCVM"
VOLUME_COL_OBLIGATIONS = "Volume_MAD_Obligations"
VOLUME_COL_AUTRE = "Volume_MAD_Autre"

SEGMENT_VOLUME_COLUMNS = (
    VOLUME_COL_ACTIONS,
    VOLUME_COL_OPCVM,
    VOLUME_COL_OBLIGATIONS,
    VOLUME_COL_AUTRE,
)


def _normalize_segment_label(val: str) -> str | None:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    s = str(val).strip()
    u = s.upper()
    if u in ("ACTION", "ACTIONS", "VALEUR MOBILIERE", "VM"):
        return SEG_ACTIONS
    if u in ("OPCVM", "OPC", "SICAV", "FCP", "FONDS"):
        return SEG_OPCVM
    if u in ("OBLIGATION", "OBLIGATIONS", "OBLIG", "TCN", "EMTN"):
        return SEG_OBLIGATIONS
    if u in ("AUTRE", "OTHER", "DIVERS"):
        return SEG_AUTRE
    return None


def _heuristic_segment(lib: str, tick: str) -> str:
    L = (lib or "").upper()
    T = (tick or "").upper().strip()

    oblig_patterns = (
        "OBLIGATION",
        "OBLIGATIONS",
        " TCN",
        "TCN ",
        "EMTN",
        "TITRIS",
        "OAT ",
        " BTN",
        "BOBLIG",
        "OBLIG ",
    )
    for p in oblig_patterns:
        if p in L or p in T:
            return SEG_OBLIGATIONS

    opcvm_patterns = (
        "OPCVM",
        "SICAV",
        " FCP",
        "FCP ",
        " FONDS",
        "FONDS ",
        "OPC ",
        "COLLECTIF",
        "PLACEMENT COLLECTIF",
    )
    for p in opcvm_patterns:
        if p in L or p in T:
            return SEG_OPCVM

    return SEG_ACTIONS


def _coerce_segment_value(raw) -> str:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return SEG_ACTIONS
    s = str(raw).strip()
    if not s:
        return SEG_ACTIONS
    if s in ALL_SEGMENTS:
        return s
    n = _normalize_segment_label(s)
    return n if n else SEG_AUTRE


def load_ticker_segment_overrides(data_dir: Path | None = None) -> dict[str, str]:
    """Charge ``data/ref/ticker_segment.csv`` si présent : Ticker -> Segment."""
    base = data_dir or Path(__file__).resolve().parent.parent / "data"
    path = base / "ref" / "ticker_segment.csv"
    if not path.exists():
        return {}
    try:
        ov = pd.read_csv(path, dtype=str)
    except (OSError, pd.errors.EmptyDataError):
        return {}
    if "Ticker" not in ov.columns or "Segment" not in ov.columns:
        return {}
    ov = ov.dropna(subset=["Ticker"])
    ov["Ticker"] = ov["Ticker"].astype(str).str.strip().str.upper()
    ov["Segment"] = ov["Segment"].astype(str).str.strip()
    return {k: _coerce_segment_value(v) for k, v in zip(ov["Ticker"], ov["Segment"])}


def assign_instrument_segment(
    df: pd.DataFrame,
    data_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Ajoute la colonne ``Segment`` (valeurs : ``ALL_SEGMENTS``).
    """
    out = df.copy()
    overrides = load_ticker_segment_overrides(data_dir)

    for col in ("Segment", "Type_Valeur", "Famille_Valeur"):
        if col in out.columns:
            out["Segment"] = out[col].map(_coerce_segment_value)
            break
    else:
        lib = out["Libelle"].astype(str) if "Libelle" in out.columns else pd.Series("", index=out.index)
        tick = out["Ticker"].astype(str) if "Ticker" in out.columns else pd.Series("", index=out.index)
        out["Segment"] = [_heuristic_segment(l, t) for l, t in zip(lib, tick)]

    if overrides and "Ticker" in out.columns:
        t_up = out["Ticker"].astype(str).str.strip().str.upper()
        m = t_up.map(lambda k: overrides.get(k))
        hit = m.notna()
        out.loc[hit, "Segment"] = m[hit].values

    out.loc[~out["Segment"].isin(ALL_SEGMENTS), "Segment"] = SEG_AUTRE
    return out


def daily_volume_by_segment(df: pd.DataFrame, data_dir: Path | None = None) -> pd.DataFrame:
    """
    Agrège ``Volume_MAD`` par ``Jour`` et segment.

    Retourne ``Jour`` + ``Volume_MAD_Actions``, ``Volume_MAD_OPCVM``, ``Volume_MAD_Obligations``, ``Volume_MAD_Autre``.
    """
    if df.empty or "Jour" not in df.columns or "Volume_MAD" not in df.columns:
        return pd.DataFrame(columns=["Jour"] + list(SEGMENT_VOLUME_COLUMNS))

    d = assign_instrument_segment(df, data_dir)
    d["Jour"] = pd.to_datetime(d["Jour"])
    pivot = (
        d.groupby(["Jour", "Segment"], observed=True)["Volume_MAD"]
        .sum()
        .unstack(fill_value=0.0)
    )
    for s in ALL_SEGMENTS:
        if s not in pivot.columns:
            pivot[s] = 0.0
    pivot = pivot.reindex(columns=list(ALL_SEGMENTS), fill_value=0.0)
    pivot.columns = list(SEGMENT_VOLUME_COLUMNS)
    return pivot.reset_index()


def enrich_market_with_segment_volumes(
    df_market: pd.DataFrame,
    df_cours_or_instr: pd.DataFrame,
    data_dir: Path | None = None,
) -> pd.DataFrame:
    """Fusionne les volumes par segment sur le DataFrame marché (clé ``Jour``)."""
    if df_market.empty or df_cours_or_instr.empty:
        return df_market
    seg = daily_volume_by_segment(df_cours_or_instr, data_dir)
    if seg.empty:
        return df_market
    m = df_market.copy()
    m["Jour"] = pd.to_datetime(m["Jour"])
    out = m.merge(seg, on="Jour", how="left")
    for c in SEGMENT_VOLUME_COLUMNS:
        if c in out.columns:
            out[c] = out[c].fillna(0.0)
    if "Volume_MAD" in out.columns:
        out["Volume_Somme_Segments"] = out[list(SEGMENT_VOLUME_COLUMNS)].sum(axis=1)
        out["Volume_Segments_vs_Marche_pct"] = np.where(
            out["Volume_MAD"].replace(0, np.nan).notna() & (out["Volume_MAD"] != 0),
            out["Volume_Somme_Segments"] / out["Volume_MAD"].replace(0, np.nan) * 100.0,
            np.nan,
        )
    return out
