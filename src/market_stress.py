"""
Market Stress Score — indicateur composite de stress de marché.

Combine trois dimensions (sur l'historique disponible, mis à jour à chaque
nouvelle séance / rechargement des données — « temps réel » au sens
opérationnel du dashboard) :

  1. Volatilité MASI (MASI_Vol_20j) — stress croît avec la volatilité ;
  2. Breadth — stress croît quand la participation haussière faiblit ;
  3. Déséquilibre agrégé des flux (|OIR| moyen par séance sur les titres).

Chaque composante est normalisée sur [0, 100] par rang percentile
historique (sauf breadth : 100 - Breadth_pct).
Le score final est la moyenne des composantes disponibles (poids égaux).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def enrich_market_stress_score(
    df_market: pd.DataFrame,
    df_orderflow: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Enrichit le DataFrame marché avec les colonnes de stress.

    Colonnes ajoutées
    -----------------
    Stress_Vol          : percentile rank de MASI_Vol_20j × 100
    Stress_Breadth      : 100 - Breadth_pct (planché/plafonné)
    OIR_Marche_MeanAbs  : moyenne des |OIR| par titre pour la séance
    Stress_OIR          : percentile rank de OIR_Marche_MeanAbs × 100
    Market_Stress_Score : moyenne nan-aware des trois (0–100)
    Market_Stress_Regime: libellé qualitatif (Calme / Faible / Modéré / Élevé)
    """
    out = df_market.copy()
    if out.empty:
        return out

    out["Jour"] = pd.to_datetime(out["Jour"])
    out = out.sort_values("Jour").reset_index(drop=True)

    # --- 1. Volatilité (niveau relatif à l'historique) ---
    if "MASI_Vol_20j" in out.columns:
        v = out["MASI_Vol_20j"]
        out["Stress_Vol"] = (v.rank(pct=True) * 100).clip(0, 100)
    else:
        out["Stress_Vol"] = np.nan

    # --- 2. Breadth (faible participation haussière = stress) ---
    if "Breadth_pct" in out.columns:
        out["Stress_Breadth"] = (100.0 - out["Breadth_pct"]).clip(0, 100)
    else:
        out["Stress_Breadth"] = np.nan

    # --- 3. OIR agrégé séance (intensité du déséquilibre moyen) ---
    if (
        df_orderflow is not None
        and not df_orderflow.empty
        and "OIR" in df_orderflow.columns
        and "Jour" in df_orderflow.columns
    ):
        of = df_orderflow.copy()
        of["Jour"] = pd.to_datetime(of["Jour"])
        oir_day = of.groupby("Jour", as_index=False).agg(
            OIR_Marche_MeanAbs=(
                "OIR",
                lambda s: float(np.nanmean(np.abs(pd.to_numeric(s, errors="coerce")))),
            )
        )
        out = out.merge(oir_day, on="Jour", how="left")
        oir_ma = out["OIR_Marche_MeanAbs"]
        out["Stress_OIR"] = (oir_ma.rank(pct=True) * 100).clip(0, 100)
    else:
        out["OIR_Marche_MeanAbs"] = np.nan
        out["Stress_OIR"] = np.nan

    # --- Score composite (poids égaux sur les composantes observées) ---
    comp = out[["Stress_Vol", "Stress_Breadth", "Stress_OIR"]]
    out["Market_Stress_Score"] = np.nanmean(comp.to_numpy(dtype=float), axis=1)

    def _regime(x: float) -> str:
        if pd.isna(x):
            return "—"
        if x >= 75:
            return "Élevé"
        if x >= 50:
            return "Modéré"
        if x >= 25:
            return "Faible"
        return "Calme"

    out["Market_Stress_Regime"] = out["Market_Stress_Score"].map(_regime)
    return out


def stress_summary_latest(df_enriched: pd.DataFrame) -> dict:
    """Dernière ligne utile pour KPI / jauge (dernière séance du DataFrame fourni)."""
    if df_enriched.empty or "Market_Stress_Score" not in df_enriched.columns:
        return {}
    last = df_enriched.iloc[-1]
    sc = last.get("Market_Stress_Score")
    return {
        "Jour": last.get("Jour"),
        "Market_Stress_Score": float(sc) if pd.notna(sc) else None,
        "Market_Stress_Regime": str(last.get("Market_Stress_Regime", "—")),
        "Stress_Vol": last.get("Stress_Vol"),
        "Stress_Breadth": last.get("Stress_Breadth"),
        "Stress_OIR": last.get("Stress_OIR"),
        "OIR_Marche_MeanAbs": last.get("OIR_Marche_MeanAbs"),
    }
