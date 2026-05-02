"""
Couche de cache centralisée pour l'application Streamlit.
Charge les données une seule fois et les met en cache avec st.cache_data.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(_ROOT / ".env")
except ImportError:
    pass

import pandas as pd
import numpy as np
import streamlit as st
from src.data_loader import load_excel, save_to_cache, validate_format, get_summary
from src.indicators import (
    compute_market_indicators,
    compute_instrument_indicators,
    compute_orderflow_indicators,
    compute_alert_scores,
    compute_advance_decline_line,
    daily_top5_volume_concentration,
)
from src.anomaly_detector import (
    detect_market_anomalies,
    detect_instrument_anomalies,
    detect_orderflow_anomalies,
    consolidate_all_anomalies,
    compute_seuils_table,
    summarize_anomalies,
)
from src.market_stress import enrich_market_stress_score
from src.instrument_segment import (
    SEGMENT_VOLUME_COLUMNS,
    enrich_market_with_segment_volumes,
)

DATA_DIR = Path(__file__).parent.parent.parent / "data"


@st.cache_data(show_spinner="Chargement des données 2025...")
def load_base_data():
    """Charge les DataFrames depuis Parquet et enrichit le marché (Market Stress Score)."""
    market = pd.read_parquet(DATA_DIR / "market_indicators.parquet")
    orderflow = pd.read_parquet(DATA_DIR / "orderflow_indicators.parquet")
    instrument = pd.read_parquet(DATA_DIR / "instrument_indicators.parquet")
    if not market.empty:
        market = enrich_market_stress_score(
            market, orderflow if not orderflow.empty else None
        )
    if (
        not market.empty
        and not instrument.empty
        and "Volume_MAD" in instrument.columns
        and not all(c in market.columns for c in SEGMENT_VOLUME_COLUMNS)
    ):
        market = enrich_market_with_segment_volumes(market, instrument, DATA_DIR)
    if (
        not market.empty
        and not instrument.empty
        and "Volume_MAD" in instrument.columns
        and "Volume_Top5_Pct" not in market.columns
    ):
        market = market.merge(
            daily_top5_volume_concentration(instrument),
            on="Jour",
            how="left",
        )
    if (
        not market.empty
        and not instrument.empty
        and "Rendement_pct" in instrument.columns
        and "AD_Line" not in market.columns
    ):
        _ad = compute_advance_decline_line(instrument[["Jour", "Rendement_pct"]])
        market = market.merge(_ad, on="Jour", how="left")
    return {
        "market":     market,
        "instrument": instrument,
        "orderflow":  orderflow,
        "alerts":     pd.read_parquet(DATA_DIR / "alert_scores.parquet"),
        "anomalies":  pd.read_parquet(DATA_DIR / "all_anomalies.parquet"),
        "seuils":     pd.read_csv(DATA_DIR / "seuils_phase3.csv"),
    }


@st.cache_data(show_spinner="Traitement du fichier uploadé...")
def process_uploaded_file(file_bytes: bytes, filename: str) -> dict:
    """
    Pipeline complet sur un nouveau fichier Excel uploadé.
    1. Chargement & nettoyage
    2. Calcul des indicateurs
    3. Détection statistique
    4. Retourne tous les résultats

    Retourne un dict avec les mêmes clés que load_base_data().
    """
    import io
    file_obj = io.BytesIO(file_bytes)

    # 1. Chargement
    raw = load_excel(file_obj)

    # 2. Validation du format
    errors = validate_format(raw)
    critical_errors = {k: v for k, v in errors.items() if v != ['FEUILLE MANQUANTE']}
    if critical_errors:
        return {'error': f"Format invalide : {critical_errors}"}

    df_ind   = raw.get('indicateurs', pd.DataFrame())
    df_idx   = raw.get('indices', pd.DataFrame())
    df_cours = raw.get('cours', pd.DataFrame())
    df_intra = raw.get('intraday', pd.DataFrame())

    result = {}

    # 3. Indicateurs marché (nécessite df_cours pour breadth)
    if not df_ind.empty and not df_idx.empty:
        df_cours_temp = df_cours.copy()
        if not df_cours_temp.empty:
            df_cours_temp = df_cours_temp.sort_values(['Ticker','Jour'])
            df_cours_temp['Rendement_pct'] = (
                df_cours_temp.groupby('Ticker')['Cours_Cloture'].pct_change() * 100
            )
        result['market'] = compute_market_indicators(df_ind, df_idx, df_cours_temp)
    else:
        result['market'] = pd.DataFrame()

    # 4. Indicateurs instruments
    if not df_cours.empty:
        result['instrument'] = compute_instrument_indicators(df_cours)
    else:
        result['instrument'] = pd.DataFrame()

    # 5. Order flow
    if not df_intra.empty and not df_cours.empty:
        result['orderflow'] = compute_orderflow_indicators(df_intra, df_cours)
    else:
        result['orderflow'] = pd.DataFrame()

    # 5b. Market Stress (volatilité + breadth + OIR agrégé)
    if not result["market"].empty:
        result["market"] = enrich_market_stress_score(
            result["market"],
            result["orderflow"] if not result.get("orderflow", pd.DataFrame()).empty else None,
        )

    # 6. Scores alertes
    if not result['instrument'].empty:
        result['alerts'] = compute_alert_scores(
            result['instrument'], result.get('market'), result.get('orderflow')
        )
    else:
        result['alerts'] = pd.DataFrame()

    # 7. Détection anomalies
    mkt_anom  = detect_market_anomalies(result['market']) if not result['market'].empty else pd.DataFrame()
    ins_anom  = detect_instrument_anomalies(result['instrument']) if not result['instrument'].empty else pd.DataFrame()
    of_anom   = detect_orderflow_anomalies(result['orderflow']) if not result['orderflow'].empty else pd.DataFrame()
    result['anomalies'] = consolidate_all_anomalies(mkt_anom, ins_anom, of_anom)
    result['seuils']    = compute_seuils_table(result['instrument'], result['market'], result.get('orderflow'))
    result['summary']   = get_summary(raw)
    result['filename']  = filename

    return result

