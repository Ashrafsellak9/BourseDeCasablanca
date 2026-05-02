"""
Module de détection statistique d'anomalies — Phase 3.

Méthodes implémentées
---------------------
1. Z-score (statique et rolling)        — détection par écarts-types
2. IQR (Interquartile Range)            — détection robuste aux outliers
3. Percentile absolu P95 / P99          — seuils extrêmes
4. CUSUM (Cumulative Sum)               — détection de changement de régime
5. Rolling Z-score                      — anomalie contextuelle (fenêtre glissante)
6. Bollinger Bands                      — seuils adaptatifs (moy ± k*σ rolling)

Pour chaque indicateur et chaque niveau (marché / instrument / flux d'ordres),
un tableau d'anomalies est produit avec :
  - date, ticker, valeur, seuil, méthode, sévérité, description
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Literal
import warnings
warnings.filterwarnings('ignore')

# Sévérité
NORMAL   = 'Normal'
FAIBLE   = 'Faible'
MODERE   = 'Modéré'
CRITIQUE = 'Critique'

# ═══════════════════════════════════════════════════════════════════════════
# 1. MÉTHODES DE DÉTECTION DE BASE
# ═══════════════════════════════════════════════════════════════════════════

def detect_zscore(
    series: pd.Series,
    threshold_warn: float = 2.0,
    threshold_crit: float = 3.0,
) -> pd.Series:
    """
    Z-score statique sur toute la série.
    Retourne la sévérité pour chaque point.
    """
    mu, sigma = series.mean(), series.std()
    if sigma == 0:
        return pd.Series(NORMAL, index=series.index)
    z = (series - mu).abs() / sigma
    return z.apply(lambda v: CRITIQUE if v >= threshold_crit
                              else MODERE if v >= threshold_warn
                              else NORMAL)


def detect_rolling_zscore(
    series: pd.Series,
    window: int = 20,
    threshold_warn: float = 2.0,
    threshold_crit: float = 3.0,
) -> pd.Series:
    """
    Z-score rolling : détecte les anomalies par rapport au contexte récent.
    Plus adapté aux séries non-stationnaires (prix, volumes en tendance).
    """
    roll_mean = series.rolling(window, min_periods=window // 2).mean()
    roll_std  = series.rolling(window, min_periods=window // 2).std()
    z = (series - roll_mean).abs() / roll_std.replace(0, np.nan)
    return z.apply(lambda v: CRITIQUE if pd.notna(v) and v >= threshold_crit
                              else MODERE if pd.notna(v) and v >= threshold_warn
                              else NORMAL)


def detect_percentile(
    series: pd.Series,
    p_low: float = 1.0,
    p_high: float = 99.0,
    p_warn_low: float = 5.0,
    p_warn_high: float = 95.0,
) -> pd.Series:
    """
    Détection par percentiles absolus sur la distribution historique complète.
    """
    vals = series.dropna()
    lo_c, hi_c = np.percentile(vals, p_low), np.percentile(vals, p_high)
    lo_w, hi_w = np.percentile(vals, p_warn_low), np.percentile(vals, p_warn_high)

    def _label(v):
        if pd.isna(v):
            return NORMAL
        if v <= lo_c or v >= hi_c:
            return CRITIQUE
        if v <= lo_w or v >= hi_w:
            return MODERE
        return NORMAL

    return series.apply(_label)


def detect_iqr(
    series: pd.Series,
    k_warn: float = 1.5,
    k_crit: float = 3.0,
) -> pd.Series:
    """
    Méthode IQR (Tukey) — robuste aux distributions non-gaussiennes.
    Seuil warn = Q1 - k*IQR / Q3 + k*IQR
    """
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lo_w, hi_w = q1 - k_warn * iqr, q3 + k_warn * iqr
    lo_c, hi_c = q1 - k_crit * iqr, q3 + k_crit * iqr

    def _label(v):
        if pd.isna(v):
            return NORMAL
        if v <= lo_c or v >= hi_c:
            return CRITIQUE
        if v <= lo_w or v >= hi_w:
            return MODERE
        return NORMAL

    return series.apply(_label)


def detect_bollinger(
    series: pd.Series,
    window: int = 20,
    k_warn: float = 2.0,
    k_crit: float = 3.0,
) -> pd.Series:
    """
    Bandes de Bollinger : seuils adaptatifs moy ± k*σ sur fenêtre glissante.
    Idéal pour les prix et cours qui évoluent dans le temps.
    """
    roll_mean = series.rolling(window, min_periods=window // 2).mean()
    roll_std  = series.rolling(window, min_periods=window // 2).std()
    upper_w = roll_mean + k_warn * roll_std
    lower_w = roll_mean - k_warn * roll_std
    upper_c = roll_mean + k_crit * roll_std
    lower_c = roll_mean - k_crit * roll_std

    def _label(idx):
        v = series.iloc[idx] if isinstance(idx, int) else series[idx]
        uw = upper_w.iloc[idx] if isinstance(idx, int) else upper_w[idx]
        lw = lower_w.iloc[idx] if isinstance(idx, int) else lower_w[idx]
        uc = upper_c.iloc[idx] if isinstance(idx, int) else upper_c[idx]
        lc = lower_c.iloc[idx] if isinstance(idx, int) else lower_c[idx]
        if pd.isna(v) or pd.isna(uw):
            return NORMAL
        if v >= uc or v <= lc:
            return CRITIQUE
        if v >= uw or v <= lw:
            return MODERE
        return NORMAL

    return pd.Series([_label(i) for i in range(len(series))], index=series.index)


def detect_cusum(
    series: pd.Series,
    k: float = 0.5,
    h: float = 5.0,
) -> pd.Series:
    """
    CUSUM (Cumulative Sum Control Chart) — détecte les shifts de moyenne.
    k = zone de tolérance (en σ), h = seuil de décision (en σ).
    Sensible aux dérives graduelles non détectées par le Z-score.
    """
    vals = series.dropna()
    mu, sigma = vals.mean(), vals.std()
    if sigma == 0:
        return pd.Series(NORMAL, index=series.index)

    cusum_pos = np.zeros(len(series))
    cusum_neg = np.zeros(len(series))
    labels = [NORMAL] * len(series)

    for i, v in enumerate(series):
        if pd.isna(v):
            continue
        z = (v - mu) / sigma
        if i > 0:
            cusum_pos[i] = max(0, cusum_pos[i-1] + z - k)
            cusum_neg[i] = max(0, cusum_neg[i-1] - z - k)
        if cusum_pos[i] > h or cusum_neg[i] > h:
            labels[i] = CRITIQUE if (cusum_pos[i] > 2*h or cusum_neg[i] > 2*h) else MODERE

    return pd.Series(labels, index=series.index)


# ═══════════════════════════════════════════════════════════════════════════
# 2. DÉTECTION MULTI-MÉTHODES PAR INDICATEUR
# ═══════════════════════════════════════════════════════════════════════════

def _severity_to_score(sev: str) -> int:
    """Convertit la sévérité en score numérique."""
    return {NORMAL: 0, FAIBLE: 1, MODERE: 2, CRITIQUE: 3}.get(sev, 0)


def _score_to_severity(score: float) -> str:
    """Score agrégé → sévérité finale."""
    if score >= 2.5:   return CRITIQUE
    if score >= 1.5:   return MODERE
    if score >= 0.5:   return FAIBLE
    return NORMAL


def detect_multi_method(
    series: pd.Series,
    methods: list[str] | None = None,
    window_rolling: int = 20,
) -> pd.DataFrame:
    """
    Applique plusieurs méthodes de détection et retourne un consensus.

    Paramètres
    ----------
    series  : pd.Series avec index = dates
    methods : liste parmi ['zscore','rolling_zscore','percentile','iqr','bollinger','cusum']
              None = toutes les méthodes

    Retourne
    --------
    DataFrame avec une colonne par méthode + colonne consensus (vote majoritaire).
    """
    if methods is None:
        methods = ['zscore', 'rolling_zscore', 'percentile', 'iqr', 'bollinger']

    results = pd.DataFrame(index=series.index)
    results['Valeur'] = series

    if 'zscore' in methods:
        results['Z_Score']        = detect_zscore(series)
    if 'rolling_zscore' in methods:
        results['Z_Rolling']      = detect_rolling_zscore(series, window=window_rolling)
    if 'percentile' in methods:
        results['Percentile']     = detect_percentile(series)
    if 'iqr' in methods:
        results['IQR']            = detect_iqr(series)
    if 'bollinger' in methods:
        results['Bollinger']      = detect_bollinger(series, window=window_rolling)
    if 'cusum' in methods:
        results['CUSUM']          = detect_cusum(series)

    method_cols = [c for c in results.columns if c != 'Valeur']

    # Score de consensus (moyenne des scores numériques)
    scores = results[method_cols].applymap(_severity_to_score)
    results['Score_Consensus'] = scores.mean(axis=1)
    results['Severite_Finale'] = results['Score_Consensus'].apply(_score_to_severity)
    results['Nb_Methodes_Alerte'] = (scores > 0).sum(axis=1)

    return results


# ═══════════════════════════════════════════════════════════════════════════
# 3. DÉTECTION PAR NIVEAU (MARCHÉ / INSTRUMENT / ORDERFLOW)
# ═══════════════════════════════════════════════════════════════════════════

def detect_market_anomalies(df_market: pd.DataFrame) -> pd.DataFrame:
    """
    Détecte les anomalies sur les indicateurs marché global.

    Retourne un DataFrame long avec toutes les anomalies détectées.
    """
    records = []
    indicator_config = {
        'Volume_MAD':              {'methods': ['zscore','rolling_zscore','percentile','iqr'], 'window': 20},
        'Volume_Relatif_Marche':   {'methods': ['zscore','percentile','bollinger'], 'window': 20},
        'MASI_Return_pct':         {'methods': ['zscore','rolling_zscore','percentile','iqr'], 'window': 20},
        'MASI_Vol_20j':            {'methods': ['zscore','rolling_zscore','percentile'], 'window': 30},
        'Breadth_pct':             {'methods': ['zscore','percentile','cusum'], 'window': 20},
        'HHI_Volume':              {'methods': ['zscore','percentile','bollinger'], 'window': 30},
        'AD_Line':                 {'methods': ['zscore','percentile','cusum'], 'window': 30},
        'MASI_VaR_Hist_95':        {'methods': ['zscore','rolling_zscore','percentile'], 'window': 40},
        'MASI_VaR_Hist_99':        {'methods': ['zscore','rolling_zscore','percentile'], 'window': 40},
    }

    for col, cfg in indicator_config.items():
        if col not in df_market.columns:
            continue
        series = df_market.set_index('Jour')[col].dropna()
        det = detect_multi_method(series, methods=cfg['methods'], window_rolling=cfg['window'])
        anomalies = det[det['Severite_Finale'] != NORMAL].copy()
        if len(anomalies) == 0:
            continue
        anomalies = anomalies.reset_index().rename(columns={'index': 'Jour'})
        anomalies['Indicateur']    = col
        anomalies['Niveau']        = 'Marché Global'
        anomalies['Ticker']        = 'MARCHE'
        records.append(anomalies[['Jour', 'Ticker', 'Niveau', 'Indicateur',
                                    'Valeur', 'Severite_Finale', 'Score_Consensus',
                                    'Nb_Methodes_Alerte']])

    if not records:
        return pd.DataFrame()
    return (pd.concat(records, ignore_index=True)
              .sort_values(['Severite_Finale', 'Score_Consensus'], ascending=[True, False])
              .reset_index(drop=True))


def detect_instrument_anomalies(
    df_instr: pd.DataFrame,
    min_seances: int = 30,
) -> pd.DataFrame:
    """
    Détecte les anomalies par instrument (niveau micro).

    Utilise 3 méthodes pour chaque indicateur clé :
    - Z-score rolling (contexte récent)
    - Percentile (seuils historiques globaux)
    - IQR (robustesse)

    Paramètres
    ----------
    min_seances : nb minimum de séances pour qu'un ticker soit analysé.

    Retourne
    --------
    DataFrame long d'anomalies avec colonne Ticker.
    """
    indicator_config = {
        'Rendement_pct':    {'methods': ['zscore','rolling_zscore','percentile','iqr'], 'window': 20, 'desc': 'Rendement anormal'},
        'Volume_Relatif':   {'methods': ['zscore','percentile','iqr'], 'window': 20, 'desc': 'Spike de volume'},
        'Volatilite_20j':   {'methods': ['zscore','rolling_zscore','percentile'], 'window': 30, 'desc': 'Volatilité extrême'},
        'Spread_pct':       {'methods': ['zscore','percentile','iqr'], 'window': 20, 'desc': 'Spread bid-ask anormal'},
        'Turnover_Ratio':   {'methods': ['zscore','percentile'], 'window': 20, 'desc': 'Turnover anormal'},
        'RSI_14j':          {'methods': ['percentile'], 'window': 14, 'desc': 'RSI extrême (survente/surachat)'},
    }

    records = []
    tickers = df_instr['Ticker'].unique()

    for ticker in tickers:
        df_t = df_instr[df_instr['Ticker'] == ticker].set_index('Jour').sort_index()
        if len(df_t) < min_seances:
            continue
        libelle = df_t['Libelle'].iloc[0] if 'Libelle' in df_t.columns else ticker

        for col, cfg in indicator_config.items():
            if col not in df_t.columns:
                continue
            series = df_t[col].dropna()
            if len(series) < max(10, cfg['window']):
                continue
            det = detect_multi_method(series, methods=cfg['methods'], window_rolling=cfg['window'])
            anomalies = det[det['Severite_Finale'] != NORMAL].copy()
            if len(anomalies) == 0:
                continue
            anomalies = anomalies.reset_index().rename(columns={'index': 'Jour'})
            anomalies['Ticker']         = ticker
            anomalies['Libelle']        = libelle
            anomalies['Niveau']         = 'Instrument'
            anomalies['Indicateur']     = col
            anomalies['Description']    = cfg['desc']
            records.append(anomalies[['Jour', 'Ticker', 'Libelle', 'Niveau',
                                        'Indicateur', 'Description',
                                        'Valeur', 'Severite_Finale',
                                        'Score_Consensus', 'Nb_Methodes_Alerte']])

    if not records:
        return pd.DataFrame()
    df_out = pd.concat(records, ignore_index=True)
    sev_order = {CRITIQUE: 4, MODERE: 3, FAIBLE: 2, NORMAL: 1}
    df_out['_sev_num'] = df_out['Severite_Finale'].map(sev_order)
    return (df_out.sort_values(['_sev_num', 'Score_Consensus'], ascending=[False, False])
                  .drop(columns=['_sev_num'])
                  .reset_index(drop=True))


def detect_orderflow_anomalies(df_of: pd.DataFrame) -> pd.DataFrame:
    """
    Détecte les anomalies sur les indicateurs de flux d'ordres.
    """
    indicator_config = {
        'OIR':                  {'methods': ['zscore','percentile','iqr'], 'window': 10, 'desc': 'Déséquilibre ordres (OIR)'},
        'OAR':                  {'methods': ['zscore','percentile','bollinger'], 'window': 10, 'desc': 'Pic d\'arrivée d\'ordres (OAR)'},
        'Volatilite_Intraday':  {'methods': ['zscore','percentile','iqr'], 'window': 10, 'desc': 'Volatilité intraday élevée'},
        'VWAP_Dev_pct':         {'methods': ['zscore','percentile','iqr'], 'window': 10, 'desc': 'Déviation VWAP vs clôture'},
        'Nb_Transactions':      {'methods': ['zscore','percentile','bollinger'], 'window': 10, 'desc': 'Nb transactions anormal'},
    }

    records = []
    for ticker, grp in df_of.groupby('Ticker'):
        grp = grp.set_index('Jour').sort_index()
        if len(grp) < 5:
            continue
        libelle = grp['Libelle'].iloc[0] if 'Libelle' in grp.columns else ticker

        for col, cfg in indicator_config.items():
            if col not in grp.columns:
                continue
            series = grp[col].dropna()
            if len(series) < 3:
                continue
            det = detect_multi_method(series, methods=cfg['methods'], window_rolling=cfg['window'])
            anomalies = det[det['Severite_Finale'] != NORMAL].copy()
            if len(anomalies) == 0:
                continue
            anomalies = anomalies.reset_index().rename(columns={'index': 'Jour'})
            anomalies['Ticker']      = ticker
            anomalies['Libelle']     = libelle
            anomalies['Niveau']      = 'Flux d\'Ordres'
            anomalies['Indicateur']  = col
            anomalies['Description'] = cfg['desc']
            records.append(anomalies[['Jour', 'Ticker', 'Libelle', 'Niveau',
                                        'Indicateur', 'Description',
                                        'Valeur', 'Severite_Finale',
                                        'Score_Consensus', 'Nb_Methodes_Alerte']])

    if not records:
        return pd.DataFrame()
    df_out = pd.concat(records, ignore_index=True)
    sev_order = {CRITIQUE: 4, MODERE: 3, FAIBLE: 2, NORMAL: 1}
    df_out['_sev_num'] = df_out['Severite_Finale'].map(sev_order)
    return (df_out.sort_values(['_sev_num', 'Score_Consensus'], ascending=[False, False])
                  .drop(columns=['_sev_num'])
                  .reset_index(drop=True))


# ═══════════════════════════════════════════════════════════════════════════
# 4. CONSOLIDATION ET EXPORT
# ═══════════════════════════════════════════════════════════════════════════

def consolidate_all_anomalies(
    df_market_anom: pd.DataFrame,
    df_instr_anom: pd.DataFrame,
    df_of_anom: pd.DataFrame,
) -> pd.DataFrame:
    """
    Fusionne toutes les anomalies des 3 niveaux en un seul DataFrame trié.
    """
    parts = [df for df in [df_market_anom, df_instr_anom, df_of_anom] if len(df) > 0]
    if not parts:
        return pd.DataFrame()

    df_all = pd.concat(parts, ignore_index=True)

    # Remplir les colonnes manquantes entre DataFrames
    for col in ['Libelle', 'Description']:
        if col not in df_all.columns:
            df_all[col] = ''
        df_all[col] = df_all[col].fillna('')

    sev_order = {CRITIQUE: 4, MODERE: 3, FAIBLE: 2, NORMAL: 1}
    df_all['_sev_num'] = df_all['Severite_Finale'].map(sev_order)
    df_all = df_all.sort_values(['_sev_num', 'Score_Consensus'], ascending=[False, False])
    df_all = df_all.drop(columns=['_sev_num']).reset_index(drop=True)
    df_all.index = df_all.index + 1  # numérotation à partir de 1
    return df_all


def compute_seuils_table(
    df_instr: pd.DataFrame,
    df_market: pd.DataFrame | None = None,
    df_of: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Génère un tableau récapitulatif des seuils statistiques pour tous les indicateurs.
    Utile pour le rapport et le dashboard.
    """
    rows = []

    def _add(serie, nom, niveau, methode='Z-score + Percentile'):
        s = serie.dropna()
        if len(s) < 5:
            return
        mu, sigma = s.mean(), s.std()
        rows.append({
            'Niveau': niveau, 'Indicateur': nom, 'Méthode': methode,
            'N': len(s), 'Moyenne': round(mu, 4), 'Écart-type': round(sigma, 4),
            'Médiane': round(s.median(), 4),
            'P1': round(s.quantile(0.01), 4), 'P5': round(s.quantile(0.05), 4),
            'P95': round(s.quantile(0.95), 4), 'P99': round(s.quantile(0.99), 4),
            'Seuil_Alerte_Bas':    round(mu - 2 * sigma, 4),
            'Seuil_Alerte_Haut':   round(mu + 2 * sigma, 4),
            'Seuil_Critique_Bas':  round(mu - 3 * sigma, 4),
            'Seuil_Critique_Haut': round(mu + 3 * sigma, 4),
        })

    # Marché global
    if df_market is not None:
        for col, nom in [
            ('Volume_MAD', 'Volume Marché (MAD)'),
            ('Volume_Relatif_Marche', 'Volume Relatif Marché'),
            ('MASI_Return_pct', 'MASI Rendement (%)'),
            ('MASI_Vol_20j', 'MASI Volatilité 20j'),
            ('Breadth_pct', 'Breadth (% hausse)'),
            ('AD_Line', 'Advance-Decline Line (cumul)'),
            ('MASI_VaR_Hist_95', 'MASI VaR historique 95 %'),
            ('MASI_VaR_Hist_99', 'MASI VaR historique 99 %'),
        ]:
            if col in df_market.columns:
                _add(df_market[col], nom, 'Marché Global')

    # Instruments
    for col, nom in [
        ('Rendement_pct', 'Rendement Instrument (%)'),
        ('Volatilite_5j', 'Volatilité 5j (%)'),
        ('Volatilite_20j', 'Volatilité 20j (%)'),
        ('Volume_Relatif', 'Volume Relatif Instrument'),
        ('Spread_pct', 'Spread Bid-Ask (%)'),
        ('Turnover_Ratio', 'Turnover Ratio'),
        ('RSI_14j', 'RSI 14j'),
    ]:
        if col in df_instr.columns:
            _add(df_instr[col], nom, 'Instrument')

    # Order flow
    if df_of is not None:
        for col, nom in [
            ('OIR', 'OIR (Lee-Ready)'),
            ('OAR', 'OAR (transactions/h)'),
            ('Volatilite_Intraday', 'Volatilité Intraday (%)'),
            ('VWAP_Dev_pct', 'VWAP Déviation (%)'),
        ]:
            if col in df_of.columns:
                _add(df_of[col], nom, 'Flux d\'Ordres')

    return pd.DataFrame(rows)


def export_anomalies_excel(
    df_all: pd.DataFrame,
    df_seuils: pd.DataFrame,
    df_market_anom: pd.DataFrame,
    df_instr_anom: pd.DataFrame,
    df_of_anom: pd.DataFrame,
    output_path: str = '../reports/anomalies_statistiques_2025.xlsx',
) -> str:
    """
    Exporte toutes les anomalies et les seuils dans un fichier Excel structuré.
    Format conçu pour la validation avec l'encadrante.
    """
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Feuille 1 : Toutes anomalies
        df_all.to_excel(writer, sheet_name='Toutes Anomalies', index=True)

        # Feuille 2 : Marché Global
        if len(df_market_anom) > 0:
            df_market_anom.to_excel(writer, sheet_name='Marché Global', index=False)

        # Feuille 3 : Instruments
        if len(df_instr_anom) > 0:
            df_instr_anom.to_excel(writer, sheet_name='Instruments', index=False)

        # Feuille 4 : Flux d'Ordres
        if len(df_of_anom) > 0:
            df_of_anom.to_excel(writer, sheet_name='Flux Ordres', index=False)

        # Feuille 5 : Seuils statistiques
        df_seuils.to_excel(writer, sheet_name='Seuils Statistiques', index=False)

        # Feuille 6 : Résumé par ticker
        if 'Ticker' in df_all.columns:
            resume = (df_all.groupby(['Ticker', 'Severite_Finale'])
                      .size().unstack(fill_value=0).reset_index())
            resume.to_excel(writer, sheet_name='Résumé par Ticker', index=False)

    return output_path


# ═══════════════════════════════════════════════════════════════════════════
# 5. FONCTIONS UTILITAIRES POUR LE DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════

def get_anomalies_for_ticker(
    df_all: pd.DataFrame,
    ticker: str,
    severite: str | None = None,
) -> pd.DataFrame:
    """Filtre les anomalies pour un ticker donné."""
    df = df_all[df_all['Ticker'] == ticker].copy()
    if severite:
        df = df[df['Severite_Finale'] == severite]
    return df.sort_values('Jour')


def get_anomalies_for_date(
    df_all: pd.DataFrame,
    date: str,
    severite_min: str = MODERE,
) -> pd.DataFrame:
    """Retourne toutes les anomalies pour une date donnée (≥ sévérité min)."""
    sev_order = {CRITIQUE: 4, MODERE: 3, FAIBLE: 2, NORMAL: 1}
    min_score = sev_order.get(severite_min, 3)
    df = df_all[df_all['Jour'] == pd.Timestamp(date)].copy()
    df['_sev_num'] = df['Severite_Finale'].map(sev_order)
    return df[df['_sev_num'] >= min_score].drop(columns=['_sev_num'])


def summarize_anomalies(df_all: pd.DataFrame) -> dict:
    """Retourne un dictionnaire de KPIs de surveillance."""
    sev_counts = df_all['Severite_Finale'].value_counts().to_dict()
    return {
        'Total': len(df_all),
        'Critique': sev_counts.get(CRITIQUE, 0),
        'Modéré': sev_counts.get(MODERE, 0),
        'Faible': sev_counts.get(FAIBLE, 0),
        'Instruments_Alertés': df_all[df_all['Ticker'] != 'MARCHE']['Ticker'].nunique(),
        'Indicateurs_Impliqués': df_all['Indicateur'].nunique(),
        'Jours_avec_Alerte': df_all['Jour'].nunique(),
        'Top_Ticker': (df_all[df_all['Ticker'] != 'MARCHE']
                        .groupby('Ticker').size().idxmax()
                        if len(df_all) > 0 else 'N/A'),
    }
