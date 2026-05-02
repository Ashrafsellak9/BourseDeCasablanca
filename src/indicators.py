"""
Module de calcul des indicateurs de surveillance BVC.

Couvre trois niveaux :
  1. Marché global      — indicateurs agrégés (MASI, volume, breadth)
  2. Instrument         — indicateurs par titre (volatilité, liquidité, OIR)
  3. Flux d'ordres      — indicateurs intraday (VWAP, OAR, Lee-Ready OIR)

Chaque fonction retourne un DataFrame enrichi prêt à l'utilisation
dans la détection d'anomalies (Phase 3) et le dashboard (Phase 4).
"""

from pathlib import Path

import pandas as pd
import numpy as np
from scipy import stats
import warnings

from src.instrument_segment import (
    assign_instrument_segment,
    assign_instrument_secteur,
    enrich_market_with_segment_volumes,
)

warnings.filterwarnings('ignore')

_DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# VaR historique MASI : fenêtre glissante (rendements journaliers en %)
MASI_VAR_HIST_WINDOW = 252
MASI_VAR_HIST_MIN_PERIODS = 60


def compute_advance_decline_line(df_cours: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule la ligne Advance-Decline cumulée par séance.

    Pour chaque jour : ``Advances`` = titres avec rendement **> 0**,
    ``Declines`` = titres avec rendement **< 0** (les flats ne comptent ni en hausse ni en baisse).
    ``AD_Net = Advances - Declines``, ``AD_Line`` = somme cumulée (tendance long terme de la participation).
    """
    if df_cours.empty or "Rendement_pct" not in df_cours.columns or "Jour" not in df_cours.columns:
        return pd.DataFrame(columns=["Jour", "Advances", "Declines", "AD_Net", "AD_Line"])

    g = df_cours.dropna(subset=["Jour", "Rendement_pct"]).copy()
    g["Jour"] = pd.to_datetime(g["Jour"])
    g["_adv"] = (g["Rendement_pct"] > 0).astype(np.int32)
    g["_dec"] = (g["Rendement_pct"] < 0).astype(np.int32)
    daily = (
        g.groupby("Jour", observed=True)
        .agg(Advances=("_adv", "sum"), Declines=("_dec", "sum"))
        .reset_index()
    )
    daily["AD_Net"] = daily["Advances"] - daily["Declines"]
    daily["AD_Line"] = daily["AD_Net"].cumsum()
    return daily.sort_values("Jour").reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════
# 1. INDICATEURS MARCHÉ GLOBAL
# ═══════════════════════════════════════════════════════════════════════════

def compute_market_indicators(
    df_ind: pd.DataFrame,
    df_idx: pd.DataFrame,
    df_cours: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calcule les indicateurs au niveau marché global (journaliers).

    Paramètres
    ----------
    df_ind   : Feuille Indicateurs (Volume_MAD, Quantite_Titres, Nb_Contrats)
    df_idx   : Feuille Indices (MASI, MSI20)
    df_cours : Feuille Cours (pour breadth et concentration)

    Retourne
    --------
    DataFrame journalier avec tous les indicateurs marché
    (dont **VaR historique** MASI 95 % / 99 % sur rendements quotidiens).
    """
    # --- Base : indicateurs agrégés ---
    df = df_ind[['Jour', 'Volume_MAD', 'Quantite_Titres', 'Nb_Contrats']].copy()

    # --- MASI ---
    masi = (df_idx[df_idx['Code_Indice'] == 'MASI']
            [['Jour', 'Cours_Haut', 'Variation_Veille_pct', 'Variation_YTD_pct']]
            .rename(columns={
                'Cours_Haut': 'MASI',
                'Variation_Veille_pct': 'MASI_Var_Veille_pct',
                'Variation_YTD_pct': 'MASI_Var_YTD_pct'
            }))
    msi20 = (df_idx[df_idx['Code_Indice'] == 'MSI20']
             [['Jour', 'Cours_Haut', 'Variation_Veille_pct']]
             .rename(columns={
                 'Cours_Haut': 'MSI20',
                 'Variation_Veille_pct': 'MSI20_Var_Veille_pct'
             }))
    df = df.merge(masi, on='Jour', how='left').merge(msi20, on='Jour', how='left')

    # --- Rendement MASI (depuis cours MASI) ---
    df['MASI_Return_pct'] = df['MASI'].pct_change() * 100

    # --- Rendement MSI 20 + corrélation glissante MASI / MSI20 (divergences inter-indices) ---
    if 'MSI20' in df.columns:
        df['MSI20_Return_pct'] = df['MSI20'].pct_change() * 100
        df['Corr_MASI_MSI20_20j'] = (
            df['MASI_Return_pct']
            .rolling(20, min_periods=10)
            .corr(df['MSI20_Return_pct'])
        )

    # --- Volatilité rolling du MASI ---
    df['MASI_Vol_5j'] = df['MASI_Return_pct'].rolling(5, min_periods=3).std()
    df['MASI_Vol_20j'] = df['MASI_Return_pct'].rolling(20, min_periods=10).std()

    # --- VaR historique MASI (rendements % ; quantiles de la fenêtre glissante) ---
    # Pertes exprimées en valeur positive (%): VaR = -Q_alpha(r), recalculé chaque jour.
    _r_m = df['MASI_Return_pct']
    df['MASI_VaR_Hist_95'] = (
        -_r_m.rolling(MASI_VAR_HIST_WINDOW, min_periods=MASI_VAR_HIST_MIN_PERIODS).quantile(0.05)
    ).clip(lower=0)
    df['MASI_VaR_Hist_99'] = (
        -_r_m.rolling(MASI_VAR_HIST_WINDOW, min_periods=MASI_VAR_HIST_MIN_PERIODS).quantile(0.01)
    ).clip(lower=0)

    # --- Volume relatif du marché ---
    df['Vol_Moyen_20j'] = df['Volume_MAD'].rolling(20, min_periods=5).mean()
    df['Volume_Relatif_Marche'] = df['Volume_MAD'] / df['Vol_Moyen_20j']

    # --- Volume par contrat (taille moyenne d'une transaction) ---
    df['Volume_Par_Contrat'] = df['Volume_MAD'] / df['Nb_Contrats'].replace(0, np.nan)

    # --- Breadth du marché : % instruments en hausse ---
    if 'Rendement_pct' in df_cours.columns:
        breadth = (df_cours.groupby('Jour')
                   .apply(lambda g: (g['Rendement_pct'].dropna() > 0).sum() /
                          max(g['Rendement_pct'].dropna().count(), 1) * 100)
                   .reset_index(name='Breadth_pct'))
        df = df.merge(breadth, on='Jour', how='left')

    # --- Advance-Decline Line (cumul hausses − baisses par séance) ---
    if 'Rendement_pct' in df_cours.columns and not df_cours.empty:
        df = df.merge(compute_advance_decline_line(df_cours), on='Jour', how='left')

    # --- Concentration de marché : HHI des volumes par instrument ---
    if 'Volume_MAD' in df_cours.columns:
        hhi = (df_cours.groupby('Jour')
               .apply(_hhi_volume)
               .reset_index(name='HHI_Volume'))
        df = df.merge(hhi, on='Jour', how='left')

    # --- Concentration : part du volume détenue par les 5 plus gros titres (feuille Cours) ---
    if not df_cours.empty and "Volume_MAD" in df_cours.columns:
        df = df.merge(daily_top5_volume_concentration(df_cours), on="Jour", how="left")

    # --- Volume par segment (actions / OPCVM / obligations) depuis la feuille Cours ---
    if not df_cours.empty and 'Volume_MAD' in df_cours.columns and 'Ticker' in df_cours.columns:
        df = enrich_market_with_segment_volumes(df, df_cours)

    # --- Z-scores journaliers ---
    _zcols = ['Volume_MAD', 'MASI_Return_pct', 'MASI_Vol_20j', 'Volume_Relatif_Marche']
    if 'Volume_Top5_Pct' in df.columns:
        _zcols.append('Volume_Top5_Pct')
    if 'AD_Line' in df.columns:
        _zcols.append('AD_Line')
    if 'MASI_VaR_Hist_95' in df.columns:
        _zcols.append('MASI_VaR_Hist_95')
    if 'MASI_VaR_Hist_99' in df.columns:
        _zcols.append('MASI_VaR_Hist_99')
    for col in _zcols:
        if col in df.columns:
            df[f'Z_{col}'] = _zscore_series(df[col])

    df = df.sort_values('Jour').reset_index(drop=True)
    return df


def enrich_market_masi_msi20_rolling_corr(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pour Parquets anciens : ajoute ``MSI20_Return_pct`` et ``Corr_MASI_MSI20_20j``
    si MASI / MSI20 sont présents.
    """
    if df.empty or 'MSI20' not in df.columns or 'MASI' not in df.columns:
        return df
    if 'Corr_MASI_MSI20_20j' in df.columns:
        return df
    out = df.sort_values('Jour').copy()
    if 'MASI_Return_pct' not in out.columns:
        out['MASI_Return_pct'] = out['MASI'].pct_change() * 100
    out['MSI20_Return_pct'] = out['MSI20'].pct_change() * 100
    out['Corr_MASI_MSI20_20j'] = (
        out['MASI_Return_pct']
        .rolling(20, min_periods=10)
        .corr(out['MSI20_Return_pct'])
    )
    return out


def enrich_market_historic_var(
    df: pd.DataFrame,
    window: int = MASI_VAR_HIST_WINDOW,
    min_periods: int = MASI_VAR_HIST_MIN_PERIODS,
) -> pd.DataFrame:
    """
    Ajoute ``MASI_VaR_Hist_95`` et ``MASI_VaR_Hist_99`` si absents (Parquets anciens).
    """
    if df.empty or 'MASI_Return_pct' not in df.columns:
        return df
    if 'MASI_VaR_Hist_95' in df.columns and 'MASI_VaR_Hist_99' in df.columns:
        return df
    out = df.sort_values('Jour').copy()
    r = out['MASI_Return_pct']
    out['MASI_VaR_Hist_95'] = (-r.rolling(window, min_periods=min_periods).quantile(0.05)).clip(lower=0)
    out['MASI_VaR_Hist_99'] = (-r.rolling(window, min_periods=min_periods).quantile(0.01)).clip(lower=0)
    return out


def _hhi_volume(grp: pd.DataFrame) -> float:
    """Indice Herfindahl-Hirschman des volumes (concentration)."""
    vol = grp['Volume_MAD'].fillna(0)
    total = vol.sum()
    if total == 0:
        return np.nan
    shares = vol / total
    return (shares ** 2).sum()


def _top5_volume_share_pct(grp: pd.DataFrame) -> float:
    """Part du volume du jour détenue par les 5 instruments les plus actifs (0–100)."""
    vol = grp["Volume_MAD"].fillna(0)
    total = float(vol.sum())
    if total <= 0:
        return np.nan
    return float(vol.nlargest(5).sum() / total * 100.0)


def daily_top5_volume_concentration(df_cours: pd.DataFrame) -> pd.DataFrame:
    """
    Pour chaque séance : % du volume total représenté par les 5 plus gros volumes instrument.

    Colonnes retournées : ``Jour``, ``Volume_Top5_Pct``.
    """
    if df_cours.empty or "Volume_MAD" not in df_cours.columns or "Jour" not in df_cours.columns:
        return pd.DataFrame(columns=["Jour", "Volume_Top5_Pct"])
    d = df_cours.copy()
    d["Jour"] = pd.to_datetime(d["Jour"])
    conc = (
        d.groupby("Jour", observed=True)
        .apply(_top5_volume_share_pct)
        .reset_index(name="Volume_Top5_Pct")
    )
    return conc


# ═══════════════════════════════════════════════════════════════════════════
# 2. INDICATEURS PAR INSTRUMENT
# ═══════════════════════════════════════════════════════════════════════════


def _add_peer_volatility_vs_sector(df: pd.DataFrame) -> pd.DataFrame:
    """Colonnes de volatilité 20j vs pairs du même ``Secteur`` le même jour."""
    if "Secteur" not in df.columns or "Volatilite_20j" not in df.columns:
        return df
    g_peer = df.groupby(["Jour", "Secteur"], observed=True)
    df = df.copy()
    df["Peers_Secteur_Nb"] = g_peer["Ticker"].transform("nunique")
    df["Volatilite_Pairs_Median"] = g_peer["Volatilite_20j"].transform("median")
    df["Volatilite_Pairs_Std"] = g_peer["Volatilite_20j"].transform("std")
    _solo = df["Peers_Secteur_Nb"] < 2
    df.loc[_solo, "Volatilite_Pairs_Median"] = np.nan
    df.loc[_solo, "Volatilite_Pairs_Std"] = np.nan
    _med = df["Volatilite_Pairs_Median"].replace(0, np.nan)
    df["Volatilite_vs_Pairs_Ratio"] = df["Volatilite_20j"] / _med
    _std_ok = df["Volatilite_Pairs_Std"].replace(0, np.nan)
    df["Z_Volatilite_vs_Pairs"] = np.where(
        (df["Peers_Secteur_Nb"] >= 3) & _std_ok.notna(),
        (df["Volatilite_20j"] - df["Volatilite_Pairs_Median"]) / _std_ok,
        np.nan,
    )
    return df


def enrich_instrument_sector_peer_volatility(
    df: pd.DataFrame,
    data_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Pour Parquets anciens : ajoute ``Secteur`` et les colonnes *pairs sectoriels*
    sans recalculer tout le pipeline.
    """
    if df.empty or "Volatilite_20j" not in df.columns:
        return df
    if "Volatilite_vs_Pairs_Ratio" in df.columns:
        return df
    out = assign_instrument_secteur(df.copy(), data_dir or _DEFAULT_DATA_DIR)
    return _add_peer_volatility_vs_sector(out)


def compute_instrument_indicators(
    df_cours: pd.DataFrame,
    data_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Calcule les indicateurs journaliers par instrument.

    Indicateurs calculés
    --------------------
    Rendement journalier, volatilités rolling (5j/10j/20j),
    volume relatif, turnover, spread, range, beta, RSI 14j,
    Z-scores individuels et percentiles.

    Retourne
    --------
    DataFrame enrichi trié par (Ticker, Jour).
    """
    _dd = data_dir or _DEFAULT_DATA_DIR

    df = df_cours.copy()
    df = df.sort_values(['Ticker', 'Jour']).reset_index(drop=True)

    df = assign_instrument_segment(df, _dd)
    df = assign_instrument_secteur(df, _dd)

    # --- Rendement journalier ---
    df['Rendement_pct'] = df.groupby('Ticker')['Cours_Cloture'].pct_change() * 100

    # --- Rendement absolu (écart cours_ref → clôture) ---
    df['Ecart_Ref_Cloture_pct'] = ((df['Cours_Cloture'] - df['Cours_Ref'])
                                    / df['Cours_Ref'].replace(0, np.nan) * 100)

    # --- Range journalier ---
    df['Range_pct'] = ((df['Cours_Haut'] - df['Cours_Bas'])
                       / df['Cours_Ref'].replace(0, np.nan) * 100)

    # --- Spread bid-ask ---
    df['Spread_pct'] = ((df['Meilleure_Offre'] - df['Meilleure_Demande']).abs()
                        / df['Cours_Ref'].replace(0, np.nan) * 100)

    # --- Volatilités rolling ---
    for w in [5, 10, 20]:
        df[f'Volatilite_{w}j'] = (df.groupby('Ticker')['Rendement_pct']
                                   .transform(lambda x: x.rolling(w, min_periods=max(3, w // 2)).std()))

    # --- Volume relatif (vs moyenne 20j) ---
    df['Vol_Moyen_20j'] = (df.groupby('Ticker')['Volume_MAD']
                            .transform(lambda x: x.rolling(20, min_periods=5).mean()))
    df['Volume_Relatif'] = df['Volume_MAD'] / df['Vol_Moyen_20j'].replace(0, np.nan)

    # --- Turnover ratio ---
    df['Turnover_Ratio'] = df['Volume_MAD'] / df['Capitalisation'].replace(0, np.nan)

    # --- RSI 14 jours ---
    df['RSI_14j'] = df.groupby('Ticker')['Rendement_pct'].transform(_rsi_14)

    # --- Momentum 5j et 20j (rendement cumulé) ---
    df['Momentum_5j'] = (df.groupby('Ticker')['Cours_Cloture']
                          .transform(lambda x: x.pct_change(5) * 100))
    df['Momentum_20j'] = (df.groupby('Ticker')['Cours_Cloture']
                           .transform(lambda x: x.pct_change(20) * 100))

    # --- Z-scores par instrument (rolling 60j) ---
    df['Z_Rendement'] = (df.groupby('Ticker')['Rendement_pct']
                          .transform(_zscore_series))
    df['Z_Volume'] = (df.groupby('Ticker')['Volume_MAD']
                       .transform(_zscore_series))
    df['Z_Volatilite'] = (df.groupby('Ticker')['Volatilite_20j']
                           .transform(_zscore_series))

    # --- Percentile rank (cross-sectionnel par jour) ---
    df['Pct_Volume_Rank'] = (df.groupby('Jour')['Volume_MAD']
                              .rank(pct=True) * 100)
    df['Pct_Volatilite_Rank'] = (df.groupby('Jour')['Volatilite_20j']
                                   .rank(pct=True) * 100)

    # --- Volatilité vs pairs sectoriels (même Jour × Secteur) ---
    df = _add_peer_volatility_vs_sector(df)

    # --- Score de liquidité composite [0-1] ---
    df['Score_Liquidite'] = _score_liquidite(df)

    return df.sort_values(['Ticker', 'Jour']).reset_index(drop=True)


def _rsi_14(series: pd.Series, period: int = 14) -> pd.Series:
    """Calcule le RSI sur une période glissante."""
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period, min_periods=period // 2).mean()
    loss = (-delta.clip(upper=0)).rolling(period, min_periods=period // 2).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _score_liquidite(df: pd.DataFrame) -> pd.Series:
    """
    Score de liquidité composite normalisé [0, 1].
    Basé sur : Turnover (40%), Volume_Relatif (40%), -Spread (20%).
    """
    scores = pd.Series(np.nan, index=df.index)
    for jour, grp in df.groupby('Jour'):
        to = grp['Turnover_Ratio'].rank(pct=True).fillna(0.5)
        vr = grp['Volume_Relatif'].rank(pct=True).fillna(0.5)
        sp = (1 - grp['Spread_pct'].rank(pct=True)).fillna(0.5)
        scores.loc[grp.index] = 0.4 * to + 0.4 * vr + 0.2 * sp
    return scores


# ═══════════════════════════════════════════════════════════════════════════
# 3. INDICATEURS DE FLUX D'ORDRES (INTRADAY)
# ═══════════════════════════════════════════════════════════════════════════

def compute_orderflow_indicators(
    df_intra: pd.DataFrame,
    df_cours: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Calcule les indicateurs de flux d'ordres depuis les données intraday.

    Méthodes utilisées
    ------------------
    - Lee-Ready tick rule : classifie chaque transaction comme
      buy-initiated (prix monte) ou sell-initiated (prix baisse)
    - VWAP : prix moyen pondéré par volume de la journée
    - OAR  : Order Arrival Rate (transactions / heure)
    - OIR  : Order Imbalance Ratio (Lee-Ready)

    Retourne
    --------
    DataFrame agrégé par (Jour, Ticker) avec tous les indicateurs.
    """
    # Dédupliquer les paires A/V → une ligne par transaction unique
    df = (df_intra[df_intra['Sens'] == 'A']
          .drop_duplicates(subset=['Num_Transaction'])
          .copy())
    df = df.sort_values(['Ticker', 'Jour', 'Heure_Transaction']).reset_index(drop=True)

    # --- Heure décimale ---
    if 'Heure_int' not in df.columns:
        df['Heure_int'] = pd.to_datetime(df['Heure_Transaction']).dt.hour
        df['Minute_int'] = pd.to_datetime(df['Heure_Transaction']).dt.minute
    df['Heure_Decimal'] = df['Heure_int'] + df['Minute_int'] / 60

    # --- Classification Lee-Ready (tick rule) ---
    df['Price_Change'] = df.groupby(['Ticker', 'Jour'])['Cours_Transaction'].diff()
    df['Tick_Direction'] = np.where(
        df['Price_Change'] > 0, 1,
        np.where(df['Price_Change'] < 0, -1, np.nan)
    )
    # Forward-fill pour les prix inchangés (uptick rule)
    df['Tick_Direction'] = (df.groupby(['Ticker', 'Jour'])['Tick_Direction']
                             .ffill().fillna(1))  # défaut = buy
    df['Is_Buy'] = df['Tick_Direction'] == 1
    df['Buy_Volume'] = df['Quantite_Titres'] * df['Is_Buy']
    df['Sell_Volume'] = df['Quantite_Titres'] * (~df['Is_Buy'])

    # === Agrégation par (Jour, Ticker) ===
    agg = df.groupby(['Jour', 'Ticker']).agg(
        Nb_Transactions=('Num_Transaction', 'count'),
        Volume_Intraday=('Quantite_Titres', 'sum'),
        Buy_Volume=('Buy_Volume', 'sum'),
        Sell_Volume=('Sell_Volume', 'sum'),
        Cours_Min=('Cours_Transaction', 'min'),
        Cours_Max=('Cours_Transaction', 'max'),
        Cours_Std=('Cours_Transaction', 'std'),
        VWAP=('Cours_Transaction', _vwap_agg),
        Duree_Active_min=('Heure_Decimal', lambda x: (x.max() - x.min()) * 60),
    ).reset_index()

    # --- OIR (Order Imbalance Ratio) ---
    agg['OIR'] = ((agg['Buy_Volume'] - agg['Sell_Volume'])
                  / (agg['Buy_Volume'] + agg['Sell_Volume']).replace(0, np.nan))

    # --- Order Arrival Rate (transactions / heure active) ---
    agg['OAR'] = agg['Nb_Transactions'] / agg['Duree_Active_min'].replace(0, np.nan) * 60

    # --- Volatilité intraday normalisée ---
    agg['Volatilite_Intraday'] = agg['Cours_Std'] / agg['VWAP'].replace(0, np.nan) * 100

    # --- Volume relatif intraday par heure ---
    vol_horaire = _volume_par_heure(df)
    agg = agg.merge(vol_horaire, on=['Jour', 'Ticker'], how='left')

    # --- Déviation VWAP vs cours de clôture ---
    if df_cours is not None and 'Cours_Cloture' in df_cours.columns:
        cloture = df_cours[['Jour', 'Ticker', 'Cours_Cloture', 'Cours_Ref']].copy()
        agg = agg.merge(cloture, on=['Jour', 'Ticker'], how='left')
        agg['VWAP_Dev_pct'] = (agg['VWAP'] - agg['Cours_Cloture']) / agg['Cours_Ref'] * 100
    else:
        agg['VWAP_Dev_pct'] = np.nan

    # --- Z-scores pour anomalies ---
    for col in ['OIR', 'OAR', 'Volatilite_Intraday', 'Nb_Transactions']:
        agg[f'Z_{col}'] = _zscore_series(agg[col])

    return agg.sort_values(['Jour', 'Ticker']).reset_index(drop=True)


def _vwap_agg(group_prix):
    """Calcule le VWAP pour une agrégation groupby (helper)."""
    return group_prix.mean()


def compute_vwap_exact(df_intra: pd.DataFrame) -> pd.DataFrame:
    """VWAP exact pondéré par quantité."""
    df = df_intra[df_intra['Sens'] == 'A'].copy()
    df['Volume_Value'] = df['Cours_Transaction'] * df['Quantite_Titres']
    vwap = (df.groupby(['Jour', 'Ticker'])
            .apply(lambda g: g['Volume_Value'].sum() / g['Quantite_Titres'].sum())
            .reset_index(name='VWAP'))
    return vwap


def _volume_par_heure(df: pd.DataFrame) -> pd.DataFrame:
    """Volume % concentré dans la 1ère et dernière heure de séance."""
    open_vol = (df[df['Heure_int'] == 9]
                .groupby(['Jour', 'Ticker'])['Quantite_Titres'].sum()
                .reset_index(name='Vol_Open'))
    close_vol = (df[df['Heure_int'] >= 15]
                 .groupby(['Jour', 'Ticker'])['Quantite_Titres'].sum()
                 .reset_index(name='Vol_Close'))
    total_vol = (df.groupby(['Jour', 'Ticker'])['Quantite_Titres'].sum()
                 .reset_index(name='Vol_Total'))
    merged = total_vol.merge(open_vol, on=['Jour', 'Ticker'], how='left')
    merged = merged.merge(close_vol, on=['Jour', 'Ticker'], how='left')
    merged['Pct_Vol_Open'] = merged['Vol_Open'] / merged['Vol_Total'].replace(0, np.nan) * 100
    merged['Pct_Vol_Close'] = merged['Vol_Close'] / merged['Vol_Total'].replace(0, np.nan) * 100
    return merged[['Jour', 'Ticker', 'Pct_Vol_Open', 'Pct_Vol_Close']]


# ═══════════════════════════════════════════════════════════════════════════
# 4. SCORING DES ALERTES (STATISTIQUE)
# ═══════════════════════════════════════════════════════════════════════════

def compute_alert_scores(
    df_instr: pd.DataFrame,
    df_market: pd.DataFrame | None = None,
    df_orderflow: pd.DataFrame | None = None,
    z_threshold: float = 2.5,
    p_high: float = 99,
    p_low: float = 1,
) -> pd.DataFrame:
    """
    Calcule un score d'alerte composite par (Jour, Ticker).

    Score [0–100] : somme pondérée d'alertes binaires sur :
      - Z-score rendement   (poids 25%)
      - Z-score volume      (poids 25%)
      - Z-score volatilité  (poids 20%)
      - OIR extrême         (poids 15%)
      - OAR extrême         (poids 15%)

    Seuils statistiques
    -------------------
    - Z > z_threshold           → alerte forte (score = 1)
    - Valeur > P99 ou < P01     → alerte extrême
    - P95 < valeur < P99        → alerte modérée (score = 0.5)

    Retourne
    --------
    DataFrame trié par score décroissant.
    """
    df = df_instr[['Jour', 'Ticker', 'Libelle',
                    'Rendement_pct', 'Volume_Relatif', 'Volatilite_20j',
                    'Turnover_Ratio', 'Spread_pct',
                    'Z_Rendement', 'Z_Volume', 'Z_Volatilite']].copy()

    # --- Seuils globaux ---
    p99_rend = df['Rendement_pct'].abs().quantile(p_high / 100)
    p95_rend = df['Rendement_pct'].abs().quantile(0.95)
    p99_vol_rel = df['Volume_Relatif'].quantile(p_high / 100)
    p95_vol_rel = df['Volume_Relatif'].quantile(0.95)
    p99_volat = df['Volatilite_20j'].quantile(p_high / 100)
    p95_volat = df['Volatilite_20j'].quantile(0.95)

    # --- Alertes rendement ---
    df['Alert_Rendement'] = (
        np.where(df['Rendement_pct'].abs() > p99_rend, 1.0,
        np.where(df['Rendement_pct'].abs() > p95_rend, 0.5, 0.0))
    )
    df['Alert_Rendement'] = np.maximum(
        df['Alert_Rendement'],
        df['Z_Rendement'].abs().gt(z_threshold).astype(float)
    )

    # --- Alertes volume ---
    df['Alert_Volume'] = (
        np.where(df['Volume_Relatif'] > p99_vol_rel, 1.0,
        np.where(df['Volume_Relatif'] > p95_vol_rel, 0.5, 0.0))
    )
    df['Alert_Volume'] = np.maximum(
        df['Alert_Volume'],
        df['Z_Volume'].abs().gt(z_threshold).astype(float)
    )

    # --- Alertes volatilité ---
    df['Alert_Volatilite'] = (
        np.where(df['Volatilite_20j'] > p99_volat, 1.0,
        np.where(df['Volatilite_20j'] > p95_volat, 0.5, 0.0))
    )

    # --- OIR et OAR (si disponible) ---
    if df_orderflow is not None:
        oir_cols = ['Jour', 'Ticker', 'OIR', 'OAR', 'Z_OIR', 'Z_OAR']
        oir_cols = [c for c in oir_cols if c in df_orderflow.columns]
        df = df.merge(df_orderflow[oir_cols], on=['Jour', 'Ticker'], how='left')

        p99_oir = df['OIR'].abs().quantile(p_high / 100) if 'OIR' in df.columns else np.nan
        df['Alert_OIR'] = (
            df['OIR'].abs().gt(p99_oir).astype(float) if 'OIR' in df.columns
            else 0.0
        )
        p99_oar = df['OAR'].quantile(p_high / 100) if 'OAR' in df.columns else np.nan
        df['Alert_OAR'] = (
            df['OAR'].gt(p99_oar).astype(float) if 'OAR' in df.columns
            else 0.0
        )
    else:
        df['Alert_OIR'] = 0.0
        df['Alert_OAR'] = 0.0

    # --- Score composite pondéré ---
    df['Score_Alerte'] = (
        0.25 * df['Alert_Rendement']
        + 0.25 * df['Alert_Volume']
        + 0.20 * df['Alert_Volatilite']
        + 0.15 * df['Alert_OIR']
        + 0.15 * df['Alert_OAR']
    ) * 100

    # --- Niveau de sévérité ---
    df['Severite'] = pd.cut(
        df['Score_Alerte'],
        bins=[-1, 20, 50, 75, 101],
        labels=['Normal', 'Faible', 'Modéré', 'Critique']
    )

    # --- Seuils P95/P99 stockés pour le dashboard ---
    df.attrs['seuils'] = {
        'rendement_p95': p95_rend, 'rendement_p99': p99_rend,
        'volume_rel_p95': p95_vol_rel, 'volume_rel_p99': p99_vol_rel,
        'volatilite_p95': p95_volat, 'volatilite_p99': p99_volat,
    }

    return df.sort_values('Score_Alerte', ascending=False).reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════
# 5. UTILITAIRES
# ═══════════════════════════════════════════════════════════════════════════

def _zscore_series(series: pd.Series) -> pd.Series:
    """Z-score robuste (sur valeurs non nulles)."""
    clean = series.dropna()
    if len(clean) < 3:
        return pd.Series(np.nan, index=series.index)
    mu, sigma = clean.mean(), clean.std()
    if sigma == 0:
        return pd.Series(0.0, index=series.index)
    return (series - mu) / sigma


def compute_beta(df_cours: pd.DataFrame, df_market: pd.DataFrame,
                 window: int = 60) -> pd.DataFrame:
    """
    Calcule le bêta rolling de chaque instrument vs le MASI.

    Bêta = Cov(Ri, Rm) / Var(Rm)  sur fenêtre glissante.
    """
    market_ret = df_market[['Jour', 'MASI_Return_pct']].set_index('Jour')
    results = []
    for ticker, grp in df_cours.groupby('Ticker'):
        grp = grp.set_index('Jour').sort_index()
        merged = grp[['Rendement_pct']].join(market_ret, how='inner')
        merged.columns = ['Ri', 'Rm']

        def rolling_beta(x):
            if x['Rm'].std() == 0:
                return np.nan
            return x['Ri'].cov(x['Rm']) / x['Rm'].var()

        betas = (merged.rolling(window, min_periods=20)
                 .apply(lambda x: x['Ri'].cov(x['Rm']) / x['Rm'].var()
                        if x['Rm'].var() > 0 else np.nan,
                        raw=False))
        betas = betas.rename(columns={'Ri': 'Beta'})['Beta']
        betas = betas.reset_index()
        betas['Ticker'] = ticker
        results.append(betas)

    if results:
        return pd.concat(results, ignore_index=True)
    return pd.DataFrame(columns=['Jour', 'Beta', 'Ticker'])


def get_top_alerts(df_alerts: pd.DataFrame, n: int = 20,
                   severite: str | None = None) -> pd.DataFrame:
    """Retourne les N premières alertes, optionnellement filtrées par sévérité."""
    df = df_alerts.copy()
    if severite:
        df = df[df['Severite'] == severite]
    return df.nlargest(n, 'Score_Alerte')


def seuils_statistiques(series: pd.Series, nom: str = '') -> dict:
    """Retourne un dictionnaire complet des seuils statistiques d'une série."""
    clean = series.dropna()
    return {
        'indicateur': nom,
        'N': len(clean),
        'moyenne': clean.mean(),
        'ecart_type': clean.std(),
        'mediane': clean.median(),
        'P01': clean.quantile(0.01),
        'P05': clean.quantile(0.05),
        'P25': clean.quantile(0.25),
        'P75': clean.quantile(0.75),
        'P95': clean.quantile(0.95),
        'P99': clean.quantile(0.99),
        'Min': clean.min(),
        'Max': clean.max(),
        'Seuil_Alerte_Bas': clean.mean() - 2 * clean.std(),
        'Seuil_Alerte_Haut': clean.mean() + 2 * clean.std(),
        'Seuil_Critique_Bas': clean.mean() - 3 * clean.std(),
        'Seuil_Critique_Haut': clean.mean() + 3 * clean.std(),
    }
