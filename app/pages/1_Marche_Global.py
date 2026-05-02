"""
Page 1 — Vue Marché Global
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# page_config is set in app.py

st.markdown("""
<style>
    [data-testid="stSidebar"] { background: #002366; }
    [data-testid="stSidebar"] * { color: #fff !important; }
    div[data-testid="metric-container"] {
        background:#EAF0FB;border-radius:8px;padding:12px;border-left:4px solid #003087;
    }
    footer { visibility: hidden; }
</style>""", unsafe_allow_html=True)

st.title("📈 Vue Marché Global")
st.caption("Indicateurs agrégés BVC 2025 — MASI, MASI 20, Volumes, Breadth")

from app.utils.data_cache import load_base_data
from app.utils.charts import (
    chart_masi,
    chart_volume_marche,
    chart_volume_top5_concentration,
    chart_volume_by_segment_stacked,
    chart_volume_segment_share_pct,
    chart_market_stress,
    chart_market_quality_mini,
    gauge_market_stress,
    market_has_segment_volumes,
    BVC_BLUE,
    BVC_GOLD,
)
from src.market_stress import stress_summary_latest, market_quality_trend_5d
from app.utils.streamlit_nav import get_query_param

data       = load_base_data()
df_market  = data['market'].copy()
df_market['Jour'] = pd.to_datetime(df_market['Jour'])
df_instr   = data['instrument'].copy()
df_instr['Jour'] = pd.to_datetime(df_instr['Jour'])

# ── Filtres sidebar ───────────────────────────────────────────────────────
date_min = df_market['Jour'].min().date()
date_max = df_market['Jour'].max().date()
default_dates = [date_min, date_max]
jour_qp = st.session_state.pop("_nav_jour", None) or get_query_param("jour")
if jour_qp:
    try:
        jd = pd.to_datetime(jour_qp).date()
        if date_min <= jd <= date_max:
            default_dates = [jd, jd]
    except Exception:
        pass

with st.sidebar:
    st.header("🔧 Filtres")
    d_start, d_end = st.date_input(
        "Période d'analyse",
        value=default_dates,
        min_value=date_min,
        max_value=date_max,
    )
    indice_sel = st.selectbox("Indice", ["MASI", "MSI20", "Les deux"])

df_f = df_market[
    (df_market['Jour'].dt.date >= d_start) &
    (df_market['Jour'].dt.date <= d_end)
].copy()

# ── KPIs ─────────────────────────────────────────────────────────────────
st.subheader("Indicateurs Clés")
k1,k2,k3,k4,k5 = st.columns(5)
masi_last  = df_f['MASI'].iloc[-1] if 'MASI' in df_f.columns and len(df_f)>0 else 0
masi_first = df_f['MASI'].iloc[0]  if 'MASI' in df_f.columns and len(df_f)>0 else 0
masi_perf  = (masi_last - masi_first) / masi_first * 100 if masi_first != 0 else 0

vol_moy = df_f['Volume_MAD'].mean() / 1e6 if 'Volume_MAD' in df_f.columns else 0
vol_max = df_f['Volume_MAD'].max() / 1e6  if 'Volume_MAD' in df_f.columns else 0
vol_20j = df_f['MASI_Vol_20j'].iloc[-1]   if 'MASI_Vol_20j' in df_f.columns and len(df_f)>0 else 0
breadth = df_f['Breadth_pct'].mean()       if 'Breadth_pct' in df_f.columns else 0
nb_s    = len(df_f)

k1.metric("MASI (Clôture)",    f"{masi_last:,.1f}",   f"{masi_perf:+.2f}% sur période")
k2.metric("Volume Moy./Séance",f"{vol_moy:.0f} M MAD", f"Max: {vol_max:.0f} M MAD")
k3.metric("Volatilité 20j",    f"{vol_20j:.3f}%")
k4.metric("Breadth Moyen",     f"{breadth:.1f}%" if breadth else "—")
k5.metric("Séances analysées", f"{nb_s}")

if "Market_Stress_Score" in df_f.columns and len(df_f) > 0:
    summ_f = stress_summary_latest(df_f)
    sc_f = summ_f.get("Market_Stress_Score")
    if sc_f is not None and pd.notna(sc_f):
        st.metric(
            "Market Stress (fin de période affichée)",
            f"{float(sc_f):.1f} / 100",
            delta=str(summ_f.get("Market_Stress_Regime", "")),
        )

# Tendance qualité sur l'historique complet (5 séances glissantes — cohérent avec l'accueil)
if "Market_Quality_Score" in df_market.columns and len(df_market) >= 6:
    tr_full = market_quality_trend_5d(df_market)
    q1, q2 = st.columns([1, 1.8])
    with q1:
        if tr_full.get("available") and tr_full.get("trend_delta") is not None:
            st.metric(
                "Tendance qualité marché (5 séances)",
                tr_full["trend_label"],
                delta=f"{tr_full['trend_delta']:+.1f} pts",
                delta_color="normal",
            )
            st.caption(tr_full.get("method", ""))
        else:
            st.caption(tr_full.get("trend_label", "—"))
            if tr_full.get("method"):
                st.caption(tr_full["method"])
    with q2:
        st.plotly_chart(
            chart_market_quality_mini(df_market),
            use_container_width=True,
            key="mq_mini_full_market",
        )

st.divider()

# ── Graphique MASI ────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📊 MASI & Variation", "📦 Volumes", "🌐 Breadth & HHI", "⚡ Stress marché", "📋 Données Brutes"]
)

with tab1:
    col_l, col_r = st.columns([3, 1])
    with col_l:
        st.plotly_chart(chart_masi(df_f), use_container_width=True, key="chart_masi_tab1")
    with col_r:
        st.markdown("**Statistiques MASI**")
        if 'MASI_Return_pct' in df_f.columns:
            ret = df_f['MASI_Return_pct'].dropna()
            stats_df = pd.DataFrame({
                'Indicateur': ['Rendement Moy.','Écart-type','Max Journalier','Min Journalier',
                                'Jours haussiers','Jours baissiers'],
                'Valeur': [
                    f"{ret.mean():.3f}%",
                    f"{ret.std():.3f}%",
                    f"{ret.max():.3f}%",
                    f"{ret.min():.3f}%",
                    f"{(ret>0).sum()}",
                    f"{(ret<0).sum()}",
                ]
            })
            st.dataframe(stats_df, hide_index=True, use_container_width=True)

with tab2:
    st.plotly_chart(chart_volume_marche(df_f), use_container_width=True, key="chart_vol_marche_tab2")

    st.markdown(
        "**Concentration du volume (top 5)** — chaque jour : somme des volumes des **5 instruments** "
        "les plus actifs ÷ volume total de tous les titres de la feuille **Cours** (×100)."
    )
    if "Volume_Top5_Pct" in df_f.columns and len(df_f) > 0:
        last_row = df_f.sort_values("Jour").iloc[-1]
        xpct = float(last_row["Volume_Top5_Pct"]) if pd.notna(last_row.get("Volume_Top5_Pct")) else None
        c_top_a, c_top_b = st.columns([1, 2.2])
        with c_top_a:
            if xpct is not None:
                st.metric(
                    "Top 5 = part du volume",
                    f"{xpct:.1f} %",
                    help="Dernière séance de la période affichée : % du volume journalier cumulé des 5 plus gros titres.",
                )
            else:
                st.caption("Indicateur non disponible pour la dernière séance.")
        with c_top_b:
            st.plotly_chart(
                chart_volume_top5_concentration(df_f),
                use_container_width=True,
                key="chart_vol_top5_conc_tab2",
            )
        ld = pd.to_datetime(last_row["Jour"]).normalize()
        top5 = (
            df_instr.loc[pd.to_datetime(df_instr["Jour"]).dt.normalize() == ld]
            .nlargest(5, "Volume_MAD", keep="first")[
                [c for c in ["Ticker", "Libelle", "Volume_MAD"] if c in df_instr.columns]
            ]
            .copy()
        )
        if not top5.empty and "Volume_MAD" in top5.columns:
            top5["Volume (M MAD)"] = (top5["Volume_MAD"] / 1e6).round(2)
            show_cols = ["Ticker", "Libelle", "Volume (M MAD)"] if "Libelle" in top5.columns else ["Ticker", "Volume (M MAD)"]
            st.caption("Les 5 titres concernés (dernière séance de la période) :")
            st.dataframe(
                top5[show_cols].reset_index(drop=True),
                use_container_width=True,
                hide_index=True,
                height=180,
            )
    else:
        st.info("Concentration top 5 non disponible (données Cours / instruments manquantes).")

    st.markdown(
        "**Volume par segment** — agrégation des volumes de la feuille **Cours** "
        "(actions / OPCVM / obligations / autre). Classification par mots-clés dans le libellé, "
        "surcharge possible via `data/ref/ticker_segment.csv` (colonnes `Ticker`, `Segment`)."
    )
    c_seg_a, c_seg_b = st.columns(2)
    with c_seg_a:
        st.plotly_chart(
            chart_volume_by_segment_stacked(df_f),
            use_container_width=True,
            key="chart_vol_segment_stack_tab2",
        )
    with c_seg_b:
        st.plotly_chart(
            chart_volume_segment_share_pct(df_f),
            use_container_width=True,
            key="chart_vol_segment_pct_tab2",
        )
    if market_has_segment_volumes(df_f) and "Volume_Segments_vs_Marche_pct" in df_f.columns:
        last_cov = df_f.sort_values("Jour").iloc[-1]
        v = last_cov.get("Volume_Segments_vs_Marche_pct")
        if pd.notna(v):
            st.caption(
                f"Dernière séance : somme des segments = **{float(v):.1f}%** du volume global (feuille Indicateurs). "
                "Un écart est normal si certains titres ne sont pas dans la feuille Cours."
            )

    # Volume par mois
    df_f['Mois'] = df_f['Jour'].dt.month
    df_f['Mois_label'] = df_f['Jour'].dt.strftime('%b %Y')
    vol_m = df_f.groupby(['Mois','Mois_label'])['Volume_MAD'].agg(['sum','mean']).reset_index()
    vol_m.columns = ['Mois_num','Mois','Volume_Total','Volume_Moyen']
    vol_m = vol_m.sort_values('Mois_num')

    fig_m = px.bar(
        vol_m, x='Mois', y='Volume_Total',
        color='Volume_Total', color_continuous_scale='Blues',
        title='Volume Total par Mois (MAD)',
        labels={'Volume_Total':'Volume (MAD)','Mois':'Mois'},
        text_auto='.2s',
    )
    fig_m.update_layout(height=320, plot_bgcolor='white', showlegend=False,
                         margin=dict(l=20,r=20,t=40,b=20))
    st.plotly_chart(fig_m, key="chart_vol_mois_bar_tab2")

with tab3:
    col_b, col_h = st.columns(2)
    with col_b:
        if 'Breadth_pct' in df_f.columns:
            fig_b = go.Figure()
            fig_b.add_trace(go.Scatter(
                x=df_f['Jour'], y=df_f['Breadth_pct'],
                fill='tozeroy', fillcolor='rgba(67,160,71,0.1)',
                line=dict(color='#43A047', width=2), name='Breadth',
            ))
            fig_b.add_hline(y=50, line_dash='dash', line_color='gray')
            fig_b.add_hline(y=df_f['Breadth_pct'].quantile(0.05), line_dash='dot',
                              line_color='orange', annotation_text='P5')
            fig_b.add_hline(y=df_f['Breadth_pct'].quantile(0.95), line_dash='dot',
                              line_color='orange', annotation_text='P95')
            fig_b.update_layout(
                title='Breadth du Marché (% titres en hausse)',
                height=350, plot_bgcolor='white',
                yaxis=dict(range=[0,100], ticksuffix='%'),
                margin=dict(l=20,r=20,t=40,b=20),
            )
            st.plotly_chart(fig_b, key="chart_breadth_tab3")
        else:
            st.info("Breadth non disponible pour cette période.")

    with col_h:
        if 'HHI_Volume' in df_f.columns:
            fig_h = go.Figure()
            fig_h.add_trace(go.Scatter(
                x=df_f['Jour'], y=df_f['HHI_Volume'],
                line=dict(color='#9C27B0', width=2), fill='tozeroy',
                fillcolor='rgba(156,39,176,0.07)', name='HHI',
            ))
            fig_h.update_layout(
                title='HHI Concentration des Volumes',
                height=350, plot_bgcolor='white',
                margin=dict(l=20,r=20,t=40,b=20),
            )
            st.plotly_chart(fig_h, key="chart_hhi_tab3")
        else:
            st.info("HHI non disponible.")

with tab4:
    st.markdown(
        "**Market Stress Score** — composite **0–100** (volatilité MASI 20j, inverse du breadth, "
        "|OIR| moyen par séance). Chaque composante est ramenée sur [0,100] par rang percentile "
        "historique ; le score final est la **moyenne** des trois."
    )
    if "Market_Stress_Score" in df_f.columns and len(df_f) > 0:
        summ_tab = stress_summary_latest(df_f)
        sc_t = summ_tab.get("Market_Stress_Score")
        reg_t = str(summ_tab.get("Market_Stress_Regime", "—"))
        c_g, c_ch = st.columns([1, 2.5])
        with c_g:
            if sc_t is not None and pd.notna(sc_t):
                st.plotly_chart(
                    gauge_market_stress(float(sc_t), reg_t),
                    use_container_width=True,
                    key="gauge_market_stress_tab4",
                )
        with c_ch:
            st.plotly_chart(
                chart_market_stress(df_f),
                use_container_width=True,
                key="chart_market_stress_tab4",
            )
        if "Market_Quality_Score" in df_f.columns and len(df_f) >= 6:
            tr_f = market_quality_trend_5d(df_f)
            st.markdown("##### Tendance qualité (5 séances) — sur la période filtrée")
            t1, t2 = st.columns([1, 2])
            with t1:
                if tr_f.get("available") and tr_f.get("trend_delta") is not None:
                    st.metric(
                        "Évolution qualité",
                        tr_f["trend_label"],
                        delta=f"{tr_f['trend_delta']:+.1f} pts",
                        delta_color="normal",
                    )
                    st.caption(tr_f.get("method", ""))
                else:
                    st.caption(tr_f.get("trend_label", "—"))
            with t2:
                st.plotly_chart(
                    chart_market_quality_mini(df_f),
                    use_container_width=True,
                    key="mq_mini_filtered_period_tab4",
                )
    else:
        st.info("Scores de stress non disponibles (données marché ou flux d'ordres insuffisants).")

with tab5:
    display_cols = [c for c in [
        'Jour', 'MASI', 'MSI20', 'MASI_Return_pct', 'MASI_Vol_20j',
        'Volume_MAD', 'Volume_Top5_Pct', 'Z_Volume_Top5_Pct', 'Volume_MAD_Actions', 'Volume_MAD_OPCVM', 'Volume_MAD_Obligations', 'Volume_MAD_Autre',
        'Volume_Somme_Segments', 'Volume_Segments_vs_Marche_pct',
        'Volume_Relatif_Marche', 'Breadth_pct', 'HHI_Volume',
        'Market_Stress_Score', 'Market_Quality_Score', 'Stress_Vol', 'Stress_Breadth', 'Stress_OIR',
        'OIR_Marche_MeanAbs', 'Market_Stress_Regime',
    ] if c in df_f.columns]
    df_show = df_f[display_cols].sort_values('Jour', ascending=False).reset_index(drop=True)
    df_show['Volume_MAD'] = (df_show['Volume_MAD'] / 1e6).round(2)
    for _vc in (
        'Volume_MAD_Actions', 'Volume_MAD_OPCVM', 'Volume_MAD_Obligations', 'Volume_MAD_Autre',
        'Volume_Somme_Segments',
    ):
        if _vc in df_show.columns:
            df_show[_vc] = (df_show[_vc] / 1e6).round(2)
    if 'Volume_Segments_vs_Marche_pct' in df_show.columns:
        df_show['Volume_Segments_vs_Marche_pct'] = df_show['Volume_Segments_vs_Marche_pct'].round(1)
    if 'Volume_Top5_Pct' in df_show.columns:
        df_show['Volume_Top5_Pct'] = df_show['Volume_Top5_Pct'].round(1)
    if 'Z_Volume_Top5_Pct' in df_show.columns:
        df_show['Z_Volume_Top5_Pct'] = df_show['Z_Volume_Top5_Pct'].round(2)
    _rename = {'Volume_MAD': 'Volume (M MAD)', 'Volume_Top5_Pct': 'Top 5 / vol. jour %', 'Z_Volume_Top5_Pct': 'Z-score Top5 %'}
    _seg_labels = {
        'Volume_MAD_Actions': 'Vol. Actions (M MAD)',
        'Volume_MAD_OPCVM': 'Vol. OPCVM (M MAD)',
        'Volume_MAD_Obligations': 'Vol. Obligations (M MAD)',
        'Volume_MAD_Autre': 'Vol. Autre (M MAD)',
        'Volume_Somme_Segments': 'Somme segments (M MAD)',
    }
    for _vc, _lab in _seg_labels.items():
        if _vc in df_show.columns:
            _rename[_vc] = _lab
    if 'Volume_Segments_vs_Marche_pct' in df_show.columns:
        _rename['Volume_Segments_vs_Marche_pct'] = 'Segments / vol. marché %'
    st.dataframe(df_show.rename(columns=_rename),
                 use_container_width=True, height=400)

    col_dl1, col_dl2 = st.columns([1,3])
    with col_dl1:
        csv = df_show.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Télécharger CSV", csv,
                           "marche_global.csv", "text/csv")

