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
    chart_market_stress,
    chart_market_quality_mini,
    gauge_market_stress,
    BVC_BLUE,
    BVC_GOLD,
)
from src.market_stress import stress_summary_latest, market_quality_trend_5d
from app.utils.streamlit_nav import get_query_param

data       = load_base_data()
df_market  = data['market'].copy()
df_market['Jour'] = pd.to_datetime(df_market['Jour'])

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
        )

st.divider()

# ── Graphique MASI ────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📊 MASI & Variation", "📦 Volumes", "🌐 Breadth & HHI", "⚡ Stress marché", "📋 Données Brutes"]
)

with tab1:
    col_l, col_r = st.columns([3, 1])
    with col_l:
        st.plotly_chart(chart_masi(df_f), use_container_width=True)
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
    st.plotly_chart(chart_volume_marche(df_f), use_container_width=True)

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
    st.plotly_chart(fig_m)

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
            st.plotly_chart(fig_b)
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
            st.plotly_chart(fig_h)
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
                )
        with c_ch:
            st.plotly_chart(chart_market_stress(df_f), use_container_width=True)
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
                )
    else:
        st.info("Scores de stress non disponibles (données marché ou flux d'ordres insuffisants).")

with tab5:
    display_cols = [c for c in [
        'Jour', 'MASI', 'MSI20', 'MASI_Return_pct', 'MASI_Vol_20j',
        'Volume_MAD', 'Volume_Relatif_Marche', 'Breadth_pct', 'HHI_Volume',
        'Market_Stress_Score', 'Market_Quality_Score', 'Stress_Vol', 'Stress_Breadth', 'Stress_OIR',
        'OIR_Marche_MeanAbs', 'Market_Stress_Regime',
    ] if c in df_f.columns]
    df_show = df_f[display_cols].sort_values('Jour', ascending=False).reset_index(drop=True)
    df_show['Volume_MAD'] = (df_show['Volume_MAD'] / 1e6).round(2)
    st.dataframe(df_show.rename(columns={'Volume_MAD':'Volume (M MAD)'}),
                 use_container_width=True, height=400)

    col_dl1, col_dl2 = st.columns([1,3])
    with col_dl1:
        csv = df_show.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Télécharger CSV", csv,
                           "marche_global.csv", "text/csv")

