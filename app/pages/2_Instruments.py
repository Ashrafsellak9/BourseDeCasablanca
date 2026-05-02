"""
Page 2 — Vue par Instrument
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

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

st.title("🔍 Analyse par Instrument")
st.caption("Profil complet d'un titre : cours, volatilité, liquidité, flux d'ordres")

from app.utils.data_cache import load_base_data
from app.utils.charts import chart_instrument_profile, chart_scatter_risk, chart_oir_heatmap
from app.utils.streamlit_nav import get_query_param

data      = load_base_data()
df_instr  = data['instrument'].copy()
df_of     = data['orderflow'].copy()
df_anom   = data['anomalies'].copy()

for df in [df_instr, df_of, df_anom]:
    df['Jour'] = pd.to_datetime(df['Jour'])

# ── Sidebar : sélection instrument ───────────────────────────────────────
tickers_info = (df_instr.groupby('Ticker')
                .agg(Libelle=('Libelle','first'), Volume_Moy=('Volume_MAD','mean'))
                .sort_values('Volume_Moy', ascending=False)
                .reset_index())
options = [f"{r.Ticker} — {r.Libelle}" for _, r in tickers_info.iterrows()]

ticker_url = st.session_state.pop("_nav_ticker", None) or get_query_param("ticker")
sel_index = 0
if ticker_url:
    tu = str(ticker_url).strip().upper()
    for i, opt in enumerate(options):
        code = opt.split(" — ")[0].strip().upper()
        if code == tu:
            sel_index = i
            break

with st.sidebar:
    st.header("🔧 Sélection")
    sel = st.selectbox("Instrument", options, index=sel_index, key="sidebar_instrument_select")
    ticker = sel.split(' — ')[0].strip()
    if "Segment" in df_instr.columns:
        _seg = df_instr.loc[df_instr["Ticker"] == ticker, "Segment"].dropna()
        if len(_seg):
            st.caption(f"Segment marché : **{_seg.iloc[-1]}**")

    date_min = df_instr['Jour'].min().date()
    date_max = df_instr['Jour'].max().date()
    d_start, d_end = st.date_input(
        "Période",
        value=[date_min, date_max],
        min_value=date_min, max_value=date_max,
    )
    st.markdown("---")
    st.subheader("Indicateurs affichés")
    show_volatilite  = st.checkbox("Volatilité Rolling", value=True)
    show_volume      = st.checkbox("Volume Relatif", value=True)
    show_rsi         = st.checkbox("RSI 14j", value=True)
    show_momentum    = st.checkbox("Momentum", value=False)

# ── Filtrage ──────────────────────────────────────────────────────────────
df_t = df_instr[
    (df_instr['Ticker'] == ticker) &
    (df_instr['Jour'].dt.date >= d_start) &
    (df_instr['Jour'].dt.date <= d_end)
].sort_values('Jour').copy()

libelle = df_t['Libelle'].iloc[0] if len(df_t) > 0 and 'Libelle' in df_t.columns else ticker

st.subheader(f"{ticker} — {libelle}")

# ── KPIs instrument ───────────────────────────────────────────────────────
if len(df_t) > 0:
    k1,k2,k3,k4,k5,k6 = st.columns(6)
    cours_last  = df_t['Cours_Cloture'].iloc[-1]
    cours_first = df_t['Cours_Cloture'].iloc[0]
    perf        = (cours_last - cours_first) / cours_first * 100 if cours_first != 0 else 0
    vol_last    = df_t['Volatilite_20j'].iloc[-1] if 'Volatilite_20j' in df_t.columns else 0
    vr_last     = df_t['Volume_Relatif'].iloc[-1] if 'Volume_Relatif' in df_t.columns else 0
    rsi_last    = df_t['RSI_14j'].iloc[-1] if 'RSI_14j' in df_t.columns else 0
    spread_moy  = df_t['Spread_pct'].mean() if 'Spread_pct' in df_t.columns else 0
    turnover    = df_t['Turnover_Ratio'].mean() * 100 if 'Turnover_Ratio' in df_t.columns else 0

    delta_perf = f"{perf:+.2f}% depuis le {d_start}"
    k1.metric("Cours Clôture",    f"{cours_last:,.2f} MAD", delta_perf)
    k2.metric("Volatilité 20j",   f"{vol_last:.3f}%")
    k3.metric("Volume Relatif",   f"{vr_last:.2f}×",
              "⚠️ Spike" if vr_last > 2 else "Normal")
    k4.metric("RSI 14j",          f"{rsi_last:.1f}",
              "Surachat" if rsi_last > 70 else "Survente" if rsi_last < 30 else "Neutre")
    k5.metric("Spread Moy.",      f"{spread_moy:.3f}%")
    k6.metric("Turnover Ratio",   f"{turnover:.4f}%")

    st.divider()

# ── Graphique profil ──────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Profil Complet", "⚡ Flux d'Ordres (Intraday)",
    "🗺️ Cartographie Risque", "📋 Données"
])

with tab1:
    if len(df_t) == 0:
        st.warning("Aucune donnée pour cet instrument sur la période sélectionnée.")
    else:
        st.plotly_chart(chart_instrument_profile(df_instr, ticker), use_container_width=True)

        # Anomalies de cet instrument
        anom_t = df_anom[
            (df_anom['Ticker'] == ticker) &
            (df_anom['Severite_Finale'].isin(['Critique','Modéré']))
        ].sort_values('Score_Consensus', ascending=False)

        if len(anom_t) > 0:
            with st.expander(f"⚠️ {len(anom_t)} Anomalies détectées — {ticker}", expanded=True):
                def color_sev(val):
                    m = {'Critique':'background-color:#FFEBEE;color:#C62828;font-weight:700',
                         'Modéré':'background-color:#FFF3E0;color:#E65100;font-weight:600'}
                    return m.get(val,'')
                cols = [c for c in ['Jour','Indicateur','Description','Valeur',
                                     'Severite_Finale','Score_Consensus','Nb_Methodes_Alerte']
                        if c in anom_t.columns]
                st.dataframe(
                    anom_t[cols].style.map(color_sev, subset=['Severite_Finale']),
                    use_container_width=True, height=280
                )

with tab2:
    df_of_t = df_of[df_of['Ticker'] == ticker].sort_values('Jour')
    if len(df_of_t) == 0:
        st.info(f"Données intraday non disponibles pour {ticker} sur la période sélectionnée.\n\n_(Les données Intraday couvrent Décembre 2025)_")
    else:
        col_a, col_b = st.columns(2)

        with col_a:
            # OIR journalier
            fig_oir = go.Figure()
            oir_colors = ['#E53935' if v < -0.2 else '#43A047' if v > 0.2 else '#9E9E9E'
                          for v in df_of_t['OIR'].fillna(0)]
            fig_oir.add_trace(go.Bar(
                x=df_of_t['Jour'], y=df_of_t['OIR'],
                marker_color=oir_colors, name='OIR', opacity=0.85,
            ))
            fig_oir.add_hline(y=0, line_color='black', line_width=0.8)
            p99_oir = df_of['OIR'].abs().quantile(0.99)
            fig_oir.add_hline(y=p99_oir, line_dash='dash', line_color='#E53935',
                               annotation_text='P99')
            fig_oir.add_hline(y=-p99_oir, line_dash='dash', line_color='#E53935')
            fig_oir.update_layout(
                title=f'OIR (Lee-Ready) — {ticker}',
                height=320, plot_bgcolor='white',
                yaxis=dict(title='OIR', zeroline=True),
                margin=dict(l=20,r=20,t=40,b=20),
            )
            st.plotly_chart(fig_oir)

        with col_b:
            # OAR journalier
            fig_oar = go.Figure()
            fig_oar.add_trace(go.Scatter(
                x=df_of_t['Jour'], y=df_of_t['OAR'],
                line=dict(color='#1565C0', width=2), fill='tozeroy',
                fillcolor='rgba(21,101,192,0.08)', name='OAR',
            ))
            p95_oar = df_of['OAR'].quantile(0.95)
            fig_oar.add_hline(y=p95_oar, line_dash='dash', line_color='orange',
                               annotation_text='P95')
            fig_oar.update_layout(
                title=f'Order Arrival Rate — {ticker}',
                height=320, plot_bgcolor='white',
                yaxis=dict(title='Transactions/h'),
                margin=dict(l=20,r=20,t=40,b=20),
            )
            st.plotly_chart(fig_oar)

        # Volatilité intraday
        col_c, col_d = st.columns(2)
        with col_c:
            fig_vi = go.Figure()
            fig_vi.add_trace(go.Bar(
                x=df_of_t['Jour'], y=df_of_t['Volatilite_Intraday'],
                marker_color='#9C27B0', opacity=0.8, name='Vol. Intraday',
            ))
            fig_vi.update_layout(
                title=f'Volatilité Intraday (%) — {ticker}',
                height=280, plot_bgcolor='white',
                margin=dict(l=20,r=20,t=40,b=20),
            )
            st.plotly_chart(fig_vi)

        with col_d:
            if 'VWAP_Dev_pct' in df_of_t.columns:
                fig_vw = go.Figure()
                fig_vw.add_trace(go.Bar(
                    x=df_of_t['Jour'], y=df_of_t['VWAP_Dev_pct'],
                    marker_color=['#E53935' if v > 0 else '#43A047'
                                  for v in df_of_t['VWAP_Dev_pct'].fillna(0)],
                    opacity=0.85, name='VWAP Dev',
                ))
                fig_vw.add_hline(y=0, line_color='black', line_width=0.8)
                fig_vw.update_layout(
                    title=f'Déviation VWAP vs Clôture — {ticker}',
                    height=280, plot_bgcolor='white',
                    margin=dict(l=20,r=20,t=40,b=20),
                )
                st.plotly_chart(fig_vw)

with tab3:
    st.plotly_chart(chart_scatter_risk(df_instr), use_container_width=True)
    st.caption("La taille des bulles = Turnover Ratio | Couleur = Volatilité (rouge = plus volatile)")

with tab4:
    cols_data = [c for c in ['Jour','Cours_Ref','Cours_Cloture','Rendement_pct',
                               'Volatilite_20j','Volume_MAD','Volume_Relatif',
                               'Spread_pct','Turnover_Ratio','RSI_14j','Momentum_20j']
                 if c in df_t.columns]
    st.dataframe(df_t[cols_data].sort_values('Jour', ascending=False).reset_index(drop=True),
                 use_container_width=True, height=400)
    csv = df_t[cols_data].to_csv(index=False).encode('utf-8')
    st.download_button(f"⬇️ Télécharger {ticker}.csv", csv, f"{ticker}_indicateurs.csv", "text/csv")

