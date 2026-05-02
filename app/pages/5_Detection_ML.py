"""
Page 5 — Détection d'anomalies (ML)
Affiche les résultats des modèles Isolation Forest et Autoencoder.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st

from app.utils.bvc_theme import inject_bvc_theme

inject_bvc_theme()

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from app.utils.data_cache import load_base_data

# page_config défini dans Accueil.py

# ── Constantes couleurs ──────────────────────────────────────────────────────
BVC_BLUE = "#002366"
BVC_GOLD = "#C8A84B"
SEV_COLORS = {
    "Critique": "#F44336",
    "Modéré":   "#FF9800",
    "Faible":   "#FFC107",
    "Normal":   "#4CAF50",
}

DATA_DIR = Path(__file__).parent.parent.parent / "data"


# ── Chargement données ML ────────────────────────────────────────────────────
@st.cache_data(show_spinner="Chargement données ML...")
def load_ml_data():
    files = {
        "market_ml":     DATA_DIR / "market_ml.parquet",
        "instrument_ml": DATA_DIR / "instrument_ml.parquet",
        "orderflow_ml":  DATA_DIR / "orderflow_ml.parquet",
        "ml_anomalies":  DATA_DIR / "ml_anomalies.parquet",
    }
    result = {}
    for key, path in files.items():
        if path.exists():
            result[key] = pd.read_parquet(path)
        else:
            result[key] = pd.DataFrame()
    return result


# ══════════════════════════════════════════════════════════════════════════════
# LAYOUT PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

st.markdown(f"""
<div class="bvc-hero-banner" style="background:linear-gradient(135deg,{BVC_BLUE} 0%,#1a3a70 60%,{BVC_GOLD} 100%);
     padding:18px 24px;border-radius:10px;margin-bottom:16px;margin-top:0;">
  <h2 style="margin:0;font-size:1.5rem;color:#ffffff !important;">🤖 Détection d'anomalies (ML)</h2>
  <p style="margin:8px 0 0 0;font-size:0.95rem;line-height:1.45;color:#f1f5ff !important;">
    Isolation Forest &middot; Autoencoder &middot; Score Combiné ML
  </p>
</div>
""", unsafe_allow_html=True)

# ── Chargement ───────────────────────────────────────────────────────────────
ml_data = load_ml_data()
base    = load_base_data()

market_ml     = ml_data.get("market_ml", pd.DataFrame())
instrument_ml = ml_data.get("instrument_ml", pd.DataFrame())
orderflow_ml  = ml_data.get("orderflow_ml", pd.DataFrame())
ml_anom       = ml_data.get("ml_anomalies", pd.DataFrame())

if market_ml.empty and instrument_ml.empty:
    st.warning("""
    **Modèles ML non encore entraînés.**

    Lancez le notebook d'entraînement :
    ```
    jupyter nbconvert --to notebook --execute notebooks/04_machine_learning.ipynb
    ```
    Ou exécutez la cellule d'entraînement directement dans le notebook.
    """)
    st.stop()

# ── KPIs ML ──────────────────────────────────────────────────────────────────
st.subheader("Vue d'ensemble — Résultats ML")

col1, col2, col3, col4, col5 = st.columns(5)

nb_crit_ml = int((ml_anom["ML_Severity"] == "Critique").sum()) if not ml_anom.empty and "ML_Severity" in ml_anom.columns else 0
nb_mod_ml  = int((ml_anom["ML_Severity"] == "Modéré").sum())   if not ml_anom.empty and "ML_Severity" in ml_anom.columns else 0
nb_total   = len(ml_anom) if not ml_anom.empty else 0

models_str = []
if not instrument_ml.empty and "IF_Score"      in instrument_ml.columns: models_str.append("IF")
if not instrument_ml.empty and "AE_ReconError" in instrument_ml.columns: models_str.append("AE")
models_available = " + ".join(models_str) if models_str else "—"

col1.metric("Total Anomalies ML",  f"{nb_total}")
col2.metric("Critiques ML",        f"{nb_crit_ml}", delta_color="inverse")
col3.metric("Modérés ML",          f"{nb_mod_ml}",  delta_color="inverse")
col4.metric("Modèles actifs",      models_available)
pct_anom = f"{100*nb_total/(len(instrument_ml)+len(market_ml)+1):.1f}%" if nb_total > 0 else "0%"
col5.metric("Taux d'anomalies",    pct_anom)

st.divider()

# ── ONGLETS ───────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Scores Marché",
    "🔬 Instruments ML",
    "🆚 Stat vs ML",
    "🔥 Heatmap ML",
    "📋 Tableau Anomalies",
])

# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Scores ML — Marché Global")

    if market_ml.empty:
        st.info("Données marché ML non disponibles.")
    else:
        if "Jour" in market_ml.columns:
            market_ml["Jour"] = pd.to_datetime(market_ml["Jour"])
            market_ml = market_ml.sort_values("Jour")

        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True,
            subplot_titles=("Score ML Combiné", "Score Isolation Forest", "Erreur Reconstruction AE"),
            vertical_spacing=0.07,
        )

        # ML Score combiné
        if "ML_Score" in market_ml.columns:
            anom_m = market_ml[market_ml["ML_IsAnomaly"] == 1] if "ML_IsAnomaly" in market_ml.columns else pd.DataFrame()
            fig.add_trace(go.Scatter(
                x=market_ml["Jour"], y=market_ml["ML_Score"],
                mode="lines", line=dict(color=BVC_BLUE, width=1.5),
                name="ML Score", showlegend=True,
            ), row=1, col=1)
            if not anom_m.empty:
                fig.add_trace(go.Scatter(
                    x=anom_m["Jour"], y=anom_m["ML_Score"],
                    mode="markers", marker=dict(color="red", size=8, symbol="x"),
                    name="Anomalies ML",
                ), row=1, col=1)
            fig.add_hline(y=80, line_dash="dash", line_color="red",   annotation_text="Critique",  row=1, col=1)
            fig.add_hline(y=60, line_dash="dash", line_color="orange", annotation_text="Modéré",   row=1, col=1)

        # IF Score
        if "IF_Score" in market_ml.columns:
            fig.add_trace(go.Scatter(
                x=market_ml["Jour"], y=market_ml["IF_Score"] * 100,
                mode="lines", line=dict(color=BVC_GOLD, width=1.2),
                name="IF Score (×100)", showlegend=True,
            ), row=2, col=1)

        # AE Reconstruction Error
        if "AE_ReconError" in market_ml.columns:
            fig.add_trace(go.Scatter(
                x=market_ml["Jour"], y=market_ml["AE_ReconError"],
                mode="lines", line=dict(color="#7E57C2", width=1.2),
                name="AE Recon. Error",
            ), row=3, col=1)

        fig.update_layout(
            height=520, title_text="Scores ML — Marché Global BVC 2025",
            hovermode="x unified",
            paper_bgcolor="white", plot_bgcolor="#F8F9FA",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Sévérité distribution
        if "ML_Severity" in market_ml.columns:
            sev_counts = market_ml["ML_Severity"].value_counts()
            fig2 = go.Figure(go.Bar(
                x=sev_counts.index,
                y=sev_counts.values,
                marker_color=[SEV_COLORS.get(s, "#9E9E9E") for s in sev_counts.index],
                text=sev_counts.values, textposition="outside",
            ))
            fig2.update_layout(
                title="Distribution Sévérités ML — Marché",
                xaxis_title="Sévérité", yaxis_title="Séances",
                height=300, paper_bgcolor="white", plot_bgcolor="#F8F9FA",
            )
            st.plotly_chart(fig2, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Détection ML par Instrument")

    if instrument_ml.empty:
        st.info("Données instrument ML non disponibles.")
    else:
        tickers = sorted(instrument_ml["Ticker"].dropna().unique().tolist()) if "Ticker" in instrument_ml.columns else []

        if tickers:
            col_sel, col_info = st.columns([1, 3])
            with col_sel:
                ticker = st.selectbox("Sélectionner un instrument", tickers, key="ml_ticker")

            df_t = instrument_ml[instrument_ml["Ticker"] == ticker].copy()
            if "Jour" in df_t.columns:
                df_t["Jour"] = pd.to_datetime(df_t["Jour"])
                df_t = df_t.sort_values("Jour")

            with col_info:
                n_anom_t = int(df_t["ML_IsAnomaly"].sum()) if "ML_IsAnomaly" in df_t.columns else 0
                avg_score = df_t["ML_Score"].mean() if "ML_Score" in df_t.columns else 0
                if_score  = df_t["IF_Score"].mean() if "IF_Score" in df_t.columns else np.nan
                st.markdown(f"""
                **{ticker}** — `{n_anom_t}` anomalies ML détectées |
                Score ML moyen : `{avg_score:.1f}` |
                Score IF moyen : `{if_score:.3f}` 
                """)

            if "ML_Score" in df_t.columns and "Jour" in df_t.columns:
                fig3 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                      subplot_titles=(f"Score ML — {ticker}", "Score IF vs AE"),
                                      vertical_spacing=0.1)
                anom_t = df_t[df_t["ML_IsAnomaly"] == 1] if "ML_IsAnomaly" in df_t.columns else pd.DataFrame()

                fig3.add_trace(go.Scatter(
                    x=df_t["Jour"], y=df_t["ML_Score"],
                    mode="lines+markers", marker=dict(size=4),
                    line=dict(color=BVC_BLUE, width=1.5), name="ML Score",
                ), row=1, col=1)
                if not anom_t.empty:
                    fig3.add_trace(go.Scatter(
                        x=anom_t["Jour"], y=anom_t["ML_Score"],
                        mode="markers", marker=dict(color="red", size=10, symbol="x"),
                        name="Anomalie",
                    ), row=1, col=1)
                fig3.add_hline(y=80, line_dash="dash", line_color="red", row=1, col=1)
                fig3.add_hline(y=60, line_dash="dash", line_color="orange", row=1, col=1)

                if "IF_Score" in df_t.columns:
                    fig3.add_trace(go.Scatter(
                        x=df_t["Jour"], y=df_t["IF_Score"] * 100,
                        mode="lines", line=dict(color=BVC_GOLD, width=1.2),
                        name="IF Score (×100)",
                    ), row=2, col=1)
                if "AE_ReconError" in df_t.columns:
                    fig3.add_trace(go.Scatter(
                        x=df_t["Jour"], y=df_t["AE_ReconError"],
                        mode="lines", line=dict(color="#7E57C2", width=1.2),
                        name="AE Recon. Error",
                    ), row=2, col=1)

                fig3.update_layout(height=450, hovermode="x unified",
                                    paper_bgcolor="white", plot_bgcolor="#F8F9FA")
                st.plotly_chart(fig3, use_container_width=True)

        # Top instruments par score ML moyen
        if "Ticker" in instrument_ml.columns and "ML_Score" in instrument_ml.columns:
            st.subheader("Top 20 Instruments — Score ML Moyen")
            top_tickers = (instrument_ml.groupby("Ticker")["ML_Score"]
                           .agg(["mean", "max", "sum"])
                           .rename(columns={"mean": "Score_Moyen", "max": "Score_Max", "sum": "Score_Total"})
                           .sort_values("Score_Moyen", ascending=False).head(20).reset_index())

            fig4 = go.Figure(go.Bar(
                y=top_tickers["Ticker"], x=top_tickers["Score_Moyen"],
                orientation="h",
                marker=dict(
                    color=top_tickers["Score_Moyen"],
                    colorscale="YlOrRd", showscale=True,
                    colorbar=dict(title="Score ML Moyen"),
                ),
                text=top_tickers["Score_Moyen"].round(1), textposition="outside",
            ))
            fig4.update_layout(
                height=500, yaxis=dict(autorange="reversed"),
                xaxis_title="Score ML Moyen", yaxis_title="Ticker",
                paper_bgcolor="white", plot_bgcolor="#F8F9FA",
            )
            st.plotly_chart(fig4, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("Comparaison : Méthodes Statistiques vs Machine Learning")

    stat_anom = base.get("anomalies", pd.DataFrame())

    if stat_anom.empty or instrument_ml.empty:
        st.info("Données insuffisantes pour la comparaison.")
    else:
        from src.ml_models import compare_methods

        stat_instr = stat_anom[stat_anom["Niveau"] == "Instrument"].copy() \
            if "Niveau" in stat_anom.columns else stat_anom.copy()

        ml_instr_anom = ml_anom[ml_anom["Niveau"] == "Instrument"].copy() \
            if not ml_anom.empty and "Niveau" in ml_anom.columns else pd.DataFrame()

        cmp = compare_methods(stat_instr, ml_instr_anom)

        if cmp.empty:
            st.info("Impossible de comparer — vérifiez les colonnes Jour/Ticker.")
        else:
            stat_total  = int(cmp["Stat_Anomaly"].sum())
            ml_total    = int(cmp["ML_Anomaly"].sum())
            consensus   = int(cmp["Consensus"].sum())
            stat_only_n = stat_total - consensus
            ml_only_n   = ml_total   - consensus

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Anomalies Statistiques",  stat_total)
            c2.metric("Anomalies ML",            ml_total)
            c3.metric("Consensus (les deux)",    consensus)
            c4.metric("ML exclusif",             ml_only_n, help="Détectées par ML mais pas par stat.")

            fig5 = go.Figure()
            cats   = ["Stat. seul", "Consensus", "ML seul"]
            vals   = [stat_only_n, consensus, ml_only_n]
            colors = [BVC_BLUE, BVC_GOLD, "#2E7D32"]
            fig5.add_trace(go.Bar(x=cats, y=vals, marker_color=colors,
                                   text=vals, textposition="outside"))
            fig5.update_layout(
                title="Répartition des Anomalies par Méthode",
                yaxis_title="Nombre", height=320,
                paper_bgcolor="white", plot_bgcolor="#F8F9FA",
            )
            st.plotly_chart(fig5, use_container_width=True)

            # Scatter : score stat vs score ML
            if "ML_Score" in cmp.columns:
                st.markdown("**Distribution du Score ML pour anomalies consensus vs non-consensus**")
                cmp["Type"] = cmp.apply(
                    lambda r: "Consensus" if r["Stat_Anomaly"] == 1 and r["ML_Anomaly"] == 1
                              else ("Stat. seul" if r["Stat_Anomaly"] == 1
                              else ("ML seul" if r["ML_Anomaly"] == 1 else "Normal")), axis=1
                )
                fig6 = px.histogram(
                    cmp, x="ML_Score", color="Type",
                    color_discrete_map={"Consensus": BVC_GOLD, "Stat. seul": BVC_BLUE,
                                        "ML seul": "#2E7D32", "Normal": "#9E9E9E"},
                    barmode="overlay", opacity=0.7, nbins=40,
                    title="Distribution Score ML par Catégorie de Détection",
                )
                fig6.update_layout(height=320, paper_bgcolor="white", plot_bgcolor="#F8F9FA")
                st.plotly_chart(fig6, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.subheader("Heatmap — Intensité Anomalies ML")

    if instrument_ml.empty or "Ticker" not in instrument_ml.columns:
        st.info("Données instruments ML non disponibles.")
    else:
        if "Jour" in instrument_ml.columns:
            instrument_ml["Mois"] = pd.to_datetime(instrument_ml["Jour"]).dt.to_period("M").astype(str)

        if "ML_IsAnomaly" in instrument_ml.columns and "Mois" in instrument_ml.columns:
            pivot = (instrument_ml.groupby(["Ticker", "Mois"])["ML_IsAnomaly"]
                     .sum().unstack(fill_value=0))
            top30 = pivot.sum(axis=1).sort_values(ascending=False).head(30).index
            pivot_top = pivot.loc[top30]

            fig7 = px.imshow(
                pivot_top.values,
                x=pivot_top.columns.tolist(),
                y=pivot_top.index.tolist(),
                color_continuous_scale="YlOrRd",
                labels=dict(x="Mois", y="Ticker", color="Nb Anomalies ML"),
                title="Heatmap Anomalies ML — Top 30 Instruments",
                aspect="auto",
            )
            fig7.update_layout(height=600, paper_bgcolor="white")
            st.plotly_chart(fig7, use_container_width=True)

        # Score ML par type de sévérité, par mois
        if "ML_Severity" in instrument_ml.columns and "Mois" in instrument_ml.columns:
            pivot_sev = (instrument_ml.groupby(["Mois", "ML_Severity"])
                         .size().unstack(fill_value=0))
            fig8 = go.Figure()
            sev_order = ["Critique", "Modéré", "Faible", "Normal"]
            for sev in sev_order:
                if sev in pivot_sev.columns:
                    fig8.add_trace(go.Bar(
                        x=pivot_sev.index, y=pivot_sev[sev],
                        name=sev, marker_color=SEV_COLORS.get(sev, "#9E9E9E"),
                    ))
            fig8.update_layout(
                barmode="stack", title="Évolution Mensuelle des Sévérités ML",
                xaxis_title="Mois", yaxis_title="Nombre d'observations",
                height=350, paper_bgcolor="white", plot_bgcolor="#F8F9FA",
                legend=dict(title="Sévérité"),
            )
            st.plotly_chart(fig8, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
with tab5:
    st.subheader("Tableau Complet — Anomalies ML")

    if ml_anom.empty:
        st.info("Aucune anomalie ML disponible.")
    else:
        # Filtres
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            sev_filter = st.multiselect(
                "Sévérité", ["Critique", "Modéré", "Faible", "Normal"],
                default=["Critique", "Modéré"], key="ml_sev_filter",
            )
        with fc2:
            niv_filter = st.multiselect(
                "Niveau", ml_anom["Niveau"].unique().tolist() if "Niveau" in ml_anom.columns else [],
                default=ml_anom["Niveau"].unique().tolist()[:2] if "Niveau" in ml_anom.columns else [],
                key="ml_niv_filter",
            )
        with fc3:
            method_filter = st.multiselect(
                "Méthode ML",
                ml_anom["Methode_ML"].unique().tolist() if "Methode_ML" in ml_anom.columns else [],
                default=ml_anom["Methode_ML"].unique().tolist() if "Methode_ML" in ml_anom.columns else [],
                key="ml_method_filter",
            )

        df_filt = ml_anom.copy()
        if sev_filter and "ML_Severity" in df_filt.columns:
            df_filt = df_filt[df_filt["ML_Severity"].isin(sev_filter)]
        if niv_filter and "Niveau" in df_filt.columns:
            df_filt = df_filt[df_filt["Niveau"].isin(niv_filter)]
        if method_filter and "Methode_ML" in df_filt.columns:
            df_filt = df_filt[df_filt["Methode_ML"].isin(method_filter)]

        st.caption(f"{len(df_filt)} anomalies affichées")

        def color_sev_ml(val):
            m = {
                "Critique": "background-color:#FFEBEE;color:#C62828;font-weight:700",
                "Modéré":   "background-color:#FFF3E0;color:#E65100;font-weight:600",
                "Faible":   "background-color:#FFFDE7;color:#F57F17",
                "Normal":   "color:#388E3C",
            }
            return m.get(val, "")

        cols_show = [c for c in ["Jour", "Ticker", "Niveau", "Methode_ML",
                                  "ML_Score", "ML_Severity", "IF_Score",
                                  "IF_Severity", "AE_ReconError", "AE_Severity"]
                     if c in df_filt.columns]

        if "ML_Severity" in df_filt.columns and "ML_Score" in df_filt.columns:
            st.dataframe(
                df_filt[cols_show]
                    .sort_values("ML_Score", ascending=False)
                    .style.map(color_sev_ml, subset=["ML_Severity"]),
                use_container_width=True, height=420,
            )
        else:
            st.dataframe(df_filt[cols_show], use_container_width=True, height=420)

        # Téléchargement
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            csv = df_filt.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "⬇ Télécharger CSV", csv,
                "anomalies_ml.csv", "text/csv",
            )
        with col_dl2:
            try:
                import io
                buf = io.BytesIO()
                with pd.ExcelWriter(buf, engine="openpyxl") as w:
                    df_filt.to_excel(w, sheet_name="Anomalies_ML", index=False)
                    ml_anom.to_excel(w, sheet_name="Toutes_Anomalies_ML", index=False)
                st.download_button(
                    "⬇ Télécharger Excel", buf.getvalue(),
                    "anomalies_ml.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            except Exception:
                pass
