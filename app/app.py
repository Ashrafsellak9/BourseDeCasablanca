"""
Système de Surveillance Intelligent — Bourse de Casablanca
Point d'entrée principal de l'application Streamlit.
"""
import streamlit as st

# set_page_config MUST be the very first Streamlit call
st.set_page_config(
    page_title="BVC Surveillance",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Imports après page_config ─────────────────────────────────────────────
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

# ── CSS BVC ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stSidebar"] { background: #002366 !important; }
    [data-testid="stSidebar"] * { color: #ffffff !important; }
    [data-testid="stSidebarNav"] a { color: #C8A84B !important; font-weight:600; }
    div[data-testid="metric-container"] {
        background: #EAF0FB; border-radius: 8px; padding: 12px;
        border-left: 4px solid #003087;
    }
    .main-header {
        background: linear-gradient(135deg, #002366 0%, #003087 60%, #C8A84B 100%);
        padding: 20px 30px; border-radius: 10px; margin-bottom: 20px;
    }
    .main-header h1 { color: white !important; margin: 0; font-size: 1.9rem; }
    .main-header p  { color: #dde3f0 !important; margin: 4px 0 0 0; }
    footer { visibility: hidden; }
    .bvc-carousel-card {
        border: 2px solid #003087; border-radius: 14px; padding: 1.1rem 1.25rem;
        background: linear-gradient(180deg, #f8fafc 0%, #eef3fb 100%);
        box-shadow: 0 4px 14px rgba(0,35,102,0.12);
        min-height: 168px;
    }
    .bvc-carousel-meta { color: #5c6b8a; font-size: 0.88rem; margin-bottom: 0.35rem; }
    .bvc-carousel-title { color: #002366; font-size: 1.15rem; font-weight: 700; margin: 0 0 0.5rem 0; }
    .bvc-carousel-ind { color: #c62828; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>&#128202; Bourse de Casablanca — Surveillance Intelligente</h1>
    <p>Analyse des flux d'ordres &middot; Détection d'anomalies &middot; Indicateurs temps réel</p>
</div>
""", unsafe_allow_html=True)

# ── Chargement données ────────────────────────────────────────────────────
from app.utils.data_cache import load_base_data
from app.utils.charts import (
    chart_masi,
    chart_volume_marche,
    chart_market_stress,
    chart_market_quality_mini,
    gauge_market_stress,
    SEV_COLORS,
)
from src.market_stress import stress_summary_latest, market_quality_trend_5d

try:
    data = load_base_data()
    df_market = data['market']
    df_instr  = data['instrument']
    df_anom   = data['anomalies']
    DATA_OK   = True
except Exception as e:
    st.error(f"Erreur chargement données : {e}")
    DATA_OK = False

if DATA_OK:
    from src.anomaly_detector import summarize_anomalies

    # ── KPIs ──────────────────────────────────────────────────────────────
    st.subheader("Vue d'ensemble — BVC 2025")
    k1,k2,k3,k4,k5,k6 = st.columns(6)

    masi_last = df_market['MASI'].iloc[-1] if 'MASI' in df_market.columns else 0
    masi_ytd  = df_market['MASI_Var_YTD_pct'].iloc[-1] if 'MASI_Var_YTD_pct' in df_market.columns else 0
    vol_last  = df_market['Volume_MAD'].iloc[-1] / 1e6
    nb_crit   = int((df_anom['Severite_Finale'] == 'Critique').sum())
    nb_mod    = int((df_anom['Severite_Finale'] == 'Modere').sum()) + int((df_anom['Severite_Finale'] == 'Modéré').sum())

    k1.metric("MASI",              f"{masi_last:,.1f}",   f"{masi_ytd:+.2f}% YTD")
    k2.metric("Séances analysées", f"{len(df_market)}")
    k3.metric("Instruments",       f"{df_instr['Ticker'].nunique()}")
    k4.metric("Dernier volume",    f"{vol_last:.0f} M MAD")
    k5.metric("Alertes Critiques", f"{nb_crit}", delta_color="inverse")
    k6.metric("Alertes Modérées",  f"{nb_mod}",  delta_color="inverse")

    # ── Market Stress Score (volatilité + breadth + OIR) — dernière séance ──
    if "Market_Stress_Score" in df_market.columns and len(df_market) > 0:
        st.subheader("⚡ Market Stress Score (global)")
        st.caption(
            "Indicateur composite 0–100 : volatilité MASI (20j), inverse du breadth, "
            "et déséquilibre moyen des flux (|OIR| moyen par séance). "
            "Mis à jour à chaque rechargement des données (dernière séance disponible)."
        )
        summ = stress_summary_latest(df_market)
        g0, g1, g2, g3, g4 = st.columns([1.35, 1, 1, 1, 2.4])

        def _fmt_m(v):
            if v is None or pd.isna(v):
                return "—"
            return f"{float(v):.1f}"

        sc = summ.get("Market_Stress_Score")
        reg = summ.get("Market_Stress_Regime", "—")
        with g0:
            if sc is not None and pd.notna(sc):
                st.plotly_chart(
                    gauge_market_stress(sc, reg),
                    use_container_width=True,
                )
            else:
                st.info("Score de stress indisponible pour la dernière séance.")
        with g1:
            st.metric("Stress vol.", _fmt_m(summ.get("Stress_Vol")))
        with g2:
            st.metric("Stress breadth", _fmt_m(summ.get("Stress_Breadth")))
        with g3:
            st.metric("Stress OIR", _fmt_m(summ.get("Stress_OIR")))
        with g4:
            if summ.get("Jour") is not None:
                st.metric("Dernière séance", str(pd.Timestamp(summ["Jour"]).date()))
            oir_m = summ.get("OIR_Marche_MeanAbs")
            st.caption(f"|OIR| moyen marché (séance) : {_fmt_m(oir_m)}")
        st.plotly_chart(
            chart_market_stress(df_market),
            use_container_width=True,
        )

        st.markdown("##### Tendance de la qualité du marché (5 séances)")
        tr = market_quality_trend_5d(df_market)
        t_left, t_right = st.columns([1.15, 2])
        with t_left:
            if tr.get("available") and tr.get("trend_delta") is not None:
                st.metric(
                    "Évolution qualité",
                    tr["trend_label"],
                    delta=f"{tr['trend_delta']:+.1f} pts (échelle 0–100)",
                    delta_color="normal",
                )
                st.caption(f"Méthode : {tr.get('method', '')}")
                if tr.get("quality_last") is not None:
                    st.caption(f"Qualité actuelle : **{tr['quality_last']:.1f}** / 100")
            else:
                st.info(tr.get("trend_label", "—"))
                if tr.get("method"):
                    st.caption(tr["method"])
        with t_right:
            st.plotly_chart(
                chart_market_quality_mini(df_market),
                use_container_width=True,
            )

        st.divider()

    # ── KPIs ML (si disponibles) ─────────────────────────────────────────
    ML_DATA_DIR = Path(__file__).parent.parent / "data"
    ml_anom_path = ML_DATA_DIR / "ml_anomalies.parquet"
    if ml_anom_path.exists():
        try:
            df_ml_anom = pd.read_parquet(ml_anom_path)
            nb_crit_ml = int((df_ml_anom['ML_Severity'] == 'Critique').sum())
            nb_mod_ml  = int((df_ml_anom['ML_Severity'] == 'Modéré').sum())
            st.caption("🤖 **Machine Learning** (Phase 5)")
            mk1, mk2, mk3, mk4 = st.columns(4)
            mk1.metric("Anomalies ML Critiques", f"{nb_crit_ml}", delta_color="inverse")
            mk2.metric("Anomalies ML Modérées",  f"{nb_mod_ml}",  delta_color="inverse")
            mk3.metric("Total anomalies ML",     f"{len(df_ml_anom)}")
            models_ok = []
            instr_ml_path = ML_DATA_DIR / "instrument_ml.parquet"
            if instr_ml_path.exists():
                df_iml = pd.read_parquet(instr_ml_path)
                if "IF_Score"      in df_iml.columns: models_ok.append("Isolation Forest")
                if "AE_ReconError" in df_iml.columns: models_ok.append("Autoencoder")
            mk4.metric("Modèles ML actifs", len(models_ok), help=" + ".join(models_ok))
        except Exception:
            pass

    st.divider()

    # ── Graphiques ────────────────────────────────────────────────────────
    col_l, col_r = st.columns([2, 1])
    with col_l:
        st.plotly_chart(chart_masi(df_market), use_container_width=True)
    with col_r:
        st.plotly_chart(chart_volume_marche(df_market), use_container_width=True)

    st.divider()

    # ── Carrousel alertes critiques + navigation ───────────────────────────
    st.subheader("Alertes critiques récentes")
    st.caption("Utilisez ◀ ▶ pour parcourir. Le bouton principal ouvre la page concernée (instrument ou marché).")

    df_crit = df_anom[df_anom['Severite_Finale'] == 'Critique'].copy()
    if 'Jour' in df_crit.columns:
        df_crit = df_crit.sort_values('Jour', ascending=False)
    df_crit = df_crit.head(30).reset_index(drop=True)

    def color_sev(val):
        m = {'Critique':'background-color:#FFEBEE;color:#C62828;font-weight:700',
             'Modéré':  'background-color:#FFF3E0;color:#E65100;font-weight:600',
             'Faible':  'background-color:#FFFDE7;color:#F57F17',
             'Normal':  'color:#388E3C'}
        return m.get(val, '')

    if len(df_crit) == 0:
        st.success("Aucune alerte critique enregistrée sur la période.")
    else:
        n_car = len(df_crit)
        if "crit_carousel_i" not in st.session_state:
            st.session_state.crit_carousel_i = 0
        if st.session_state.crit_carousel_i >= n_car:
            st.session_state.crit_carousel_i = max(0, n_car - 1)
        st.session_state.crit_carousel_i = int(st.session_state.crit_carousel_i) % n_car

        c_prev, c_card, c_next = st.columns([0.5, 5.5, 0.5])
        with c_prev:
            if st.button("◀", key="crit_car_prev", help="Alerte précédente"):
                st.session_state.crit_carousel_i = (st.session_state.crit_carousel_i - 1) % n_car
                st.rerun()
        with c_next:
            if st.button("▶", key="crit_car_next", help="Alerte suivante"):
                st.session_state.crit_carousel_i = (st.session_state.crit_carousel_i + 1) % n_car
                st.rerun()

        with c_card:
            row = df_crit.iloc[st.session_state.crit_carousel_i]
            jour_s = pd.to_datetime(row["Jour"]).strftime("%Y-%m-%d") if pd.notna(row.get("Jour")) else "—"
            tkr = row.get("Ticker", "")
            tkr = "" if pd.isna(tkr) else str(tkr).strip()
            lib = row.get("Libelle", "")
            lib = "" if pd.isna(lib) else str(lib).strip()
            niv = row.get("Niveau", "")
            niv = "" if pd.isna(niv) else str(niv).strip()
            ind = row.get("Indicateur", "")
            ind = "" if pd.isna(ind) else str(ind).strip()
            val = row.get("Valeur", "")
            sc = row.get("Score_Consensus", "")
            desc = row.get("Description", "")
            desc = "" if pd.isna(desc) else str(desc).strip()
            titre_disp = f"{tkr}" + (f" — {lib}" if lib else "")

            st.markdown(
                f'<div class="bvc-carousel-card">'
                f'<div class="bvc-carousel-meta">Critique · {jour_s} · {niv}</div>'
                f'<div class="bvc-carousel-title">{titre_disp}</div>'
                f'<div class="bvc-carousel-ind">{ind}</div>'
                f'<p style="margin:0.4rem 0 0.2rem 0;color:#333;font-size:0.95rem;">{desc[:220]}{"…" if len(desc) > 220 else ""}</p>'
                f'<p style="margin:0;color:#555;font-size:0.9rem;">Valeur : <b>{val}</b> &nbsp;·&nbsp; Consensus : <b>{sc}</b></p>'
                f'</div>',
                unsafe_allow_html=True,
            )

            st.caption(f"Carte {st.session_state.crit_carousel_i + 1} / {n_car}")

            link_cols = st.columns([1, 1, 1])
            t_up = tkr.upper() if tkr else ""
            car_key = st.session_state.crit_carousel_i
            with link_cols[0]:
                if t_up and t_up not in ("MARCHE", "MARCHÉ", "MARKET"):
                    if st.button(f"🔍 Ouvrir {tkr}", key=f"nav_instr_{car_key}_{t_up}", type="primary"):
                        st.session_state["_nav_ticker"] = tkr
                        st.switch_page("pages/2_Instruments.py")
                elif t_up in ("MARCHE", "MARCHÉ", "MARKET") or (niv and "march" in niv.lower()):
                    if st.button("📈 Vue marché global", key=f"nav_marche_{car_key}", type="primary"):
                        if jour_s != "—":
                            st.session_state["_nav_jour"] = jour_s
                        st.switch_page("pages/1_Marche_Global.py")
                else:
                    if st.button("🚨 Centre d'alertes", key=f"nav_alert_{car_key}", type="primary"):
                        st.switch_page("pages/3_Alertes.py")
            with link_cols[1]:
                if st.button("📋 Toutes les alertes", key=f"nav_all_alert_{car_key}"):
                    st.switch_page("pages/3_Alertes.py")
            with link_cols[2]:
                if st.button("📑 Liste instruments", key=f"nav_instr_list_{car_key}"):
                    st.switch_page("pages/2_Instruments.py")

    st.divider()
    st.subheader("Tableau — anomalies critiques / modérées (top 10)")
    top_anom = (df_anom[df_anom['Severite_Finale'].isin(['Critique','Modéré'])]
                .sort_values('Score_Consensus', ascending=False).head(10))

    if len(top_anom) > 0:
        cols_show = [c for c in ['Jour','Ticker','Libelle','Niveau','Indicateur',
                                  'Description','Valeur','Severite_Finale','Score_Consensus']
                     if c in top_anom.columns]
        st.dataframe(
            top_anom[cols_show].style.map(color_sev, subset=['Severite_Finale']),
            use_container_width=True, height=340,
        )
    else:
        st.success("Aucune anomalie critique ou modérée dans le top 10.")

    st.divider()

    # ── Navigation ────────────────────────────────────────────────────────
    st.subheader("Pages disponibles")
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.info("**📈 Marché Global**\nMASI, volumes, breadth, concentration")
    c2.info("**🔍 Instruments**\nProfil complet par titre")
    c3.info("**🚨 Centre d'Alertes**\nToutes les anomalies détectées")
    c4.info("**🤖 Machine Learning**\nIsolation Forest + Autoencoder")
    c5.info("**📤 Upload Excel**\nImporter un nouveau fichier")

    st.caption("Naviguez via le menu de gauche | Données : DATASET-2025.xlsx")

