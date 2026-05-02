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
from app.utils.charts import chart_masi, chart_volume_marche, SEV_COLORS

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

    # ── Top alertes ───────────────────────────────────────────────────────
    st.subheader("Dernières Anomalies Critiques")
    top_anom = (df_anom[df_anom['Severite_Finale'].isin(['Critique','Modéré'])]
                .sort_values('Score_Consensus', ascending=False).head(10))

    def color_sev(val):
        m = {'Critique':'background-color:#FFEBEE;color:#C62828;font-weight:700',
             'Modéré':  'background-color:#FFF3E0;color:#E65100;font-weight:600',
             'Faible':  'background-color:#FFFDE7;color:#F57F17',
             'Normal':  'color:#388E3C'}
        return m.get(val, '')

    if len(top_anom) > 0:
        cols_show = [c for c in ['Jour','Ticker','Libelle','Niveau','Indicateur',
                                  'Description','Valeur','Severite_Finale','Score_Consensus']
                     if c in top_anom.columns]
        st.dataframe(
            top_anom[cols_show].style.map(color_sev, subset=['Severite_Finale']),
            use_container_width=True, height=340,
        )
    else:
        st.success("Aucune anomalie critique.")

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

