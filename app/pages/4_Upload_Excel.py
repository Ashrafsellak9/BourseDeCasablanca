"""
Page 4 — Upload Excel + Pipeline Automatique
Permet d'importer un nouveau fichier du même format que DATASET-2025.xlsx
et calcule automatiquement tous les indicateurs + anomalies.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import pandas as pd
import io

# page_config is set in app.py
st.markdown("""
<style>
    [data-testid="stSidebar"] { background: #002366; }
    [data-testid="stSidebar"] * { color: #fff !important; }
    div[data-testid="metric-container"] {
        background:#EAF0FB;border-radius:8px;padding:12px;border-left:4px solid #003087;
    }
    .upload-zone {
        border: 2px dashed #C8A84B; border-radius: 12px;
        padding: 30px; text-align: center; background: #FAFBFF;
        margin: 20px 0;
    }
    footer { visibility: hidden; }
</style>""", unsafe_allow_html=True)

st.title("📤 Import & Analyse Automatique")
st.caption("Importez un nouveau fichier Excel au même format que DATASET-2025 — calcul automatique de tous les indicateurs")

# ── Instructions format ───────────────────────────────────────────────────
with st.expander("📋 Format attendu du fichier Excel", expanded=False):
    st.markdown("""
    Le fichier doit contenir **4 feuilles** avec les colonnes suivantes :

    | Feuille | Colonnes obligatoires |
    |---|---|
    | **Indicateurs** | `Jour`, `Volume`, `Quantite Titre`, `Nb Contrat` |
    | **Indices** | `Jour`, `Code Indice`, `Variation Veille`, `Indice Ph J`, `Indice Pb J` |
    | **Cours** | `Jour`, `Ticker`, `Cours Clôture`, `Volume`, `Capitalisation` |
    | **Intraday** | `Jour`, `Ticker`, `Sens` (A/V), `Cours Transaction`, `Quantite Titre` |

    > ℹ️ Les noms de colonnes doivent être identiques au fichier DATASET-2025.xlsx.
    > La feuille Intraday est optionnelle (les indicateurs OIR/OAR ne seront pas calculés si absente).
    """)

# ── Upload ────────────────────────────────────────────────────────────────
st.markdown('<div class="upload-zone">', unsafe_allow_html=True)
uploaded = st.file_uploader(
    "Déposez votre fichier Excel ici (.xlsx)",
    type=["xlsx"],
    help="Fichier au même format que DATASET-2025.xlsx",
)
st.markdown('</div>', unsafe_allow_html=True)

if uploaded is None:
    st.info("⬆️ Uploadez un fichier Excel pour démarrer l'analyse automatique.")

    st.markdown("---")
    st.subheader("Exemple : Données 2025 (référence)")
    from app.utils.data_cache import load_base_data
    from src.data_loader import get_summary

    data_ref = load_base_data()
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Séances (Marché)", f"{len(data_ref['market'])}")
    c2.metric("Lignes Instruments", f"{len(data_ref['instrument']):,}")
    c3.metric("Transactions Intraday", f"{len(data_ref['orderflow']):,}")
    c4.metric("Anomalies détectées", f"{len(data_ref['anomalies']):,}")

else:
    # ── Pipeline de traitement ────────────────────────────────────────────
    from app.utils.data_cache import process_uploaded_file
    from app.utils.charts import (
        chart_masi, chart_volume_marche, chart_instrument_profile,
        chart_anomalies_chronology, chart_heatmap_alerts, SEV_COLORS
    )
    from src.anomaly_detector import summarize_anomalies

    file_bytes = uploaded.read()
    filename   = uploaded.name

    with st.spinner(f"⚙️ Traitement de **{filename}**..."):
        result = process_uploaded_file(file_bytes, filename)

    if 'error' in result:
        st.error(f"❌ Erreur : {result['error']}")
        st.stop()

    st.success(f"✅ Fichier **{filename}** traité avec succès !")

    # ── Résumé du fichier ─────────────────────────────────────────────────
    st.subheader("📊 Résumé du Fichier Importé")

    if 'summary' in result and result['summary'] is not None:
        df_summary = result['summary']
        st.dataframe(df_summary, use_container_width=True, hide_index=True)

    st.divider()

    # ── KPIs ──────────────────────────────────────────────────────────────
    df_market  = result.get('market', pd.DataFrame())
    df_instr   = result.get('instrument', pd.DataFrame())
    df_of      = result.get('orderflow', pd.DataFrame())
    df_alerts  = result.get('alerts', pd.DataFrame())
    df_anom    = result.get('anomalies', pd.DataFrame())

    for df in [df_market, df_instr, df_of, df_alerts, df_anom]:
        if not df.empty and 'Jour' in df.columns:
            df['Jour'] = pd.to_datetime(df['Jour'])

    kpis = summarize_anomalies(df_anom) if not df_anom.empty else {}
    k1,k2,k3,k4,k5 = st.columns(5)
    k1.metric("Séances",         f"{len(df_market)}")
    k2.metric("Instruments",     f"{df_instr['Ticker'].nunique() if not df_instr.empty else 0}")
    k3.metric("🔴 Critiques",    f"{kpis.get('Critique',0)}")
    k4.metric("🟠 Modérées",     f"{kpis.get('Modéré',0)}")
    k5.metric("Titres alertés",  f"{kpis.get('Instruments_Alertés',0)}")

    st.divider()

    # ── Onglets résultats ─────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Marché Global", "🔍 Instruments", "🚨 Anomalies", "⬇️ Exports"
    ])

    with tab1:
        if df_market.empty:
            st.warning("Feuille Indicateurs ou Indices manquante.")
        else:
            col_l, col_r = st.columns([2,1])
            with col_l:
                if 'MASI' in df_market.columns:
                    st.plotly_chart(chart_masi(df_market), use_container_width=True)
                else:
                    st.info("MASI non disponible.")
            with col_r:
                st.plotly_chart(chart_volume_marche(df_market), use_container_width=True)

            # Statistiques descriptives
            num_cols = df_market.select_dtypes(include='number').columns.tolist()
            if num_cols:
                with st.expander("Statistiques descriptives — Marché Global"):
                    st.dataframe(df_market[num_cols].describe().T.round(4),
                                 use_container_width=True)

    with tab2:
        if df_instr.empty:
            st.warning("Feuille Cours manquante ou vide.")
        else:
            tickers = sorted(df_instr['Ticker'].unique().tolist())
            sel_t   = st.selectbox("Choisir un instrument", tickers)
            if sel_t:
                st.plotly_chart(chart_instrument_profile(df_instr, sel_t),
                                use_container_width=True)

            # Tableau de synthèse
            with st.expander("Résumé par Instrument"):
                resume = df_instr.groupby('Ticker').agg(
                    Séances=('Jour','count'),
                    Cours_Moyen=('Cours_Cloture','mean'),
                    Volatilite_Moy=('Volatilite_20j','mean'),
                    Volume_Relatif_Moy=('Volume_Relatif','mean'),
                    Turnover_Moy=('Turnover_Ratio','mean'),
                    RSI_Moyen=('RSI_14j','mean'),
                ).round(4).reset_index()
                st.dataframe(resume, use_container_width=True, height=350)

    with tab3:
        if df_anom.empty:
            st.success("✅ Aucune anomalie détectée.")
        else:
            st.plotly_chart(chart_anomalies_chronology(df_anom), use_container_width=True)

            col_a, col_b = st.columns(2)
            with col_a:
                sev_counts = df_anom['Severite_Finale'].value_counts().reset_index()
                sev_counts.columns = ['Sévérité','Nb']
                import plotly.graph_objects as go
                fig_pie = go.Figure(go.Pie(
                    labels=sev_counts['Sévérité'],
                    values=sev_counts['Nb'],
                    marker_colors=[SEV_COLORS.get(s,'gray') for s in sev_counts['Sévérité']],
                    hole=0.4,
                ))
                fig_pie.update_layout(title='Répartition Sévérité', height=280,
                                       margin=dict(l=20,r=20,t=40,b=20))
                st.plotly_chart(fig_pie)

            with col_b:
                if not df_alerts.empty:
                    st.plotly_chart(chart_heatmap_alerts(df_alerts, 15), use_container_width=True)

            # Tableau anomalies
            def color_sev(val):
                m = {'Critique':'background-color:#FFEBEE;color:#C62828;font-weight:700',
                     'Modéré':'background-color:#FFF3E0;color:#E65100;font-weight:600'}
                return m.get(val,'')

            cols = [c for c in ['Jour','Ticker','Libelle','Niveau','Indicateur',
                                  'Valeur','Severite_Finale','Score_Consensus']
                    if c in df_anom.columns]
            top_anom = df_anom[df_anom['Severite_Finale'].isin(['Critique','Modéré'])].head(50)
            st.dataframe(
                top_anom[cols].style.map(color_sev, subset=['Severite_Finale']),
                use_container_width=True, height=400,
            )

    with tab4:
        st.subheader("Télécharger les Résultats")
        col_d1, col_d2, col_d3 = st.columns(3)

        with col_d1:
            if not df_market.empty:
                csv_m = df_market.to_csv(index=False).encode('utf-8')
                st.download_button("📈 Marché Global (CSV)", csv_m,
                                   f"marche_{filename}.csv", "text/csv")

        with col_d2:
            if not df_instr.empty:
                csv_i = df_instr.to_csv(index=False).encode('utf-8')
                st.download_button("🔍 Indicateurs Instruments (CSV)", csv_i,
                                   f"instruments_{filename}.csv", "text/csv")

        with col_d3:
            if not df_anom.empty:
                csv_a = df_anom.to_csv(index=False).encode('utf-8')
                st.download_button("🚨 Anomalies (CSV)", csv_a,
                                   f"anomalies_{filename}.csv", "text/csv")

        # Export Excel complet
        if not df_anom.empty:
            st.markdown("---")
            output_xl = io.BytesIO()
            with pd.ExcelWriter(output_xl, engine='openpyxl') as writer:
                if not df_market.empty:
                    df_market.to_excel(writer, sheet_name='Marché Global', index=False)
                if not df_instr.empty:
                    df_instr.to_excel(writer, sheet_name='Instruments', index=False)
                if not df_of.empty:
                    df_of.to_excel(writer, sheet_name='Order Flow', index=False)
                if not df_anom.empty:
                    df_anom.to_excel(writer, sheet_name='Anomalies', index=False)

            st.download_button(
                "📊 Télécharger Rapport Complet (Excel)",
                data=output_xl.getvalue(),
                file_name=f"rapport_surveillance_{filename}",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )

