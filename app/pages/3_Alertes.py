"""
Page 3 — Centre d'Alertes
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st

from app.utils.bvc_theme import inject_bvc_theme

inject_bvc_theme()

import pandas as pd
import numpy as np
import plotly.graph_objects as go

# page_config défini dans Accueil.py

st.title("🚨 Centre d'Alertes")
st.caption("Toutes les anomalies détectées — BVC 2025 | 6 méthodes statistiques | 3 niveaux")

from app.utils.data_cache import load_base_data
from src.critical_alert_notifier import (
    notify_new_critical_alerts,
    notification_status_summary,
)
from app.utils.charts import (
    chart_anomalies_chronology, chart_heatmap_alerts,
    chart_oir_heatmap, SEV_COLORS
)
from src.anomaly_detector import summarize_anomalies

data      = load_base_data()
df_anom   = data['anomalies'].copy()
df_alerts = data['alerts'].copy()
df_of     = data['orderflow'].copy()
df_seuils = data['seuils'].copy()

for df in [df_anom, df_alerts, df_of]:
    df['Jour'] = pd.to_datetime(df['Jour'])

_auto_env = os.environ.get("BVC_CRITICAL_ALERT_AUTO", "").lower() in ("1", "true", "yes")
if _auto_env and "bvc_critical_notify_auto_done" not in st.session_state:
    st.session_state["bvc_critical_notify_auto_done"] = True
    _rep_auto = notify_new_critical_alerts(df_anom, df_alerts)
    if _rep_auto.get("ok"):
        st.toast(
            f"Notification envoyée ({_rep_auto.get('new_count', 0)} alerte(s)) — {_rep_auto.get('ref_jour', '')}",
            icon="🔔",
        )
    elif _rep_auto.get("errors"):
        st.session_state["_bvc_auto_push_errors"] = _rep_auto["errors"]
        st.toast("Notification push : erreur (voir bas du menu latéral)", icon="⚠️")

# ── Sidebar filtres ───────────────────────────────────────────────────────
with st.sidebar:
    st.header("🔧 Filtres")

    sev_options = ['Tous','Critique','Modéré','Faible']
    sev_sel     = st.selectbox("Sévérité", sev_options)

    niv_options = ['Tous'] + sorted(df_anom['Niveau'].dropna().unique().tolist())
    niv_sel     = st.selectbox("Niveau", niv_options)

    ind_options = ['Tous'] + sorted(df_anom['Indicateur'].dropna().unique().tolist())
    ind_sel     = st.selectbox("Indicateur", ind_options)

    all_tickers = sorted(df_anom['Ticker'].dropna().unique().tolist())
    ticker_sel  = st.multiselect("Ticker(s)", all_tickers, default=[])

    date_min = df_anom['Jour'].min().date()
    date_max = df_anom['Jour'].max().date()
    d_start, d_end = st.date_input(
        "Période",
        value=[date_min, date_max],
        min_value=date_min, max_value=date_max,
    )

    _push_errs = st.session_state.pop("_bvc_auto_push_errors", None)
    if _push_errs:
        st.divider()
        st.markdown("**Push auto**")
        for _e in _push_errs:
            st.caption(str(_e))

# ── Filtrage ──────────────────────────────────────────────────────────────
df_f = df_anom[
    (df_anom['Jour'].dt.date >= d_start) &
    (df_anom['Jour'].dt.date <= d_end)
].copy()

if sev_sel != 'Tous':
    df_f = df_f[df_f['Severite_Finale'] == sev_sel]
if niv_sel != 'Tous':
    df_f = df_f[df_f['Niveau'] == niv_sel]
if ind_sel != 'Tous':
    df_f = df_f[df_f['Indicateur'] == ind_sel]
if ticker_sel:
    df_f = df_f[df_f['Ticker'].isin(ticker_sel)]

# ── KPIs ──────────────────────────────────────────────────────────────────
kpis = summarize_anomalies(df_f)
k1,k2,k3,k4,k5 = st.columns(5)
k1.metric("Total Anomalies",    f"{kpis['Total']:,}")
k2.metric("🔴 Critiques",       f"{kpis['Critique']:,}")
k3.metric("🟠 Modérées",        f"{kpis['Modéré']:,}")
k4.metric("Instruments alertés",f"{kpis['Instruments_Alertés']}")
k5.metric("Jours avec alerte",  f"{kpis['Jours_avec_Alerte']}")

with st.expander("🔔 Notifications push (e-mail / Slack)", expanded=False):
    st.markdown(
        "Envoie les lignes **Critique** pour la **dernière séance** présente dans les données "
        "(anomalies statistiques + scores d’alerte instruments). Les envois déjà effectués ne sont "
        "pas dupliqués (fichier d’empreintes sur disque)."
    )
    _summ = notification_status_summary()
    u1, u2, u3 = st.columns(3)
    u1.metric("E-mail configuré", "Oui" if _summ["email_configured"] else "Non")
    u2.metric("Slack configuré", "Oui" if _summ["slack_configured"] else "Non")
    u3.metric("Au moins un canal", "Oui" if (_summ["email_configured"] or _summ["slack_configured"]) else "Non")
    st.caption(
        "Variables : `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASSWORD`, `SMTP_USE_TLS`, "
        "`ALERT_EMAIL_FROM`, `ALERT_EMAIL_TO`, `SLACK_WEBHOOK_URL`. "
        "Auto au chargement de cette page : `BVC_CRITICAL_ALERT_AUTO=true`."
    )
    st.code(_summ["state_path"], language="text")

    if st.button("Envoyer les nouvelles alertes critiques maintenant", type="primary"):
        with st.spinner("Envoi en cours…"):
            _rep = notify_new_critical_alerts(df_anom, df_alerts)
        if _rep.get("ok"):
            st.success(
                f"Envoi réussi — {_rep.get('new_count', 0)} nouvelle(s) alerte(s) "
                f"({_rep.get('ref_jour', '')}). E-mail: {_rep.get('email_sent')}, Slack: {_rep.get('slack_sent')}."
            )
        elif _rep.get("skipped"):
            st.info(_rep.get("reason", "—"))
        else:
            st.warning(_rep.get("reason", "Échec ou configuration incomplète."))
        if _rep.get("errors"):
            for err in _rep["errors"]:
                st.error(err)

st.divider()

# ── Onglets ───────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📅 Chronologie", "🗺️ Heatmaps", "⚡ OIR & Flux", "📊 Seuils Statistiques", "📋 Tableau Complet"
])

with tab1:
    st.plotly_chart(chart_anomalies_chronology(df_anom), use_container_width=True)

    # Répartition par indicateur
    col_a, col_b = st.columns(2)
    with col_a:
        sev_counts = df_f['Severite_Finale'].value_counts().reset_index()
        sev_counts.columns = ['Sévérité', 'Nb']
        fig_pie = go.Figure(go.Pie(
            labels=sev_counts['Sévérité'],
            values=sev_counts['Nb'],
            marker_colors=[SEV_COLORS.get(s,'gray') for s in sev_counts['Sévérité']],
            hole=0.4,
        ))
        fig_pie.update_layout(title='Répartition par Sévérité', height=300,
                               margin=dict(l=20,r=20,t=40,b=20))
        st.plotly_chart(fig_pie)

    with col_b:
        ind_counts = (df_f[df_f['Severite_Finale'].isin(['Critique','Modéré'])]
                      .groupby('Indicateur').size().nlargest(10).reset_index(name='Nb'))
        fig_ind = go.Figure(go.Bar(
            y=ind_counts['Indicateur'][::-1], x=ind_counts['Nb'][::-1],
            orientation='h', marker_color='#E53935', opacity=0.85,
        ))
        fig_ind.update_layout(title='Top Indicateurs Déclencheurs', height=300,
                               margin=dict(l=20,r=20,t=40,b=20), plot_bgcolor='white')
        st.plotly_chart(fig_ind)

with tab2:
    col_l, col_r = st.columns(2)
    with col_l:
        n_t = st.slider("Nb titres (heatmap alertes)", 10, 40, 20)
        st.plotly_chart(chart_heatmap_alerts(df_alerts, n_t), use_container_width=True)
    with col_r:
        n_t2 = st.slider("Nb titres (heatmap OIR)", 10, 30, 20)
        if not df_of.empty:
            st.plotly_chart(chart_oir_heatmap(df_of, n_t2), use_container_width=True)

with tab3:
    if df_of.empty:
        st.info("Données order flow non disponibles.")
    else:
        col_oir, col_oar = st.columns(2)
        with col_oir:
            oir_dist = df_of['OIR'].dropna()
            p99 = oir_dist.abs().quantile(0.99)
            fig_oir = go.Figure()
            fig_oir.add_trace(go.Histogram(
                x=oir_dist, nbinsx=60, name='OIR',
                marker_color='#1565C0', opacity=0.75,
            ))
            fig_oir.add_vline(x=oir_dist.mean(), line_dash='dash', line_color='orange',
                               annotation_text=f'Moy={oir_dist.mean():.3f}')
            fig_oir.add_vline(x=p99, line_dash='dot', line_color='red',
                               annotation_text='P99')
            fig_oir.add_vline(x=-p99, line_dash='dot', line_color='red')
            fig_oir.update_layout(
                title='Distribution OIR (tous instruments)',
                height=350, plot_bgcolor='white',
                margin=dict(l=20,r=20,t=40,b=20),
            )
            st.plotly_chart(fig_oir)

        with col_oar:
            oar_dist = df_of['OAR'].dropna()
            fig_oar = go.Figure()
            fig_oar.add_trace(go.Histogram(
                x=oar_dist, nbinsx=50, name='OAR',
                marker_color='#2E7D32', opacity=0.75,
            ))
            fig_oar.add_vline(x=oar_dist.quantile(0.95), line_dash='dash',
                               line_color='orange', annotation_text='P95')
            fig_oar.add_vline(x=oar_dist.quantile(0.99), line_dash='dot',
                               line_color='red', annotation_text='P99')
            fig_oar.update_layout(
                title='Distribution OAR (transactions/h)',
                height=350, plot_bgcolor='white',
                margin=dict(l=20,r=20,t=40,b=20),
            )
            st.plotly_chart(fig_oar)

        # Top OIR extrêmes
        st.markdown("**Top 15 — OIR les Plus Extrêmes (Décembre 2025)**")
        top_oir = (df_of.reindex(df_of['OIR'].abs().sort_values(ascending=False).index)
                   .head(15)[['Jour','Ticker','OIR','OAR','Volatilite_Intraday','Nb_Transactions']])
        st.dataframe(top_oir, use_container_width=True, hide_index=True)

with tab4:
    if df_seuils.empty:
        st.info("Seuils non disponibles.")
    else:
        niveau_sel = st.selectbox("Filtrer par niveau", ['Tous'] + df_seuils['Niveau'].unique().tolist())
        df_s = df_seuils if niveau_sel == 'Tous' else df_seuils[df_seuils['Niveau'] == niveau_sel]

        def highlight_thresholds(df):
            return df.style.background_gradient(
                subset=[c for c in ['P99','Seuil_Critique_Haut'] if c in df.columns],
                cmap='Reds'
            ).background_gradient(
                subset=[c for c in ['P1','Seuil_Critique_Bas'] if c in df.columns],
                cmap='Blues_r'
            )

        st.dataframe(highlight_thresholds(df_s), use_container_width=True, height=400)
        csv_s = df_s.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Télécharger Seuils CSV", csv_s, "seuils_statistiques.csv", "text/csv")

with tab5:
    n_total = len(df_f)
    st.markdown(f"**{n_total:,} anomalies** après filtrage")

    cols_show = [c for c in ['Jour','Ticker','Libelle','Niveau','Indicateur','Description',
                             'Valeur','Severite_Finale','Score_Consensus','Nb_Methodes_Alerte']
                 if c in df_f.columns]
    df_all = df_f[cols_show].reset_index(drop=True)

    # Le style pandas + 10–20k lignes = HTML énorme → navigateur très lent.
    _max_ui = min(n_total, 10_000)
    _default_ui = min(750, _max_ui) if n_total else 1
    n_display = st.number_input(
        "Lignes affichées dans le tableau (aperçu)",
        min_value=1,
        max_value=_max_ui if _max_ui else 1,
        value=_default_ui if _max_ui else 1,
        step=1,
        help="Un extrait seulement pour un rendu fluide. "
        "Exports CSV / Excel = **toutes** les lignes filtrées.",
        disabled=n_total == 0,
    )
    if n_total:
        st.caption(
            f"Aperçu : **{min(n_display, n_total):,}** / **{n_total:,}** lignes — "
            "couleurs de sévérité uniquement si l’aperçu ≤ 1 000 lignes."
        )

    def color_sev(val):
        m = {'Critique':'background-color:#FFEBEE;color:#C62828;font-weight:700',
             'Modéré':'background-color:#FFF3E0;color:#E65100;font-weight:600',
             'Faible':'background-color:#FFFDE7;color:#F57F17',
             'Normal':'color:#388E3C'}
        return m.get(val,'')

    if n_total == 0:
        st.info("Aucune anomalie pour les filtres choisis.")
    else:
        df_table = df_all.head(n_display)
        if len(df_table) <= 1000 and 'Severite_Finale' in df_table.columns:
            st.dataframe(
                df_table.style.map(color_sev, subset=['Severite_Finale']),
                use_container_width=True,
                height=520,
                hide_index=True,
            )
        else:
            st.dataframe(df_table, use_container_width=True, height=520, hide_index=True)

    col1, col2 = st.columns([1, 3])
    with col1:
        csv = df_all.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Exporter CSV (tout le filtre)", csv, "anomalies_filtrees.csv", "text/csv")

    with col2:
        import io
        output = io.BytesIO()
        df_all.to_excel(output, index=False, engine='openpyxl')
        st.download_button(
            "⬇️ Exporter Excel (tout le filtre)",
            data=output.getvalue(),
            file_name="anomalies_filtrees.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

