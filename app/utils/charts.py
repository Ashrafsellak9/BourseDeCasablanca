"""
Fonctions de visualisation réutilisables pour le dashboard BVC.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

SEV_COLORS = {
    'Normal':   '#4CAF50',
    'Faible':   '#FFC107',
    'Modéré':   '#FF9800',
    'Critique': '#F44336',
}

BVC_BLUE   = '#003087'
BVC_GOLD   = '#C8A84B'
BVC_LIGHT  = '#EAF0FB'


def kpi_card(label: str, value: str, delta: str = '', color: str = BVC_BLUE) -> str:
    """Retourne le HTML d'une carte KPI."""
    delta_html = f'<span style="font-size:12px;color:#888">{delta}</span>' if delta else ''
    return f"""
    <div style="background:{BVC_LIGHT};border-left:4px solid {color};
                border-radius:6px;padding:14px 18px;margin:4px 0;">
        <div style="font-size:13px;color:#555;font-weight:500">{label}</div>
        <div style="font-size:24px;font-weight:700;color:{color}">{value}</div>
        {delta_html}
    </div>"""


def chart_masi(df_market: pd.DataFrame) -> go.Figure:
    """Graphique MASI avec volatilité rolling."""
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.65, 0.35],
        subplot_titles=['MASI — Niveau (Points)', 'Variation Journalière (%)'],
        vertical_spacing=0.08,
    )
    fig.add_trace(go.Scatter(
        x=df_market['Jour'], y=df_market['MASI'],
        name='MASI', line=dict(color=BVC_BLUE, width=2),
        fill='tozeroy', fillcolor='rgba(0,48,135,0.06)',
    ), row=1, col=1)

    if 'MASI_Vol_20j' in df_market.columns:
        fig.add_trace(go.Scatter(
            x=df_market['Jour'], y=df_market['MASI_Vol_20j'],
            name='Volatilité 20j', line=dict(color='#E53935', width=1.2, dash='dot'),
            yaxis='y3',
        ), row=1, col=1)

    # Variation colorée
    colors = ['#E53935' if v < 0 else '#43A047'
              for v in df_market['MASI_Return_pct'].fillna(0)]
    fig.add_trace(go.Bar(
        x=df_market['Jour'], y=df_market['MASI_Return_pct'],
        name='Variation (%)', marker_color=colors, opacity=0.85,
    ), row=2, col=1)
    fig.add_hline(y=0, line_color='gray', line_width=0.8, row=2, col=1)

    fig.update_layout(
        height=450, showlegend=True,
        legend=dict(orientation='h', y=1.02),
        plot_bgcolor='white', paper_bgcolor='white',
        margin=dict(l=20, r=20, t=40, b=20),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor='#f0f0f0')
    return fig


def chart_volume_marche(df_market: pd.DataFrame) -> go.Figure:
    """Volume journalier avec moyenne mobile et seuil P99."""
    p99 = df_market['Volume_MAD'].quantile(0.99)
    mu  = df_market['Volume_MAD'].mean()

    anomaly_mask = df_market['Volume_MAD'] > p99
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_market.loc[~anomaly_mask, 'Jour'],
        y=df_market.loc[~anomaly_mask, 'Volume_MAD'] / 1e6,
        name='Volume Normal', marker_color=BVC_BLUE, opacity=0.7,
    ))
    fig.add_trace(go.Bar(
        x=df_market.loc[anomaly_mask, 'Jour'],
        y=df_market.loc[anomaly_mask, 'Volume_MAD'] / 1e6,
        name='Volume Anormal (>P99)', marker_color='#E53935', opacity=0.9,
    ))
    # Moyenne mobile 20j
    vol_ma20 = df_market['Volume_MAD'].rolling(20, min_periods=5).mean()
    fig.add_trace(go.Scatter(
        x=df_market['Jour'], y=vol_ma20 / 1e6,
        name='Moy. 20j', line=dict(color=BVC_GOLD, width=2),
    ))
    fig.add_hline(y=p99 / 1e6, line_dash='dash', line_color='#E53935',
                   annotation_text=f'P99 = {p99/1e6:.0f} M MAD')
    fig.add_hline(y=mu / 1e6, line_dash='dot', line_color='gray',
                   annotation_text=f'Moy = {mu/1e6:.0f} M MAD')
    fig.update_layout(
        title='Volume Journalier du Marché (M MAD)',
        height=350, barmode='overlay',
        plot_bgcolor='white', paper_bgcolor='white',
        legend=dict(orientation='h', y=1.02),
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


def chart_instrument_profile(df: pd.DataFrame, ticker: str) -> go.Figure:
    """Profil complet d'un instrument : cours, volume, volatilité, RSI."""
    df_t = df[df['Ticker'] == ticker].sort_values('Jour')
    libelle = df_t['Libelle'].iloc[0] if 'Libelle' in df_t.columns and len(df_t) > 0 else ticker

    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        row_heights=[0.35, 0.22, 0.22, 0.21],
        subplot_titles=[
            f'{ticker} — {libelle} | Cours de Clôture',
            'Volume Relatif (×moy 20j)',
            'Volatilité Rolling 20j (%)',
            'RSI 14 Jours',
        ],
        vertical_spacing=0.06,
    )

    # Cours
    fig.add_trace(go.Scatter(
        x=df_t['Jour'], y=df_t['Cours_Cloture'],
        name='Clôture', line=dict(color=BVC_BLUE, width=2),
        fill='tozeroy', fillcolor='rgba(0,48,135,0.05)',
    ), row=1, col=1)

    # Volume relatif
    if 'Volume_Relatif' in df_t.columns:
        colors_v = ['#E53935' if v > 2 else '#FF9800' if v > 1.5 else BVC_BLUE
                    for v in df_t['Volume_Relatif'].fillna(1)]
        fig.add_trace(go.Bar(
            x=df_t['Jour'], y=df_t['Volume_Relatif'],
            name='Vol. Relatif', marker_color=colors_v, opacity=0.85,
        ), row=2, col=1)
        fig.add_hline(y=2.0, line_dash='dash', line_color='#E53935', row=2, col=1)
        fig.add_hline(y=1.0, line_dash='dot', line_color='gray', row=2, col=1)

    # Volatilité
    if 'Volatilite_20j' in df_t.columns:
        fig.add_trace(go.Scatter(
            x=df_t['Jour'], y=df_t['Volatilite_20j'],
            name='Vol. 20j', line=dict(color='#9C27B0', width=1.5),
            fill='tozeroy', fillcolor='rgba(156,39,176,0.07)',
        ), row=3, col=1)
        p95_vol = df_t['Volatilite_20j'].quantile(0.95)
        fig.add_hline(y=p95_vol, line_dash='dash', line_color='orange',
                       annotation_text='P95', row=3, col=1)

    # RSI
    if 'RSI_14j' in df_t.columns:
        fig.add_trace(go.Scatter(
            x=df_t['Jour'], y=df_t['RSI_14j'],
            name='RSI 14j', line=dict(color='#FF6F00', width=1.5),
        ), row=4, col=1)
        fig.add_hline(y=70, line_dash='dash', line_color='#E53935', row=4, col=1)
        fig.add_hline(y=30, line_dash='dash', line_color='#43A047', row=4, col=1)
        fig.add_hrect(y0=70, y1=100, fillcolor='rgba(229,57,53,0.05)', row=4, col=1)
        fig.add_hrect(y0=0, y1=30, fillcolor='rgba(67,160,71,0.05)', row=4, col=1)
        fig.update_yaxes(range=[0, 100], row=4, col=1)

    fig.update_layout(
        height=700, showlegend=False,
        plot_bgcolor='white', paper_bgcolor='white',
        margin=dict(l=20, r=20, t=60, b=20),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor='#f0f0f0')
    return fig


def chart_anomalies_chronology(df_anom: pd.DataFrame) -> go.Figure:
    """Graphique en barres empilées : anomalies par jour et sévérité."""
    if df_anom.empty:
        return go.Figure()
    df_anom = df_anom.copy()
    df_anom['Jour'] = pd.to_datetime(df_anom['Jour'])
    daily = (df_anom[df_anom['Severite_Finale'].isin(['Modéré','Critique'])]
             .groupby(['Jour','Severite_Finale']).size().unstack(fill_value=0).reset_index())

    fig = go.Figure()
    for sev, color in [('Critique','#F44336'), ('Modéré','#FF9800')]:
        if sev in daily.columns:
            fig.add_trace(go.Bar(
                x=daily['Jour'], y=daily[sev],
                name=sev, marker_color=color, opacity=0.9,
            ))
    fig.update_layout(
        title='Chronologie des Alertes (Modéré + Critique)',
        barmode='stack', height=300,
        plot_bgcolor='white', paper_bgcolor='white',
        legend=dict(orientation='h', y=1.02),
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(showgrid=False),
        yaxis=dict(gridcolor='#f0f0f0', title='Nb alertes'),
    )
    return fig


def chart_heatmap_alerts(df_alerts: pd.DataFrame, n_tickers: int = 20) -> go.Figure:
    """Heatmap score d'alerte par (Ticker × Mois)."""
    df = df_alerts.copy()
    df['Jour'] = pd.to_datetime(df['Jour'])
    df['Mois'] = df['Jour'].dt.strftime('%b')
    df['Mois_num'] = df['Jour'].dt.month

    top_t = df.groupby('Ticker')['Score_Alerte'].mean().nlargest(n_tickers).index
    pivot = (df[df['Ticker'].isin(top_t)]
             .groupby(['Ticker','Mois_num'])['Score_Alerte'].mean()
             .unstack(fill_value=0))
    mois_labels = {1:'Jan',2:'Fév',3:'Mar',4:'Avr',5:'Mai',6:'Jun',
                   7:'Jul',8:'Aoû',9:'Sep',10:'Oct',11:'Nov',12:'Déc'}
    pivot.columns = [mois_labels.get(c, str(c)) for c in pivot.columns]

    fig = go.Figure(go.Heatmap(
        z=pivot.values, x=list(pivot.columns), y=list(pivot.index),
        colorscale='YlOrRd', zmin=0, zmax=80,
        colorbar=dict(title='Score'),
        hovertemplate='%{y} | %{x}<br>Score: %{z:.1f}<extra></extra>',
    ))
    fig.update_layout(
        title=f'Score d\'Alerte Moyen — Top {n_tickers} Instruments × Mois',
        height=max(300, n_tickers * 22 + 80),
        margin=dict(l=80, r=20, t=50, b=40),
        paper_bgcolor='white',
        xaxis=dict(side='top'),
    )
    return fig


def chart_oir_heatmap(df_of: pd.DataFrame, n_tickers: int = 20) -> go.Figure:
    """Heatmap OIR par (Ticker × Jour)."""
    top_t = df_of.groupby('Ticker')['Nb_Transactions'].sum().nlargest(n_tickers).index
    df_f = df_of[df_of['Ticker'].isin(top_t)].copy()
    df_f['Jour'] = pd.to_datetime(df_f['Jour']).dt.strftime('%d/%m')
    pivot = df_f.pivot_table(index='Ticker', columns='Jour', values='OIR', aggfunc='mean')

    fig = go.Figure(go.Heatmap(
        z=pivot.values, x=list(pivot.columns), y=list(pivot.index),
        colorscale='RdYlGn', zmid=0, zmin=-0.6, zmax=0.6,
        colorbar=dict(title='OIR'),
        hovertemplate='%{y} | %{x}<br>OIR: %{z:.3f}<extra></extra>',
    ))
    fig.update_layout(
        title=f'OIR (Lee-Ready) — Top {n_tickers} Titres',
        height=max(300, n_tickers * 22 + 80),
        margin=dict(l=80, r=20, t=50, b=60),
        paper_bgcolor='white',
        xaxis=dict(tickangle=-45),
    )
    return fig


def chart_scatter_risk(df_instr: pd.DataFrame) -> go.Figure:
    """Scatter volatilité × volume relatif moyen par instrument."""
    resume = df_instr.groupby('Ticker').agg(
        Volatilite=('Volatilite_20j','mean'),
        Volume_Rel=('Volume_Relatif','mean'),
        Turnover=('Turnover_Ratio','mean'),
        Libelle=('Libelle','first'),
    ).reset_index().dropna()

    fig = px.scatter(
        resume, x='Volume_Rel', y='Volatilite',
        size='Turnover', text='Ticker',
        color='Volatilite', color_continuous_scale='RdYlGn_r',
        hover_data=['Libelle','Turnover'],
        labels={'Volume_Rel':'Volume Relatif Moyen','Volatilite':'Volatilité Moy. 20j (%)'},
    )
    fig.update_traces(textposition='top center', textfont_size=9)
    fig.update_layout(
        title='Cartographie Risque — Volatilité × Volume par Instrument',
        height=480, showlegend=False,
        plot_bgcolor='white', paper_bgcolor='white',
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig

