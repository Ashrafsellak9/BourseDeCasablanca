"""
Fonctions de visualisation réutilisables pour le dashboard BVC.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from src.instrument_segment import SEGMENT_VOLUME_COLUMNS

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


def chart_volume_top5_concentration(df_market: pd.DataFrame) -> go.Figure:
    """Évolution de la part du volume journalier détenue par les 5 instruments les plus actifs (%)."""
    if "Volume_Top5_Pct" not in df_market.columns:
        fig = go.Figure()
        fig.add_annotation(
            text="Indicateur Volume_Top5_Pct indisponible (régénérer les Parquet ou recharger les données).",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font=dict(size=13),
        )
        fig.update_layout(height=300, plot_bgcolor="white")
        return fig

    dm = df_market.copy()
    dm["Jour"] = pd.to_datetime(dm["Jour"])
    y = dm["Volume_Top5_Pct"].clip(lower=0, upper=100)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dm["Jour"],
        y=y,
        mode="lines",
        name="Top 5",
        line=dict(color=BVC_GOLD, width=2.2),
        fill="tozeroy",
        fillcolor="rgba(200,168,75,0.14)",
    ))
    mu = float(y.mean()) if y.notna().any() else np.nan
    if pd.notna(mu):
        fig.add_hline(
            y=mu,
            line_dash="dot",
            line_color="#666",
            annotation_text=f"Moyenne période : {mu:.1f}%",
        )
    fig.update_layout(
        title="Concentration : top 5 instruments (% du volume total de la séance)",
        height=340,
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
        margin=dict(l=20, r=20, t=48, b=20),
        yaxis=dict(range=[0, 100], ticksuffix="%", title="% du volume jour"),
    )
    return fig


def market_has_segment_volumes(df: pd.DataFrame) -> bool:
    """True si le DataFrame marché contient les colonnes de volume par segment."""
    return all(c in df.columns for c in SEGMENT_VOLUME_COLUMNS)


def chart_volume_by_segment_stacked(df_market: pd.DataFrame) -> go.Figure:
    """Histogrammes empilés : volume MAD par segment (actions, OPCVM, obligations, autre)."""
    if not market_has_segment_volumes(df_market):
        fig = go.Figure()
        fig.add_annotation(
            text="Volumes par segment indisponibles (régénérer les Parquet ou vérifier la feuille Cours).",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font=dict(size=14),
        )
        fig.update_layout(height=320, plot_bgcolor="white")
        return fig

    dm = df_market.copy()
    dm["Jour"] = pd.to_datetime(dm["Jour"])
    labels = {
        "Volume_MAD_Actions": "Actions",
        "Volume_MAD_OPCVM": "OPCVM",
        "Volume_MAD_Obligations": "Obligations",
        "Volume_MAD_Autre": "Autre",
    }
    colors = [BVC_BLUE, "#2E7D32", "#F57C00", "#9E9E9E"]
    fig = go.Figure()
    for i, col in enumerate(SEGMENT_VOLUME_COLUMNS):
        fig.add_trace(go.Bar(
            x=dm["Jour"],
            y=dm[col].fillna(0) / 1e6,
            name=labels.get(col, col),
            marker_color=colors[i % len(colors)],
            opacity=0.88,
        ))
    fig.update_layout(
        title="Volume journalier par segment (M MAD — empilé)",
        barmode="stack",
        height=380,
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(orientation="h", y=1.02),
        margin=dict(l=20, r=20, t=50, b=20),
        yaxis_title="Volume (M MAD)",
    )
    return fig


def chart_volume_segment_share_pct(df_market: pd.DataFrame) -> go.Figure:
    """Répartition journalière du volume entre segments (% de la somme des segments)."""
    if not market_has_segment_volumes(df_market):
        fig = go.Figure()
        fig.update_layout(height=280, plot_bgcolor="white")
        return fig

    dm = df_market.copy()
    dm["Jour"] = pd.to_datetime(dm["Jour"])
    tot = dm[list(SEGMENT_VOLUME_COLUMNS)].sum(axis=1).replace(0, np.nan)
    labels = {
        "Volume_MAD_Actions": "Actions",
        "Volume_MAD_OPCVM": "OPCVM",
        "Volume_MAD_Obligations": "Obligations",
        "Volume_MAD_Autre": "Autre",
    }
    colors = [BVC_BLUE, "#2E7D32", "#F57C00", "#9E9E9E"]
    fig = go.Figure()
    for i, col in enumerate(SEGMENT_VOLUME_COLUMNS):
        pct = np.where(tot.notna(), dm[col].fillna(0) / tot * 100.0, 0.0)
        fig.add_trace(go.Bar(
            x=dm["Jour"],
            y=pct,
            name=labels.get(col, col),
            marker_color=colors[i % len(colors)],
            opacity=0.88,
        ))
    fig.update_layout(
        title="Part du volume par segment (% de la somme des segments — quotidien)",
        barmode="stack",
        height=360,
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(orientation="h", y=1.02),
        margin=dict(l=20, r=20, t=50, b=20),
        yaxis=dict(range=[0, 100], ticksuffix="%"),
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


def chart_market_stress(df_market: pd.DataFrame) -> go.Figure:
    """
    Score de stress marché composite + décomposition (volatilité, breadth, OIR).
    Mis à jour à chaque rechargement des données (dernière séance = bord droit du graphique).
    """
    req = ["Jour", "Market_Stress_Score", "Stress_Vol", "Stress_Breadth", "Stress_OIR"]
    if not all(c in df_market.columns for c in req):
        fig = go.Figure()
        fig.add_annotation(
            text="Colonnes Market Stress absentes — rechargez les données.",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
        )
        fig.update_layout(height=360, title="Market Stress Score")
        return fig

    d = df_market.sort_values("Jour").copy()
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.42, 0.58],
        subplot_titles=[
            "Market Stress Score — composite (0–100)",
            "Composantes (même échelle)",
        ],
        vertical_spacing=0.1,
    )
    fig.add_trace(
        go.Scatter(
            x=d["Jour"],
            y=d["Market_Stress_Score"],
            name="Stress global",
            line=dict(color="#B71C1C", width=2.5),
            fill="tozeroy",
            fillcolor="rgba(183,28,28,0.08)",
        ),
        row=1,
        col=1,
    )
    fig.add_hline(y=50, line_dash="dash", line_color="orange", opacity=0.7, row=1, col=1)
    fig.add_hline(y=75, line_dash="dot", line_color="red", opacity=0.6, row=1, col=1)

    fig.add_trace(
        go.Scatter(x=d["Jour"], y=d["Stress_Vol"], name="Stress vol. MASI", line=dict(color="#1565C0")),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=d["Jour"], y=d["Stress_Breadth"], name="Stress breadth", line=dict(color="#6A1B9A")),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=d["Jour"], y=d["Stress_OIR"], name="Stress OIR (|OIR| moy.)", line=dict(color="#2E7D32")),
        row=2,
        col=1,
    )
    fig.update_layout(
        height=520,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=20, r=20, t=56, b=20),
    )
    fig.update_yaxes(range=[0, 105], row=1, col=1)
    fig.update_yaxes(range=[0, 105], row=2, col=1)
    return fig


def gauge_market_stress(score: float, regime: str) -> go.Figure:
    """Jauge du stress (dernière séance)."""
    val = float(score) if score is not None and not (isinstance(score, float) and np.isnan(score)) else 0.0
    val = max(0.0, min(100.0, val))
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=val,
            number={"suffix": "/100", "font": {"size": 36}},
            title={"text": f"Régime : {regime}", "font": {"size": 14}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": "#002366"},
                "bgcolor": "white",
                "borderwidth": 1,
                "bordercolor": "#ccc",
                "steps": [
                    {"range": [0, 25], "color": "#E8F5E9"},
                    {"range": [25, 50], "color": "#FFF9C4"},
                    {"range": [50, 75], "color": "#FFE0B2"},
                    {"range": [75, 100], "color": "#FFCDD2"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 3},
                    "thickness": 0.8,
                    "value": 85,
                },
            },
        )
    )
    fig.update_layout(height=280, margin=dict(l=24, r=24, t=40, b=16))
    return fig


def chart_market_quality_mini(df_market: pd.DataFrame, n_sessions: int = 22) -> go.Figure:
    """
    Qualité marché = 100 − stress (plus haut = mieux). Dernières séances pour lecture tendance.
    """
    if "Market_Quality_Score" not in df_market.columns:
        fig = go.Figure()
        fig.add_annotation(
            text="Score qualité absent.",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
        )
        fig.update_layout(height=200)
        return fig

    d = (
        df_market.sort_values("Jour")
        .dropna(subset=["Market_Quality_Score"])
        .tail(n_sessions)
    )
    fig = go.Figure(
        go.Scatter(
            x=d["Jour"],
            y=d["Market_Quality_Score"],
            mode="lines+markers",
            line=dict(color="#1B5E20", width=2.2),
            marker=dict(size=6, color="#2E7D32"),
            fill="tozeroy",
            fillcolor="rgba(46,125,50,0.1)",
            hovertemplate="%{x|%Y-%m-%d}<br>Qualité: %{y:.1f}/100<extra></extra>",
        )
    )
    fig.add_hline(y=50, line_dash="dash", line_color="#9E9E9E", opacity=0.7)
    fig.update_layout(
        title="Qualité du marché (100 = meilleur)",
        height=240,
        yaxis=dict(range=[0, 100], title=""),
        xaxis_title="",
        margin=dict(l=12, r=12, t=44, b=8),
        showlegend=False,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return fig

