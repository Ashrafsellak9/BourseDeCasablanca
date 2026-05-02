"""
Thème BVC clair partagé — à appeler sur chaque page Streamlit.
Sans cela, seul Accueil.py injecte le CSS : l’accueil et les sous-pages divergent.
"""
from __future__ import annotations

import streamlit as st

# Style commun (identique partout : sidebar #002366, fond #f5f7fb, métriques #EAF0FB).
_BVC_CORE_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body {
    margin: 0 !important;
    padding: 0 !important;
    background-color: #f5f7fb !important;
}
html, body, .stApp {
    background-color: #f5f7fb !important;
    color: #1a2744 !important;
    font-family: 'Inter', sans-serif !important;
}
/* Bande blanche en haut : pas d’espace sous la barre Streamlit + conteneur collé */
[data-testid="stAppViewContainer"] {
    padding-top: 0 !important;
    margin-top: 0 !important;
}
[data-testid="stAppViewContainer"] > .main,
[data-testid="stAppViewContainer"] section.main,
[data-testid="stAppViewContainer"] > div {
    padding-top: 0 !important;
    margin-top: 0 !important;
}
.stApp > div:first-child {
    padding-top: 0 !important;
}
header[data-testid="stHeader"] {
    display: none !important;
    height: 0 !important;
    min-height: 0 !important;
    visibility: hidden !important;
    overflow: hidden !important;
}
.block-container {
    padding-top: 0 !important;
    padding-left: 1.5rem !important;
    padding-right: 1.5rem !important;
    max-width: 100% !important;
    margin-top: 0 !important;
}
/* Titres : zone principale seulement (la sidebar a son propre contraste) */
section[data-testid="stMain"] h1,
section[data-testid="stMain"] h2,
section[data-testid="stMain"] h3,
section[data-testid="stMain"] h4,
section.main h1,
section.main h2,
section.main h3,
section.main h4 {
    color: #002366 !important;
}
section[data-testid="stMain"] [data-testid="stMarkdownContainer"] p,
section[data-testid="stMain"] [data-testid="stMarkdownContainer"] li,
section.main [data-testid="stMarkdownContainer"] p,
section.main [data-testid="stMarkdownContainer"] li {
    color: #334155 !important;
}

/*
 * Bandeau ML (.bvc-hero-banner) : les règles « main … stMarkdownContainer p »
 * ont une spécificité plus forte que « .bvc-hero-banner p » — il faut répéter
 * le chemin complet pour titre + sous-titre lisibles sur le dégradé.
 */
section[data-testid="stMain"] [data-testid="stMarkdownContainer"] .bvc-hero-banner h1,
section[data-testid="stMain"] [data-testid="stMarkdownContainer"] .bvc-hero-banner h2,
section[data-testid="stMain"] [data-testid="stMarkdownContainer"] .bvc-hero-banner h3,
section.main [data-testid="stMarkdownContainer"] .bvc-hero-banner h1,
section.main [data-testid="stMarkdownContainer"] .bvc-hero-banner h2,
section.main [data-testid="stMarkdownContainer"] .bvc-hero-banner h3 {
    color: #ffffff !important;
}
section[data-testid="stMain"] [data-testid="stMarkdownContainer"] .bvc-hero-banner p,
section[data-testid="stMain"] [data-testid="stMarkdownContainer"] .bvc-hero-banner li,
section.main [data-testid="stMarkdownContainer"] .bvc-hero-banner p,
section.main [data-testid="stMarkdownContainer"] .bvc-hero-banner li {
    color: #f1f5ff !important;
}
.bvc-hero-banner h1,
.bvc-hero-banner h2,
.bvc-hero-banner h3 {
    color: #ffffff !important;
}
.bvc-hero-banner p,
.bvc-hero-banner li {
    color: #f1f5ff !important;
}

section[data-testid="stMain"] .bvc-hero-banner h1,
section[data-testid="stMain"] .bvc-hero-banner h2,
section[data-testid="stMain"] .bvc-hero-banner h3,
section.main .bvc-hero-banner h1,
section.main .bvc-hero-banner h2,
section.main .bvc-hero-banner h3 {
    color: #ffffff !important;
}
section[data-testid="stMain"] .bvc-hero-banner p,
section.main .bvc-hero-banner p {
    color: #f1f5ff !important;
}

[data-testid="stDecoration"] {
    background-color: #f5f7fb !important;
}
section[data-testid="stMain"],
section.main {
    background-color: #f5f7fb !important;
}
section[data-testid="stMain"] > div {
    padding-top: 0 !important;
}
[data-testid="stCaption"] { color: #64748b !important; }

[data-testid="stSidebar"] { background: #002366 !important; }
[data-testid="stSidebar"] * { color: #fff !important; }
[data-testid="stSidebar"] a,
[data-testid="stSidebarNav"] a { color: #fff !important; }
[data-testid="stSidebarNav"] a[aria-current="page"] {
    background: rgba(255,255,255,0.14) !important;
    border-radius: 6px !important;
}
/* Libellés / légendes widgets dans la sidebar (sinon règles « main » en #334155 les masquent) */
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] label p,
[data-testid="stSidebar"] .stSelectbox label p,
[data-testid="stSidebar"] .stMultiSelect label p,
[data-testid="stSidebar"] .stTextInput label p,
[data-testid="stSidebar"] .stDateInput label p,
[data-testid="stSidebar"] .stSlider label p,
[data-testid="stSidebar"] .stRadio label p,
[data-testid="stSidebar"] .stCheckbox label p,
[data-testid="stSidebar"] .stNumberInput label p {
    color: #f1f5ff !important;
}
[data-testid="stSidebar"] [data-testid="stCaption"] {
    color: rgba(255, 255, 255, 0.88) !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #ffffff !important;
}

div[data-testid="metric-container"],
[data-testid="stMetric"] {
    background: #EAF0FB !important;
    border-radius: 8px !important;
    padding: 12px !important;
    border: 1px solid #c5d4eb !important;
    border-left: 4px solid #003087 !important;
}
[data-testid="stMetricValue"] { color: #002366 !important; }
[data-testid="stMetricLabel"] p { color: #5a6b8a !important; font-size: 11px !important; }

.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid #c5d4eb !important;
}
.stTabs [data-baseweb="tab"] { color: #64748b !important; }
.stTabs [aria-selected="true"] {
    color: #002366 !important;
    border-bottom: 2px solid #003087 !important;
}

[data-testid="stDataFrame"] {
    border: 1px solid #c5d4eb !important;
    border-radius: 8px !important;
}
[data-testid="stPlotlyChart"] {
    background: #fff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 8px !important;
    padding: 4px !important;
}

.stButton > button {
    background: #fff !important;
    border: 1px solid #c5d4eb !important;
    color: #002366 !important;
    border-radius: 6px !important;
}
.stButton > button[kind="primary"],
.stButton > button[data-testid="baseButton-primary"] {
    background: linear-gradient(180deg, #003087 0%, #002366 100%) !important;
    color: #fff !important;
    border-color: #002366 !important;
}
div[data-testid="stDownloadButton"] button {
    background: #fff !important;
    border: 1px solid #c5d4eb !important;
    color: #002366 !important;
}

.stSelectbox > div > div, .stMultiSelect > div > div,
.stTextInput > div > div input, .stDateInput > div > div input,
.stNumberInput > div > div input {
    background: #fff !important;
    border: 1px solid #c5d4eb !important;
    color: #002366 !important;
    border-radius: 6px !important;
}
/* Libellés widgets : uniquement dans la zone principale (fond clair) */
section[data-testid="stMain"] .stSelectbox label p,
section[data-testid="stMain"] .stMultiSelect label p,
section[data-testid="stMain"] .stTextInput label p,
section[data-testid="stMain"] .stDateInput label p,
section[data-testid="stMain"] .stSlider label p,
section[data-testid="stMain"] .stRadio label p,
section[data-testid="stMain"] .stCheckbox label p,
section[data-testid="stMain"] .stNumberInput label p,
section.main .stSelectbox label p,
section.main .stMultiSelect label p,
section.main .stTextInput label p,
section.main .stDateInput label p,
section.main .stSlider label p,
section.main .stRadio label p,
section.main .stCheckbox label p,
section.main .stNumberInput label p {
    color: #334155 !important;
}

.streamlit-expanderHeader {
    background: #fff !important;
    border: 1px solid #c5d4eb !important;
    color: #002366 !important;
    border-radius: 6px !important;
}
.streamlit-expanderContent {
    background: #fafbff !important;
    border: 1px solid #c5d4eb !important;
    border-top: none !important;
    color: #334155 !important;
}

.stSlider > div > div > div { background: #cbd5e1 !important; }
.stSlider > div > div > div > div { background: #003087 !important; }

[data-testid="stAlert"] {
    background: #fff !important;
    border: 1px solid #c5d4eb !important;
}
hr { border-color: #c5d4eb !important; }

footer, #MainMenu { visibility: hidden !important; }
[data-testid="stToolbar"] { display: none !important; }
</style>
"""

# Uniquement l’accueil (en-tête + carrousel alertes).
_BVC_HOME_EXTRA_CSS = """
<style>
.main-header {
    background: linear-gradient(135deg, #002366 0%, #003087 55%, #C8A84B 130%);
    padding: 22px 28px;
    border-radius: 10px;
    margin-bottom: 1rem;
    margin-top: 0;
    border: 1px solid #003087;
}
.main-header h1 { color: #fff !important; margin: 0; font-size: 1.85rem; }
.main-header p { color: #E8ECF5 !important; margin: 6px 0 0 0; font-size: 0.95rem; }
/* Même cause que le bandeau ML : section…h1 / stMarkdownContainer p gagnent sur .main-header */
section[data-testid="stMain"] [data-testid="stMarkdownContainer"] .main-header h1,
section.main [data-testid="stMarkdownContainer"] .main-header h1,
section[data-testid="stMain"] .main-header h1,
section.main .main-header h1 {
    color: #ffffff !important;
    margin: 0 !important;
    font-size: 1.85rem !important;
}
section[data-testid="stMain"] [data-testid="stMarkdownContainer"] .main-header p,
section.main [data-testid="stMarkdownContainer"] .main-header p,
section[data-testid="stMain"] .main-header p,
section.main .main-header p {
    color: #f1f5ff !important;
    margin: 6px 0 0 0 !important;
    font-size: 0.95rem !important;
}
.bvc-carousel-card {
    border: 1px solid #c5d4eb !important;
    border-radius: 10px !important;
    padding: 1.1rem 1.25rem !important;
    background: linear-gradient(180deg, #ffffff 0%, #f0f4fb 100%) !important;
    box-shadow: 0 2px 12px rgba(0,35,102,0.08) !important;
    min-height: 168px !important;
}
.bvc-carousel-meta { color: #64748b !important; font-size: 0.88rem !important; margin-bottom: 0.35rem !important; }
.bvc-carousel-title { color: #002366 !important; font-size: 1.12rem !important; font-weight: 700 !important; margin: 0 0 0.5rem 0 !important; }
.bvc-carousel-ind { color: #b8922e !important; font-weight: 600 !important; }
.bvc-carousel-desc { color: #475569 !important; font-size: 0.95rem !important; margin: 0.4rem 0 0.2rem 0 !important; }
.bvc-carousel-meta2 { color: #64748b !important; font-size: 0.9rem !important; margin: 0 !important; }
</style>
"""

# Page Upload : zone dépôt (inchangée visuellement).
UPLOAD_ZONE_CSS = """
.upload-zone {
    border: 2px dashed #C8A84B;
    border-radius: 12px;
    padding: 30px;
    text-align: center;
    background: #FAFBFF;
    margin: 20px 0;
    color: #002366;
}
"""


def inject_bvc_theme(*, home: bool = False, extra_css: str | None = None) -> None:
    """Injecte le thème BVC clair. `home=True` ajoute en-tête + carrousel accueil."""
    chunks: list[str] = [_BVC_CORE_CSS]
    if home:
        chunks.append(_BVC_HOME_EXTRA_CSS)
    if extra_css and extra_css.strip():
        chunks.append(f"<style>\n{extra_css.strip()}\n</style>")
    st.markdown("".join(chunks), unsafe_allow_html=True)
