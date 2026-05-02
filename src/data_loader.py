"""
Module de chargement et préparation des données BVC.
Gère l'ingestion depuis le fichier Excel DATASET-2025 et tout nouveau fichier
de même format uploadé via l'interface.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
import warnings
warnings.filterwarnings('ignore')

# Chemins par défaut
DATA_DIR = Path(__file__).parent.parent / "data"
DEFAULT_FILE = DATA_DIR / "DATASET-2025.xlsx"


# ─────────────────────────────────────────────────────────────
# 1.  CHARGEMENT BRUT
# ─────────────────────────────────────────────────────────────

def load_excel(filepath: str | Path = DEFAULT_FILE) -> dict[str, pd.DataFrame]:
    """
    Charge toutes les feuilles d'un fichier Excel BVC et retourne un dict.
    Compatible avec DATASET-2025.xlsx et tout fichier de même format.

    Returns
    -------
    dict avec clés : 'indicateurs', 'indices', 'cours', 'intraday'
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Fichier introuvable : {filepath}")

    xf = pd.ExcelFile(filepath)
    available = {s.lower(): s for s in xf.sheet_names}
    result = {}

    # --- Indicateurs ---
    if 'indicateurs' in available:
        df = pd.read_excel(xf, sheet_name=available['indicateurs'], parse_dates=['Jour'])
        result['indicateurs'] = _clean_indicateurs(df)

    # --- Indices ---
    if 'indices' in available:
        df = pd.read_excel(xf, sheet_name=available['indices'], parse_dates=['Jour'])
        result['indices'] = _clean_indices(df)

    # --- Cours ---
    if 'cours' in available:
        df = pd.read_excel(xf, sheet_name=available['cours'], parse_dates=['Jour'])
        result['cours'] = _clean_cours(df)

    # --- Intraday ---
    if 'intraday' in available:
        df = pd.read_excel(xf, sheet_name=available['intraday'], parse_dates=['Jour'])
        result['intraday'] = _clean_intraday(df)

    return result


# ─────────────────────────────────────────────────────────────
# 2.  NETTOYAGE PAR FEUILLE
# ─────────────────────────────────────────────────────────────

def _clean_indicateurs(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=['Jour'])
    df.columns = _normalize_cols(df.columns)
    rename = {
        'jour': 'Jour', 'volume': 'Volume_MAD',
        'quantite_titre': 'Quantite_Titres', 'nb_contrat': 'Nb_Contrats'
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    df = df.sort_values('Jour').reset_index(drop=True)
    return df


def _clean_indices(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=['Jour'])
    df.columns = _normalize_cols(df.columns)
    rename = {
        'jour': 'Jour', 'code_indice': 'Code_Indice', 'libelle_fr': 'Libelle',
        'variation_veille': 'Variation_Veille_pct', 'variation_30_12': 'Variation_YTD_pct',
        'indice_ph_j': 'Cours_Haut', 'indice_pb_j': 'Cours_Bas'
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    if 'Code_Indice' in df.columns:
        df['Code_Indice'] = df['Code_Indice'].str.strip()
    df = df.sort_values(['Jour', 'Code_Indice']).reset_index(drop=True)
    return df


def _clean_cours(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=['Jour'])
    # Supprimer colonnes non nommées
    df = df.drop(columns=[c for c in df.columns if 'Unnamed' in str(c)], errors='ignore')
    df.columns = _normalize_cols(df.columns)
    rename = {
        'jour': 'Jour', 'code_valeur': 'Code_Valeur', 'ticker': 'Ticker',
        'libelle_fr': 'Libelle', 'cours_de_reference': 'Cours_Ref',
        'cours_cloture': 'Cours_Cloture', 'cours_meilleure_ofr': 'Meilleure_Offre',
        'cours_meilleure_dem': 'Meilleure_Demande', 'capitalisation': 'Capitalisation',
        'cours_pb_j': 'Cours_Bas', 'cours_ph_j': 'Cours_Haut',
        'volume': 'Volume_MAD', 'quantite_titre': 'Quantite_Titres',
        'nb_contrat': 'Nb_Contrats'
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    if 'Ticker' in df.columns:
        df['Ticker'] = df['Ticker'].str.strip()
    df = df.sort_values(['Jour', 'Ticker']).reset_index(drop=True)
    return df


def _clean_intraday(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=['Jour'])
    df.columns = _normalize_cols(df.columns)
    rename = {
        'jour': 'Jour', 'quantite_titre': 'Quantite_Titres', 'nb_contrat': 'Nb_Contrats',
        'ticker': 'Ticker', 'libelle_fr': 'Libelle', 'sens': 'Sens',
        'heure_transaction': 'Heure_Transaction', 'cours_transaction': 'Cours_Transaction',
        'num_transaction': 'Num_Transaction'
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    if 'Ticker' in df.columns:
        df['Ticker'] = df['Ticker'].str.strip()
    if 'Heure_Transaction' in df.columns:
        df['Heure_dt'] = pd.to_datetime(df['Heure_Transaction'], errors='coerce')
        df['Heure_int'] = df['Heure_dt'].dt.hour
        df['Minute_int'] = df['Heure_dt'].dt.minute
    df = df.sort_values(['Jour', 'Heure_Transaction']).reset_index(drop=True)
    return df


def _normalize_cols(columns) -> list[str]:
    """Normalise les noms de colonnes : minuscules + underscores."""
    return [
        str(c).strip().lower()
         .replace(' ', '_').replace('-', '_')
         .replace('é', 'e').replace('è', 'e').replace('ê', 'e')
         .replace('à', 'a').replace('ô', 'o').replace('ç', 'c')
        for c in columns
    ]


# ─────────────────────────────────────────────────────────────
# 3.  CHARGEMENT DEPUIS PARQUET (CACHE)
# ─────────────────────────────────────────────────────────────

def load_from_cache() -> Optional[dict[str, pd.DataFrame]]:
    """Charge les données depuis les fichiers Parquet si disponibles."""
    files = {
        'indicateurs': DATA_DIR / 'indicateurs_enrichis.parquet',
        'indices': DATA_DIR / 'indices_enrichis.parquet',
        'cours': DATA_DIR / 'cours_enrichis.parquet',
        'intraday': DATA_DIR / 'intraday_enrichis.parquet',
    }
    if not all(f.exists() for f in files.values()):
        return None
    return {k: pd.read_parquet(v) for k, v in files.items()}


def save_to_cache(data: dict[str, pd.DataFrame]) -> None:
    """Sauvegarde les DataFrames enrichis en Parquet."""
    DATA_DIR.mkdir(exist_ok=True)
    mapping = {
        'indicateurs': 'indicateurs_enrichis.parquet',
        'indices': 'indices_enrichis.parquet',
        'cours': 'cours_enrichis.parquet',
        'intraday': 'intraday_enrichis.parquet',
    }
    for key, filename in mapping.items():
        if key in data:
            data[key].to_parquet(DATA_DIR / filename, index=False)


# ─────────────────────────────────────────────────────────────
# 4.  VALIDATION DU FORMAT
# ─────────────────────────────────────────────────────────────

REQUIRED_COLS = {
    'indicateurs': ['Jour', 'Volume_MAD', 'Quantite_Titres', 'Nb_Contrats'],
    'indices': ['Jour', 'Code_Indice', 'Variation_Veille_pct', 'Cours_Haut', 'Cours_Bas'],
    'cours': ['Jour', 'Ticker', 'Cours_Cloture', 'Volume_MAD', 'Capitalisation'],
    'intraday': ['Jour', 'Ticker', 'Cours_Transaction', 'Quantite_Titres', 'Sens'],
}


def validate_format(data: dict[str, pd.DataFrame]) -> dict[str, list[str]]:
    """
    Valide que les colonnes requises sont présentes dans chaque feuille.

    Returns
    -------
    dict de listes de colonnes manquantes par feuille (vide = OK)
    """
    errors = {}
    for sheet, required in REQUIRED_COLS.items():
        if sheet not in data:
            errors[sheet] = ['FEUILLE MANQUANTE']
            continue
        cols = list(data[sheet].columns)
        missing = [c for c in required if c not in cols]
        if missing:
            errors[sheet] = missing
    return errors


def get_summary(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Retourne un tableau récapitulatif des données chargées."""
    rows = []
    for sheet, df in data.items():
        rows.append({
            'Feuille': sheet,
            'Lignes': len(df),
            'Colonnes': len(df.columns),
            'Date_Min': df['Jour'].min() if 'Jour' in df.columns else None,
            'Date_Max': df['Jour'].max() if 'Jour' in df.columns else None,
            'Nulls_Total': df.isnull().sum().sum(),
        })
    return pd.DataFrame(rows)
