"""
Module Machine Learning — Phase 5 BVC Surveillance.

Deux modèles non supervisés pour la détection d'anomalies :

1. Isolation Forest  (sklearn) — arbre d'isolation par partitionnement aléatoire
2. Autoencoder       (Keras)   — reconstruction d'erreur sur représentation compressée

Les deux modèles opèrent sur trois niveaux de features :
  - Marché Global   : rendements MASI, volatilité, volume, breadth, HHI
  - Instrument      : retours, volatilités rolling, RSI, spread, turnover, OIR
  - Flux d'Ordres   : OIR, OAR, VWAP déviation, volatilité intraday

Chaque modèle produit :
  - anomaly_score : score continu (plus élevé = plus anormal)
  - is_anomaly    : booléen (flag d'anomalie)
  - severity_ml   : Normal / Faible / Modéré / Critique
"""

from __future__ import annotations

import os
import pickle
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── Répertoire modèles ──────────────────────────────────────────────────────
MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

# ── Constantes sévérité ─────────────────────────────────────────────────────
NORMAL   = "Normal"
FAIBLE   = "Faible"
MODERE   = "Modéré"
CRITIQUE = "Critique"

# ── Features par niveau ─────────────────────────────────────────────────────
MARKET_FEATURES = [
    "MASI_Rendement_pct", "MASI_Vol_10j", "Volume_MAD",
    "Volume_Relatif_Marche", "Breadth_pct", "HHI_Volume", "AD_Line",
]

INSTRUMENT_FEATURES = [
    "Rendement_pct", "Volatilite_5j", "Volatilite_10j", "Volatilite_20j",
    "Volume_Relatif", "RSI_14", "Spread_pct", "Turnover_Ratio",
    "Momentum_5j", "Momentum_20j",
]

ORDERFLOW_FEATURES = [
    "OIR", "OAR", "Intraday_Vol", "VWAP_Deviation_pct",
    "Spoof_Intensity_pct",
]


# ═══════════════════════════════════════════════════════════════════════════
# UTILITAIRES
# ═══════════════════════════════════════════════════════════════════════════

def _select_features(df: pd.DataFrame, feature_list: list[str]) -> list[str]:
    """Retourne les features disponibles dans le DataFrame."""
    return [f for f in feature_list if f in df.columns]


def _prepare_matrix(
    df: pd.DataFrame,
    features: list[str],
    fill_method: str = "median",
) -> tuple[np.ndarray, pd.Index]:
    """
    Extrait et normalise la matrice de features, gère les NaN.
    Retourne (X_scaled, valid_index).
    """
    sub = df[features].copy()
    if fill_method == "median":
        sub = sub.fillna(sub.median())
    else:
        sub = sub.fillna(0)
    # Supprime les lignes encore NaN
    valid = sub.dropna()
    scaler = StandardScaler()
    X = scaler.fit_transform(valid.values)
    return X, valid.index, scaler


def _score_to_severity(scores: np.ndarray, contamination: float = 0.05) -> list[str]:
    """
    Convertit les scores d'anomalie normalisés [0..1] en sévérité.
    """
    p75 = np.percentile(scores, 75)
    p90 = np.percentile(scores, 90)
    p95 = np.percentile(scores, 95)

    sevs = []
    for s in scores:
        if s >= p95:
            sevs.append(CRITIQUE)
        elif s >= p90:
            sevs.append(MODERE)
        elif s >= p75:
            sevs.append(FAIBLE)
        else:
            sevs.append(NORMAL)
    return sevs


# ═══════════════════════════════════════════════════════════════════════════
# 1. ISOLATION FOREST
# ═══════════════════════════════════════════════════════════════════════════

class IsolationForestDetector:
    """
    Détecteur d'anomalies basé sur Isolation Forest.

    L'Isolation Forest isole les anomalies en partitionnant aléatoirement
    les données. Les anomalies nécessitent moins de partitions pour être
    isolées (score proche de 1).

    Paramètres
    ----------
    contamination : fraction attendue d'anomalies (défaut 5 %)
    n_estimators  : nombre d'arbres (défaut 200)
    random_state  : graine aléatoire pour reproductibilité
    """

    def __init__(
        self,
        contamination: float = 0.05,
        n_estimators: int = 200,
        random_state: int = 42,
    ):
        self.contamination = contamination
        self.n_estimators  = n_estimators
        self.random_state  = random_state
        self.models: dict   = {}   # level -> IsolationForest
        self.scalers: dict  = {}   # level -> StandardScaler
        self.features: dict = {}   # level -> list[str]

    # ── Entraînement ────────────────────────────────────────────────────────
    def fit(
        self,
        df_market: pd.DataFrame,
        df_instrument: pd.DataFrame,
        df_orderflow: pd.DataFrame,
    ) -> "IsolationForestDetector":
        """Entraîne un modèle Isolation Forest par niveau."""
        levels = {
            "market":     (df_market,     MARKET_FEATURES),
            "instrument": (df_instrument, INSTRUMENT_FEATURES),
            "orderflow":  (df_orderflow,  ORDERFLOW_FEATURES),
        }
        for level, (df, feat_list) in levels.items():
            feats = _select_features(df, feat_list)
            if not feats or df.empty:
                continue
            X, idx, scaler = _prepare_matrix(df, feats)
            if len(X) < 10:
                continue
            model = IsolationForest(
                n_estimators=self.n_estimators,
                contamination=self.contamination,
                random_state=self.random_state,
                n_jobs=-1,
            )
            model.fit(X)
            self.models[level]  = model
            self.scalers[level] = scaler
            self.features[level] = feats
        return self

    # ── Prédiction ──────────────────────────────────────────────────────────
    def predict(
        self,
        df: pd.DataFrame,
        level: str,
    ) -> pd.DataFrame:
        """
        Applique le modèle pré-entraîné sur df.

        Retourne un DataFrame avec les colonnes :
          IF_Score, IF_IsAnomaly, IF_Severity
        """
        if level not in self.models:
            return pd.DataFrame(index=df.index)

        feats   = self.features[level]
        scaler  = self.scalers[level]
        model   = self.models[level]
        avail   = _select_features(df, feats)

        sub = df[avail].copy().fillna(df[avail].median())
        valid = sub.dropna()
        if valid.empty:
            return pd.DataFrame(index=df.index)

        X = scaler.transform(valid[avail].values)

        # score_samples : plus négatif = plus anormal → on inverse
        raw_scores  = model.score_samples(X)
        # Normaliser en [0, 1] : anomaly_score = 1 - (score + 0.5)
        norm_scores = 1.0 - (raw_scores - raw_scores.min()) / (
            raw_scores.max() - raw_scores.min() + 1e-10
        )
        predictions = model.predict(X)   # -1 = anomalie, +1 = normal
        is_anomaly  = (predictions == -1).astype(int)
        severities  = _score_to_severity(norm_scores)

        result = pd.DataFrame(
            {
                "IF_Score":     norm_scores,
                "IF_IsAnomaly": is_anomaly,
                "IF_Severity":  severities,
            },
            index=valid.index,
        )
        return df.join(result[["IF_Score", "IF_IsAnomaly", "IF_Severity"]])

    # ── Persistence ─────────────────────────────────────────────────────────
    def save(self, path: Optional[Path] = None) -> Path:
        path = path or MODELS_DIR / "isolation_forest.pkl"
        with open(path, "wb") as f:
            pickle.dump(self, f)
        return path

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "IsolationForestDetector":
        path = path or MODELS_DIR / "isolation_forest.pkl"
        with open(path, "rb") as f:
            return pickle.load(f)


# ═══════════════════════════════════════════════════════════════════════════
# 2. AUTOENCODER (Keras / TensorFlow)
# ═══════════════════════════════════════════════════════════════════════════

class AutoencoderDetector:
    """
    Détecteur d'anomalies basé sur un Autoencoder.

    L'autoencoder apprend à reconstruire les données normales via une
    couche bottleneck (dimension réduite). Les anomalies produisent une
    erreur de reconstruction élevée.

    Architecture : Input → Dense(64) → Dense(32) → Dense(bottleneck)
                         → Dense(32) → Dense(64)  → Output (reconstruction)

    Paramètres
    ----------
    encoding_dim  : dimension de l'espace latent
    epochs        : nombre d'époques d'entraînement
    batch_size    : taille du batch
    contamination : fraction attendue d'anomalies (pour le seuil)
    """

    def __init__(
        self,
        encoding_dim: int = 8,
        epochs: int = 50,
        batch_size: int = 32,
        contamination: float = 0.05,
    ):
        self.encoding_dim  = encoding_dim
        self.epochs        = epochs
        self.batch_size    = batch_size
        self.contamination = contamination
        self.models: dict   = {}
        self.scalers: dict  = {}
        self.features: dict = {}
        self.thresholds: dict = {}

    def _build_model(self, input_dim: int):
        """Construit l'architecture de l'autoencoder."""
        try:
            import tensorflow as tf
            from tensorflow import keras
            tf.get_logger().setLevel("ERROR")
        except ImportError:
            raise ImportError(
                "TensorFlow requis pour l'Autoencoder. "
                "Installez-le avec : pip install tensorflow"
            )

        bottleneck = max(self.encoding_dim, 2)
        hidden1    = min(64, input_dim * 4)
        hidden2    = min(32, input_dim * 2)

        inputs  = keras.Input(shape=(input_dim,))
        encoded = keras.layers.Dense(hidden1, activation="relu")(inputs)
        encoded = keras.layers.Dense(hidden2, activation="relu")(encoded)
        encoded = keras.layers.Dense(bottleneck, activation="relu")(encoded)
        decoded = keras.layers.Dense(hidden2, activation="relu")(encoded)
        decoded = keras.layers.Dense(hidden1, activation="relu")(decoded)
        outputs = keras.layers.Dense(input_dim, activation="linear")(decoded)

        model = keras.Model(inputs, outputs, name="autoencoder")
        model.compile(optimizer="adam", loss="mse")
        return model

    # ── Entraînement ────────────────────────────────────────────────────────
    def fit(
        self,
        df_market: pd.DataFrame,
        df_instrument: pd.DataFrame,
        df_orderflow: pd.DataFrame,
        verbose: int = 0,
    ) -> "AutoencoderDetector":
        """Entraîne un autoencoder par niveau."""
        try:
            import tensorflow as tf
            tf.get_logger().setLevel("ERROR")
        except ImportError:
            raise ImportError("TensorFlow requis pour l'Autoencoder.")

        levels = {
            "market":     (df_market,     MARKET_FEATURES),
            "instrument": (df_instrument, INSTRUMENT_FEATURES),
            "orderflow":  (df_orderflow,  ORDERFLOW_FEATURES),
        }
        for level, (df, feat_list) in levels.items():
            feats = _select_features(df, feat_list)
            if not feats or df.empty:
                continue
            X, idx, scaler = _prepare_matrix(df, feats)
            if len(X) < 20:
                continue

            model = self._build_model(X.shape[1])
            model.fit(
                X, X,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_split=0.1,
                verbose=verbose,
                callbacks=[
                    __import__("tensorflow").keras.callbacks.EarlyStopping(
                        patience=5, restore_best_weights=True
                    )
                ],
            )
            # Seuil d'anomalie = percentile (1 - contamination) des erreurs
            recon = model.predict(X, verbose=0)
            errors = np.mean((X - recon) ** 2, axis=1)
            threshold = np.percentile(errors, (1 - self.contamination) * 100)

            self.models[level]     = model
            self.scalers[level]    = scaler
            self.features[level]   = feats
            self.thresholds[level] = threshold
        return self

    # ── Prédiction ──────────────────────────────────────────────────────────
    def predict(
        self,
        df: pd.DataFrame,
        level: str,
    ) -> pd.DataFrame:
        """
        Applique l'autoencoder pré-entraîné sur df.

        Retourne un DataFrame avec les colonnes :
          AE_ReconError, AE_IsAnomaly, AE_Severity
        """
        if level not in self.models:
            return df.copy()

        feats     = self.features[level]
        scaler    = self.scalers[level]
        model     = self.models[level]
        threshold = self.thresholds[level]
        avail     = _select_features(df, feats)

        sub   = df[avail].copy().fillna(df[avail].median())
        valid = sub.dropna()
        if valid.empty:
            return df.copy()

        X     = scaler.transform(valid[avail].values)
        recon = model.predict(X, verbose=0)
        errors = np.mean((X - recon) ** 2, axis=1)

        # Normaliser erreurs en [0, 1]
        norm_errors = (errors - errors.min()) / (
            errors.max() - errors.min() + 1e-10
        )
        is_anomaly = (errors > threshold).astype(int)
        severities = _score_to_severity(norm_errors)

        result = pd.DataFrame(
            {
                "AE_ReconError": errors,
                "AE_IsAnomaly":  is_anomaly,
                "AE_Severity":   severities,
            },
            index=valid.index,
        )
        return df.join(result[["AE_ReconError", "AE_IsAnomaly", "AE_Severity"]])

    # ── Persistence ─────────────────────────────────────────────────────────
    def save(self, path: Optional[Path] = None) -> Path:
        """Sauvegarde les weights Keras + metadata."""
        path = path or MODELS_DIR / "autoencoder"
        path = Path(path)
        path.mkdir(exist_ok=True)

        # Sauvegarde chaque sous-modèle
        meta = {
            "encoding_dim":  self.encoding_dim,
            "epochs":        self.epochs,
            "batch_size":    self.batch_size,
            "contamination": self.contamination,
            "scalers":       self.scalers,
            "features":      self.features,
            "thresholds":    self.thresholds,
        }
        with open(path / "meta.pkl", "wb") as f:
            pickle.dump(meta, f)

        for level, model in self.models.items():
            model.save(str(path / f"model_{level}.keras"))

        return path

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "AutoencoderDetector":
        try:
            import tensorflow as tf
            tf.get_logger().setLevel("ERROR")
        except ImportError:
            raise ImportError("TensorFlow requis pour charger l'Autoencoder.")

        path = Path(path or MODELS_DIR / "autoencoder")
        with open(path / "meta.pkl", "rb") as f:
            meta = pickle.load(f)

        detector = cls(
            encoding_dim=meta["encoding_dim"],
            epochs=meta["epochs"],
            batch_size=meta["batch_size"],
            contamination=meta["contamination"],
        )
        detector.scalers    = meta["scalers"]
        detector.features   = meta["features"]
        detector.thresholds = meta["thresholds"]

        for level_file in path.glob("model_*.keras"):
            level = level_file.stem.replace("model_", "")
            detector.models[level] = tf.keras.models.load_model(str(level_file))

        return detector


# ═══════════════════════════════════════════════════════════════════════════
# 3. PIPELINE COMPLET ML
# ═══════════════════════════════════════════════════════════════════════════

def train_all_models(
    df_market: pd.DataFrame,
    df_instrument: pd.DataFrame,
    df_orderflow: pd.DataFrame,
    contamination: float = 0.05,
    ae_epochs: int = 50,
    verbose: int = 0,
    save: bool = True,
) -> tuple[IsolationForestDetector, AutoencoderDetector]:
    """
    Entraîne l'Isolation Forest et l'Autoencoder sur les données 2025.

    Paramètres
    ----------
    df_market, df_instrument, df_orderflow : DataFrames pré-calculés (Phase 2)
    contamination : fraction estimée d'anomalies
    ae_epochs     : nombre d'époques pour l'autoencoder
    verbose       : niveau de verbosité Keras
    save          : sauvegarde les modèles dans models/

    Retourne
    --------
    (if_detector, ae_detector)
    """
    print("→ Entraînement Isolation Forest...")
    if_det = IsolationForestDetector(contamination=contamination)
    if_det.fit(df_market, df_instrument, df_orderflow)
    if save:
        p = if_det.save()
        print(f"  ✓ Isolation Forest sauvegardé → {p}")

    print("→ Entraînement Autoencoder...")
    try:
        ae_det = AutoencoderDetector(
            contamination=contamination,
            epochs=ae_epochs,
        )
        ae_det.fit(df_market, df_instrument, df_orderflow, verbose=verbose)
        if save:
            p = ae_det.save()
            print(f"  ✓ Autoencoder sauvegardé → {p}")
    except ImportError as e:
        print(f"  ⚠ Autoencoder ignoré : {e}")
        ae_det = None

    return if_det, ae_det


def apply_ml_detection(
    df: pd.DataFrame,
    level: str,
    if_detector: Optional[IsolationForestDetector] = None,
    ae_detector: Optional[AutoencoderDetector] = None,
) -> pd.DataFrame:
    """
    Applique IF et/ou AE sur un DataFrame et calcule un score ML combiné.

    Score ML combiné (0-100) :
      - IF seul      : IF_Score × 100
      - AE seul      : AE_ReconError normalisée × 100
      - IF + AE      : moyenne pondérée (60 % IF + 40 % AE)

    Sévérité finale ML basée sur le score combiné :
      Critique ≥ 80 | Modéré ≥ 60 | Faible ≥ 40 | Normal < 40
    """
    result = df.copy()

    if if_detector is not None:
        result = if_detector.predict(result, level)

    if ae_detector is not None:
        result = ae_detector.predict(result, level)

    # ── Score combiné ─────────────────────────────────────────────────────
    has_if = "IF_Score" in result.columns
    has_ae = "AE_ReconError" in result.columns

    if has_if and has_ae:
        ae_norm = (
            (result["AE_ReconError"] - result["AE_ReconError"].min())
            / (result["AE_ReconError"].max() - result["AE_ReconError"].min() + 1e-10)
        )
        result["ML_Score"] = (
            0.6 * result["IF_Score"] * 100 + 0.4 * ae_norm * 100
        ).round(1)
    elif has_if:
        result["ML_Score"] = (result["IF_Score"] * 100).round(1)
    elif has_ae:
        ae_norm = (
            (result["AE_ReconError"] - result["AE_ReconError"].min())
            / (result["AE_ReconError"].max() - result["AE_ReconError"].min() + 1e-10)
        )
        result["ML_Score"] = (ae_norm * 100).round(1)
    else:
        result["ML_Score"] = 0.0

    # ── Sévérité ML ─────────────────────────────────────────────────────────
    def _sev(score):
        if score >= 80:   return CRITIQUE
        if score >= 60:   return MODERE
        if score >= 40:   return FAIBLE
        return NORMAL

    result["ML_Severity"] = result["ML_Score"].apply(_sev)

    # ── Flag anomalie combinée ───────────────────────────────────────────────
    if has_if and has_ae:
        result["ML_IsAnomaly"] = (
            (result.get("IF_IsAnomaly", 0) == 1) |
            (result.get("AE_IsAnomaly", 0) == 1)
        ).astype(int)
    elif has_if:
        result["ML_IsAnomaly"] = result.get("IF_IsAnomaly", 0)
    elif has_ae:
        result["ML_IsAnomaly"] = result.get("AE_IsAnomaly", 0)
    else:
        result["ML_IsAnomaly"] = 0

    return result


def get_ml_anomalies(
    df: pd.DataFrame,
    level: str,
    ticker_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Extrait les lignes flaggées comme anomalies ML avec leur contexte.

    Retourne un DataFrame structuré comme les anomalies statistiques (Phase 3)
    pour intégration dans le dashboard.
    """
    if "ML_IsAnomaly" not in df.columns or "ML_Score" not in df.columns:
        return pd.DataFrame()

    anom = df[df["ML_IsAnomaly"] == 1].copy()
    if anom.empty:
        return pd.DataFrame()

    date_col   = next((c for c in ["Jour", "Date"] if c in anom.columns), None)
    ticker_col = ticker_col or next(
        (c for c in ["Ticker", "Valeur", "Instrument"] if c in anom.columns), None
    )

    rows = []
    for _, row in anom.iterrows():
        r = {
            "Niveau":          level,
            "Methode_ML":      "IF+AE" if "AE_ReconError" in df.columns and "IF_Score" in df.columns
                               else ("IF" if "IF_Score" in df.columns else "AE"),
            "ML_Score":        row.get("ML_Score", 0),
            "IF_Score":        row.get("IF_Score", np.nan),
            "AE_ReconError":   row.get("AE_ReconError", np.nan),
            "IF_Severity":     row.get("IF_Severity", ""),
            "AE_Severity":     row.get("AE_Severity", ""),
            "ML_Severity":     row.get("ML_Severity", NORMAL),
        }
        if date_col:
            r["Jour"] = row[date_col]
        if ticker_col:
            r["Ticker"] = row[ticker_col]
        rows.append(r)

    return pd.DataFrame(rows).sort_values("ML_Score", ascending=False)


def compare_methods(
    df_stat: pd.DataFrame,
    df_ml: pd.DataFrame,
    join_on: list[str] = ["Jour", "Ticker"],
) -> pd.DataFrame:
    """
    Compare les anomalies statistiques (Phase 3) vs ML (Phase 5).
    Retourne un tableau de synthèse par date/ticker avec :
      - Stat_Anomaly (0/1) — détecté par méthode statistique
      - ML_Anomaly   (0/1) — détecté par ML
      - Consensus    (0/1) — détecté par les deux
    """
    join_stat = [c for c in join_on if c in df_stat.columns]
    join_ml   = [c for c in join_on if c in df_ml.columns]

    if not join_stat or not join_ml:
        return pd.DataFrame()

    left  = df_stat[join_stat + ["Severite_Finale"]].drop_duplicates().copy()
    right = df_ml[join_ml   + ["ML_Severity", "ML_Score"]].drop_duplicates().copy()

    left["Stat_Anomaly"] = (
        left["Severite_Finale"].isin([MODERE, CRITIQUE])
    ).astype(int)
    right["ML_Anomaly"] = (
        right["ML_Severity"].isin([MODERE, CRITIQUE])
    ).astype(int)

    merged = pd.merge(
        left[join_stat + ["Stat_Anomaly"]],
        right[join_ml  + ["ML_Anomaly", "ML_Score"]],
        on=[c for c in join_on if c in left.columns and c in right.columns],
        how="outer",
    ).fillna(0)

    merged["Stat_Anomaly"] = merged["Stat_Anomaly"].astype(int)
    merged["ML_Anomaly"]   = merged["ML_Anomaly"].astype(int)
    merged["Consensus"]    = (
        (merged["Stat_Anomaly"] == 1) & (merged["ML_Anomaly"] == 1)
    ).astype(int)

    return merged.sort_values("ML_Score", ascending=False)
