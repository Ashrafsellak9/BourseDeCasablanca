# Surveillance intelligente — Bourse de Casablanca (BVC)

Application **Streamlit** de tableau de bord et d’analyse : indicateurs de marché, instruments, alertes statistiques, import Excel, détection d’anomalies par **machine learning** (Isolation Forest, autoencodeur).

---

## Prérequis

- **Python** 3.11 ou supérieur (le dépôt a été utilisé avec Python 3.13 sous Anaconda).
- Fichier Excel de référence **`DATASET-2025.xlsx`** (ou équivalent strictement au même format de feuilles / colonnes) à la racine du projet ou chemin attendu par les notebooks de préparation.
- Après exécution du pipeline de données, le dossier **`data/`** doit contenir les **fichiers Parquet** et CSV attendus par l’application (voir section *Données*).

---

## Installation

À la racine du dépôt :

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Sous Linux ou macOS : `source .venv/bin/activate` à la place de l’activation Windows.

---

## Données

L’application charge principalement les artefacts dans **`data/`**, notamment :

| Fichier | Rôle |
|---------|------|
| `market_indicators.parquet` | Séries marché (MASI, volumes, stress, etc.) |
| `instrument_indicators.parquet` | Données par titre |
| `orderflow_indicators.parquet` | Flux d’ordres agrégés |
| `alert_scores.parquet` | Scores d’alerte |
| `all_anomalies.parquet` | Anomalies consolidées |
| `seuils_phase3.csv` | Seuils pour la phase statistique |

Ces fichiers sont en principe **produits par les notebooks** du dossier `notebooks/` à partir du fichier Excel (ordre logique typique : exploration → indicateurs → détection statistique ; le notebook **04** pour le ML génère les fichiers `*_ml.parquet` utilisés par la page « Détection ML »).

Si les Parquet sont absents, l’application affichera une erreur au chargement. La page **Import des données** permet de charger un nouveau fichier **.xlsx** au bon format et d’exécuter le pipeline en session.

---

## Configuration optionnelle (`.env`)

Pour les **notifications** (e-mail / Slack) sur les alertes critiques :

1. Copier **`.env.example`** vers **`.env`** à la racine du projet.
2. Renseigner les variables (SMTP, destinataires, webhook Slack, etc.) selon les commentaires dans `.env.example`.

Sans `.env`, l’application fonctionne pour la consultation des tableaux de bord ; seules les fonctions d’envoi d’alertes restent inactives ou en mode défaut.

---

## Lancement de l’application

### Windows

Double-cliquer sur **`run_app.bat`** (ajuste le chemin vers `streamlit` si Anaconda n’est pas installé au même endroit que dans le script).

### Ligne de commande

```bash
cd app
streamlit run Accueil.py --server.port 8501
```

Puis ouvrir l’URL indiquée (souvent `http://localhost:8501`).

---

## Navigation (menu latéral)

| Entrée du menu | Contenu principal |
|----------------|-------------------|
| **Accueil** | Synthèse, stress marché, graphiques, alertes critiques récentes |
| **Marche BVC** | Vue marché agrégée (MASI, volumes, etc.) |
| **Instruments** | Analyse par titre |
| **Alertes** | Centre d’alertes et filtres |
| **Import donnees** | Import Excel et pipeline automatique |
| **Detection ML** | Scores et anomalies ML |

Les libellés exacts du menu peuvent légèrement varier selon la version de Streamlit (espaces à partir des noms de fichiers dans `app/pages/`).

---

## Structure utile du dépôt

```
app/
  Accueil.py          # Point d’entrée Streamlit
  pages/              # Pages multipages
  utils/              # Cache, graphiques, thème UI
src/                  # Chargement données, indicateurs, anomalies, ML, notifications
data/                 # Données générées (Parquet, CSV)
notebooks/            # Chaîne de traitement et modèles
docs/                 # Diagrammes et documentation projet
rapport/              # Sources du rapport LaTeX
```

---

## Livraison et validation

- **Tests automatisés** : le dépôt ne contient pas de suite de tests unitaires ; la validation repose sur des **parcours manuels** des pages après installation et mise en place des données.
- **Rapport** : voir `rapport/Rapport_PFE_BVC.tex` et `docs/diagrammes_projet.md` pour la description fonctionnelle et l’architecture.

Pour toute démonstration auprès d’une encadrante, prévoir : environnement installé, dossier `data/` à jour (ou import Excel de démonstration), et éventuellement captures d’écran des écrans clés.
