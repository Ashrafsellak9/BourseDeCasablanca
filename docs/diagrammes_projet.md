# Diagrammes — Surveillance intelligente BVC (PFE)

Ce document regroupe **quatre vues** du projet : classes (UML simplifié), cas d’utilisation, architecture et planning Gantt.  
Les blocs sont au format **[Mermaid](https://mermaid.js.org/)** : rendu natif dans GitHub, GitLab, Cursor, Obsidian ; export possible via [Mermaid Live Editor](https://mermaid.live) (PNG/SVG).

---

## 1. Diagramme de classes (UML)

Vue **logique** des principaux modules Python et de leurs relations (agrégation / dépendance). Les « classes » correspondent aux **fichiers / responsabilités** du dépôt, pas à chaque fonction.

```mermaid
classDiagram
    direction TB

    class StreamlitApp

    class DataCache {
        +load_base_data()
        +process_uploaded_file()
    }

    class Charts {
        +chart_masi()
        +chart_instrument_profile()
        +chart_anomalies_chronology()
    }

    class DataLoader {
        +load_excel()
        +validate_format()
        +save_to_cache()
        +get_summary()
    }

    class Indicators {
        +compute_market_indicators()
        +compute_instrument_indicators()
        +compute_orderflow_indicators()
        +compute_spoofing_orderflow_features()
        +enrich_orderflow_spoofing()
        +compute_alert_scores()
        +enrich_market_masi_msi20_rolling_corr()
        +enrich_market_historic_var()
        +enrich_instrument_sector_peer_volatility()
        +compute_advance_decline_line()
        +daily_top5_volume_concentration()
    }

    class AnomalyDetector {
        +detect_market_anomalies()
        +detect_instrument_anomalies()
        +detect_orderflow_anomalies()
        +consolidate_all_anomalies()
        +compute_seuils_table()
        +summarize_anomalies()
        +detect_zscore()
        +detect_multi_method()
    }

    class MLModels {
        +isolation_forest_fit()
        +autoencoder_fit()
    }

    class MarketStress {
        +enrich_market_stress_score()
        +stress_summary_latest()
    }

    class InstrumentSegment {
        +enrich_market_with_segment_volumes()
    }

    class MacroEvents {
        +load_macro_events()
        +filter_macro_events_for_period()
    }

    class VolumeForecast {
        +fit_volume_forecast()
    }

    class Notifications {
        +from_env()
        +send_email()
        +send_slack()
    }

    class CriticalAlertNotifier {
        +notify_new_critical_alerts()
        +notification_status_summary()
    }

    class DataStore

    StreamlitApp --> DataCache
    StreamlitApp --> Charts
    DataCache --> DataLoader
    DataCache --> Indicators
    DataCache --> AnomalyDetector
    DataCache --> MarketStress
    DataCache --> InstrumentSegment
    Indicators --> DataStore
    AnomalyDetector --> DataStore
    StreamlitApp --> MacroEvents
    StreamlitApp --> VolumeForecast
    StreamlitApp --> MLModels
    CriticalAlertNotifier --> Notifications
    StreamlitApp --> CriticalAlertNotifier
    MarketStress ..> Indicators : OIR agrégé
```

---

## 2. Diagramme de cas d’utilisation (Use Case)

**Acteurs** : analyste / décideur marché, gestionnaire de risques, administrateur technique (optionnel).

```mermaid
flowchart TB
    subgraph ACT["Acteurs"]
        A1[Analyste marche]
        A2[Analyste instrument risque]
        A3[Responsable technique donnees]
    end

    subgraph SYST["Systeme Surveillance BVC"]
        UC1["UC1: Vue ensemble marche"]
        UC2["UC2: Stress marche et qualite"]
        UC3["UC3: Graphiques MASI volume macro"]
        UC4["UC4: Profil instrument liquidite"]
        UC5["UC5: Flux ordres OIR OAR VWAP"]
        UC6["UC6: Proxy motifs spoofing"]
        UC7["UC7: Anomalies statistiques"]
        UC8["UC8: Scores alerte"]
        UC9["UC9: Notifications critiques"]
        UC10["UC10: Anomalies ML"]
        UC11["UC11: Upload Excel pipeline"]
        UC12["UC12: Export donnees seuils"]
        UC13["UC13: Prevision volume marche"]
    end

    A1 --> UC1
    A1 --> UC2
    A1 --> UC3
    A1 --> UC7
    A1 --> UC8
    A1 --> UC10

    A2 --> UC4
    A2 --> UC5
    A2 --> UC6
    A2 --> UC7
    A2 --> UC8

    A3 --> UC11
    A3 --> UC12
    A1 --> UC9
    A2 --> UC9

    A1 --> UC13
```

**Tableau de traçabilité (résumé)**

| UC   | Description courte                         | Pages / modules principaux      |
|------|---------------------------------------------|-----------------------------------|
| UC1–3 | Tableau de bord, stress, MASI, macro       | `app.py`, `1_Marche_Global.py`, `charts.py`, `market_stress.py`, `macro_events.py` |
| UC4–6 | Instrument, orderflow, spoofing            | `2_Instruments.py`, `indicators.py` |
| UC7–9 | Anomalies, alertes, notifications          | `3_Alertes.py`, `anomaly_detector.py`, `critical_alert_notifier.py`, `notifications.py` |
| UC10 | ML                                         | `5_Machine_Learning.py`, `ml_models.py`, notebooks `04_*` |
| UC11 | Upload                                     | `4_Upload_Excel.py`, `data_cache.process_uploaded_file` |
| UC12 | Données / seuils                           | Parquets, `seuils_phase3.csv`, exports notebook |
| UC13 | Prévision volume                           | `1_Marche_Global.py`, `volume_forecast.py` |

---

## 3. Diagramme d’architecture

Vue **multicouche** : présentation, orchestration cache, domaine métier, persistance.

```mermaid
flowchart TB
    subgraph PRES["Presentation Streamlit"]
        APP[app.py accueil]
        P1[1_Marche_Global]
        P2[2_Instruments]
        P3[3_Alertes]
        P4[4_Upload_Excel]
        P5[5_Machine_Learning]
        NAV[streamlit_nav]
        CH[charts Plotly]
    end

    subgraph ORCH["Orchestration cache"]
        DC[data_cache load_base_data]
    end

    subgraph DOM["Domaine src"]
        DL[data_loader]
        IND[indicators]
        ANO[anomaly_detector]
        ML[ml_models]
        MS[market_stress]
        SEG[instrument_segment]
        MACRO[macro_events]
        VF[volume_forecast]
        NOTIF[notifications]
        CRIT[critical_alert_notifier]
    end

    subgraph DATA["Persistance"]
        XLS[(Excel DATASET)]
        PQ[(Parquet)]
        CSV[(CSV seuils ref)]
    end

    APP --> DC
    P1 --> DC
    P2 --> DC
    P3 --> DC
    P4 --> DC
    P5 --> DC
    P1 --> CH
    P2 --> CH
    P3 --> CH
    APP --> CH

    DC --> DL
    DC --> IND
    DC --> ANO
    DC --> MS
    DC --> SEG
    DC --> PQ
    DC --> CSV

    P4 --> DL
    DL --> XLS
    IND --> PQ
    ANO --> PQ
    P5 --> PQ
    P1 --> VF
    P1 --> MACRO
    APP --> MACRO
    P3 --> CRIT
    CRIT --> NOTIF
```

**Flux principal (données figées)**

```mermaid
sequenceDiagram
    participant U as Utilisateur
    participant ST as Streamlit
    participant DC as data_cache
    participant PQ as Parquet CSV
    participant IND as indicators
    participant ANO as anomaly_detector

    U->>ST: Ouvre application
    ST->>DC: load_base_data()
    DC->>PQ: read market / instrument / orderflow / alerts / anomalies / seuils
    PQ-->>DC: DataFrames
    DC->>IND: enrichissements si colonnes manquantes
    IND-->>DC: DataFrames enrichis
    DC-->>ST: dict données
    ST-->>U: Dashboard + pages
```

---

## 4. Diagramme de Gantt (planning type PFE)

Planning **indicatif** sur ~5 mois (à adapter aux dates réelles de ton stage). Les phases reflètent les notebooks et livrables du dépôt.

```mermaid
gantt
    title Planning type PFE Surveillance BVC
    dateFormat  YYYY-MM-DD
    axisFormat  %b

    section Cadrage
    Analyse du besoin et périmètre données     :a1, 2025-09-01, 14d
    Revue DATASET Excel et qualité données    :a2, after a1, 10d

    section Fondations données
    Exploration statistique notebook 01       :b1, after a2, 12d
    Spécification schémas et cache Parquet      :b2, after b1, 7d

    section Indicateurs
    Indicateurs marché et instruments n02    :c1, after b2, 21d
    Flux ordres intraday VWAP OIR OAR        :c2, after c1, 14d
    Stress marché segments sectorisation     :c3, after c2, 10d

    section Detection
    Anomalies statistiques phase 3 n03        :d1, after c3, 14d
    Scores alertes et consolidation            :d2, after d1, 7d

    section ML et motifs avancés
    Entraînement ML notebook 04                :e1, after d2, 14d
    Heuristique spoofing proxy orderflow     :e2, after e1, 7d

    section Application
    UI Streamlit pages et graphiques           :f1, after c1, 35d
    Upload Excel et pipeline bout en bout      :f2, after f1, 7d
    Notifications alertes critiques            :f3, after d2, 10d

    section Qualité et livrables
    Tests intégration et jeux de données       :g1, after f2, 10d
    Redaction rapport LaTeX                    :g2, after g1, 21d
    Soutenance                                 :milestone, ms1, after g2, 0d
```

Pour coller à ton calendrier réel, modifie les dates (`2025-09-01`, durées `14d`, etc.) directement dans le bloc `gantt` ci-dessus.

---

## Utilisation rapide

1. **Prévisualisation** : ouvre ce fichier dans un éditeur compatible Mermaid (Cursor, VS Code + extension Mermaid).
2. **Export image** : copie un bloc dans [mermaid.live](https://mermaid.live) → Export PNG/SVG.
3. **LaTeX** : importe les SVG générés dans `rapport/` avec `\includegraphics`.

---

*Document généré pour le projet BourseDeCasablancaPFE — cohérent avec la structure du dépôt au moment de la rédaction.*
