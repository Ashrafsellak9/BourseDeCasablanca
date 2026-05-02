# Rapport PFE — LaTeX

## Fichier principal
- `Rapport_PFE_BVC.tex` : rapport complet (introduction → conclusion + annexes).

## Compilation
Depuis ce dossier `rapport/` :

```bash
pdflatex Rapport_PFE_BVC.tex
pdflatex Rapport_PFE_BVC.tex
```

Deux passes permettent de résoudre correctement la table des matières et les références croisées (`cleveref`).

## Prérequis
- Distribution LaTeX : **TeX Live**, **MiKTeX** ou **MacTeX**.
- Les figures sont chargées depuis **`../reports/`** (chemins relatifs). Vérifie que les fichiers `.png` existent bien.

## Personnalisation
- Page de garde : noms d’établissement, encadrants, filière.
- Chapitres « Cadre institutionnel » et « Abstract » : compléter les zones *À compléter*.
- Bibliographie : remplacer / compléter les entrées `thebibliography`.

## Option : chapitre équations détaillé
Pour insérer le bloc mathématique long déjà fourni dans le chat, créer `chapitre_equations.tex` et ajouter dans le préambule :
```latex
\input{chapitre_equations.tex}
```
(à placer au bon endroit, par ex. après `\chapter{Méthodologie...}`.)
