# Rapport PFE — LaTeX

## Fichier principal
- `Rapport_PFE_BVC.tex` : rapport complet (introduction → conclusion + annexes).

## Compilation
Depuis ce dossier `rapport/` :

```bash
pdflatex Rapport_PFE_BVC.tex
pdflatex Rapport_PFE_BVC.tex
```

Deux passes permettent de résoudre correctement la table des matières et les références croisées.

Si LaTeX affiche encore une erreur après une modification des références, supprimer les fichiers auxiliaires générés (`.aux`, `.toc`, `.lof`, `.lot`, `.out`, `.log`) puis relancer deux compilations.

## Prérequis
- Distribution LaTeX : **TeX Live**, **MiKTeX** ou **MacTeX**.
- Les figures analytiques sont chargées depuis **`../reports/`**.
- Les logos et captures d'écran intégrés au rapport sont dans **`rapport/assets/`**.

## Personnalisation
- La page de garde est déjà renseignée avec EHTP, la filière SIG, les encadrants et les deux logos.
- Les captures principales de l'application sont intégrées dans le chapitre « Réalisation informatique ».
- Les résultats chiffrés ont été ajoutés dans le chapitre « Résultats et discussion ».
- La bibliographie peut être complétée si l'encadrant demande des références académiques supplémentaires.

## Assets ajoutés
- `assets/logo_bvc.png`
- `assets/logo_ehtp.png`
- `assets/screen_accueil.jpeg`
- `assets/screen_instrument.jpeg`
- `assets/screen_alertes_heatmap.jpeg`
- `assets/screen_ml_global.jpeg`
- `assets/screen_import.jpeg`
