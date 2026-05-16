# Soutenance PFE -- Presentation LaTeX

Fichier principal :

```bash
soutenance/presentation_pfe_bvc.tex
```

Contenu : deck Beamer en 14 slides, aligne sur le conducteur de soutenance
fourni (page de garde, plan, 6 parties, conclusion).

Compilation depuis la racine du projet :

```bash
pdflatex -interaction=nonstopmode -output-directory=soutenance soutenance/presentation_pfe_bvc.tex
pdflatex -interaction=nonstopmode -output-directory=soutenance soutenance/presentation_pfe_bvc.tex
```

Les figures analytiques sont chargees depuis `reports/`.
Les captures de l'application sont copiees dans `soutenance/screens/` avec des
noms simples pour eviter les problemes LaTeX lies aux espaces et accents :

- `accueil.jpeg`
- `alertes_heatmap.jpeg`
- `ml_global.jpeg`
- `instrument.jpeg`

Les logos fournis sont dans :

- `soutenance/bvc_logo_user.png`
- `soutenance/ehtp_logo.png`

Si LaTeX n'est pas installe, installer MiKTeX ou TeX Live puis relancer les
commandes ci-dessus.
