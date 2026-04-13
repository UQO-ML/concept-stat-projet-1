# concept-stat-projet-1

Projet de cours en statistique / apprentissage automatique centré sur la reproduction de l'article:
**"Classification of Firewall Log Files with Multiclass Support Vector Machine"** (Ertam & Kaya, 2018).

Le depot contient:
- un pipeline Python reproductible (`firewall_svm.py`);
- un notebook principal d'analyse (`Projet.ipynb`);
- des utilitaires de visualisation, d'interpretation et de suivi des runs (`code/`);
- des notebooks de devoirs (`Devoir partie 1`, `Devoir partie 2`);
- une option d'execution Docker GPU.

## Vue d'ensemble du projet

L'objectif est de comparer 4 noyaux SVM en classification multiclasse sur les actions pare-feu:
- `linear`
- `poly`
- `rbf`
- `sigmoid`

Le pipeline produit:
- Precision, Recall, F1 (moyenne macro, en %);
- rapport de classification par classe;
- matrice de confusion;
- courbes ROC;
- comparaison experimentale vs valeurs de reference de la Table III de l'article.

## Structure utile

```text
concept-stat-projet-1/
├── firewall_svm.py                 # pipeline principal CLI (runs horodates)
├── Projet.ipynb                    # notebook principal (reproduction + analyse)
├── log2.csv                        # dataset firewall (source locale)
├── code/
│   ├── firewall_visualization.py   # matrices, ROC, graphiques comparatifs
│   ├── metric_interpretations.py   # interpretation metriques + comparaison article
│   ├── hyperparam_advisor.py       # aide a l'analyse hyperparametres SVM
│   └── run_timestamps.py           # suivi fraicheur / manifests run-*
├── run-YYYY.../                    # artefacts generes par `firewall_svm.py`
├── Devoir partie 1/
├── Devoir partie 2/
├── Dockerfile
├── docker-compose.yml
├── requirements-base.txt
└── requirements-tf-gpu.txt
```

## Installation (local CPU)

Prerequis:
- Python 3.10+ (3.11 recommande)
- `pip` a jour

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
pip install -r requirements-base.txt
```

## Lancer le pipeline principal

Depuis la racine du depot:

```bash
python firewall_svm.py
```

Comportement:
- utilise automatiquement `log2.csv` s'il est present;
- sinon bascule sur des donnees simulees (`make_classification`).

Vous pouvez forcer une source:

```bash
python firewall_svm.py log2.csv
# ou
python firewall_svm.py https://.../fichier.csv
```

## Artefacts generes

Chaque execution cree un dossier `run-<timestamp>/` contenant:
- `resultats.txt` : trace complete de l'execution;
- `run_manifest.json` : metadonnees (source, classes, meilleur noyau, F1 par classe, etc.);
- `comparaison_svm.png`;
- `matrice_confusion.png`;
- `roc_allow.png`, `roc_deny.png`, `roc_drop.png`, `roc_reset_both.png`.

Ces fichiers servent de base pour la section "Simulation et resultats" du projet.

## Notebook principal (`Projet.ipynb`)

Le notebook:
- reutilise les fonctions de `firewall_svm.py` (pas de duplication majeure de logique);
- execute les 4 noyaux avec interpretation des metriques;
- compare les resultats obtenus avec la Table III de l'article;
- lit les manifests de runs pour verifier la fraicheur et la coherence des experiences.

Demarrage:

```bash
jupyter lab
```

Puis ouvrir `Projet.ipynb`.

## Option Docker GPU (NVIDIA)

Le service `jupyter-gpu` est defini dans `docker-compose.yml`.

Prerequis:
- Docker + Docker Compose
- pilote NVIDIA fonctionnel (`nvidia-smi`)
- NVIDIA Container Toolkit

Lancement:

```bash
docker compose build jupyter-gpu
docker compose up jupyter-gpu
```

Acces:
- URL: `http://localhost:8889`
- token: `concept-stat`

Arret:

```bash
docker compose down
```

## Notes methodologiques

- Les metriques principales du pipeline sont en **macro** (coherent avec l'analyse multiclasses et la classe rare `reset-both`).
- Le dataset est fortement desequilibre; l'interpretation ne doit pas se limiter a l'accuracy globale.
- Le script inclut une optimisation optionnelle (`GridSearchCV`) du meilleur noyau apres comparaison initiale.

## References

- F. Ertam, M. Kaya, *Classification of Firewall Log Files with Multiclass Support Vector Machine*, 2018.
- UCI / Kaggle Internet Firewall Data Set (11 features numeriques + classe `Action`).
