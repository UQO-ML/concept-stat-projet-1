## Projet concept-stat-projet-1

Ce README est centre sur `Devoir partie 2` avec 2 parcours:

- GPU avec Docker + NVIDIA (recommande sur Linux NVIDIA)
- CPU avec environnement Python `venv` (Linux, Windows, macOS)

## Structure du depot

```text
concept-stat-projet-1/
├── Devoir partie 1/
├── Devoir partie 2/
│   ├── DEVOIR Partie 2.ipynb
│   ├── DEVOIR Partie 2-gpu.ipynb
│   └── code/
│       └── visualization.py
├── Devoirs ID/
│   └── DEVOIR Partie 2.ipynb
├── Projet ID/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── requirements-base.txt
├── requirements-tf-cpu.txt
├── requirements-tf-gpu.txt
└── scripts/
```

## Prerequis

### Commun

- Python 3.10+ (3.11 recommande)
- `pip` a jour
- Jupyter Notebook ou JupyterLab

### Pour le mode GPU Docker (Linux NVIDIA)

- Docker Engine + Docker Compose
- Pilote NVIDIA fonctionnel (`nvidia-smi`)
- NVIDIA Container Toolkit (`nvidia-container-toolkit` / runtime `nvidia`)

Le service `jupyter-gpu` utilise deja:

- `gpus` avec `driver: nvidia`
- `NVIDIA_VISIBLE_DEVICES=all`

## Guide principal: Devoir partie 2

### Option A - GPU via Docker (NVIDIA)

Le conteneur s'appuie sur l'image TensorFlow NVIDIA (`nvcr.io/nvidia/tensorflow`) et monte le projet dans `/workspace`.

#### Linux (fish)

```fish
cd "/home/kilo/Work/Cours - UQO/concept-statistique/concept-stat-projet-1"
docker compose build jupyter-gpu
docker compose up jupyter-gpu
```

Jupyter sera disponible sur:

- URL: `http://localhost:8889`
- Token: `concept-stat`

Verification GPU depuis un autre terminal:

```fish
docker compose exec jupyter-gpu python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

Arret:

```fish
docker compose down
```

#### Windows (PowerShell)

```powershell
cd "C:\chemin\vers\concept-stat-projet-1"
docker compose build jupyter-gpu
docker compose up jupyter-gpu
```

#### macOS

Docker fonctionne sur macOS, mais le passthrough GPU NVIDIA n'est pas supporte nativement.
Sur macOS, utiliser plutot le parcours CPU avec `venv` ci-dessous.

### Option B - CPU via venv (recommande hors Linux NVIDIA)

Installe uniquement le profil CPU.

#### Linux (fish)

```fish
cd "/home/kilo/Work/Cours - UQO/concept-statistique/concept-stat-projet-1"
python3 -m venv .venv
source .venv/bin/activate.fish
python -m pip install -U pip setuptools wheel
pip install -r requirements.txt
pip install -r requirements-tf-cpu.txt
jupyter lab
```

Ouvrir ensuite:

- `Devoir partie 2/DEVOIR Partie 2.ipynb`
- ou `Devoir partie 2/DEVOIR Partie 2-gpu.ipynb` (execution CPU possible, mais notebook concu pour test GPU)

#### Windows

PowerShell:

```powershell
cd "C:\chemin\vers\concept-stat-projet-1"
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -U pip setuptools wheel
pip install -r requirements.txt
pip install -r requirements-tf-cpu.txt
jupyter lab
```

CMD:

```cmd
cd C:\chemin\vers\concept-stat-projet-1
python -m venv .venv
.venv\Scripts\activate
python -m pip install -U pip setuptools wheel
pip install -r requirements.txt
pip install -r requirements-tf-cpu.txt
jupyter lab
```

#### macOS (zsh/bash)

```bash
cd "/chemin/vers/concept-stat-projet-1"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
pip install -r requirements.txt
pip install -r requirements-tf-cpu.txt
jupyter lab
```

## Autres dossiers: execution avec venv

Pour les dossiers suivants, utiliser le meme `venv` CPU (pas besoin de Docker GPU):

- `Devoir partie 1/`
- `Devoirs ID/`
- `Projet ID/`

Exemple d'ouverture apres activation du `venv`:

```bash
jupyter lab "Devoir partie 1"
jupyter lab "Devoirs ID"
jupyter lab "Projet ID"
```

## Verification rapide

Verifier l'environnement Python actif:

```bash
python -c "import sys; print(sys.executable)"
```

Verifier TensorFlow:

```bash
python -c "import tensorflow as tf; print('TF:', tf.__version__); print('GPUs:', tf.config.list_physical_devices('GPU'))"
```
