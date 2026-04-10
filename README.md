## Environnement de développement

### Prerequis

- **Python** : 3.10+ (recommande 3.11 ou 3.12).
- **OS** : Linux ou Windows.
- **GPU (optionnel)** : pilote NVIDIA installe et visible via `nvidia-smi`.

## Gestion des dependances

Le projet utilise des fichiers `requirements` par profil pour eviter les conflits CPU/GPU:

- `requirements.txt` -> alias vers `requirements-base.txt`
- `requirements-base.txt` -> dependances communes
- `requirements-tf-cpu.txt` -> profil TensorFlow CPU
- `requirements-tf-gpu.txt` -> profil TensorFlow GPU

### Pourquoi cette separation

`tensorflow-cpu` et `tensorflow[and-cuda]` ne doivent pas etre installes ensemble dans le meme environnement.

## Installation avec venv

### Linux / macOS

```bash
cd /chemin/vers/concept-stat-projet-1
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools wheel
```

Installe ensuite **un seul** profil TensorFlow:

```bash
# Base uniquement
pip install -r requirements.txt

# Profil TensorFlow CPU
pip install -r requirements-tf-cpu.txt

# Profil TensorFlow GPU
pip install -r requirements-tf-gpu.txt
```

### Windows (PowerShell / CMD)

```cmd
cd C:\chemin\vers\concept-stat-projet-1
python -m venv .venv
.venv\Scripts\activate
pip install -U pip setuptools wheel
```

Puis installe un seul profil:

```cmd
pip install -r requirements.txt
pip install -r requirements-tf-cpu.txt
```

Pour Windows + GPU, suis la documentation officielle TensorFlow/NVIDIA pour les bibliotheques CUDA/cuDNN compatibles.

## Activation runtime TensorFlow GPU (Linux)

Si TensorFlow est installe avec `requirements-tf-gpu.txt` mais que `GPU: []` apparait, charge les chemins de bibliotheques du venv:

```bash
source .venv/bin/activate
. ./scripts/tf_gpu_env.sh
```

Pour fish:

```fish
source .venv/bin/activate.fish
source ./scripts/tf_gpu_env.fish
```

## Docker (recommande)

L'approche la plus fiable pour utiliser TensorFlow est via Docker.
Deux services sont disponibles : **GPU** et **CPU** (fallback optimise).

### Prerequis Docker

- Docker Engine installe (`docker --version`)
- Pour le service GPU : NVIDIA Container Toolkit (`nvidia-ctk --version`) et pilote NVIDIA (`nvidia-smi`)

### Service GPU

Le conteneur embarque CUDA et cuDNN pre-configures ; seul le pilote NVIDIA de l'hote est requis.

```bash
docker compose up jupyter-gpu --build
```

Jupyter GPU accessible sur **http://localhost:8889** (sans token).

### Service CPU (fallback AVX512 / oneDNN)

Si le GPU n'est pas disponible, le service CPU exploite automatiquement les meilleurs jeux d'instructions
du processeur (AVX2, AVX512F, AVX512_VNNI, AVX512_BF16, FMA) via oneDNN.

```bash
docker compose up jupyter-cpu --build
```

Jupyter CPU accessible sur **http://localhost:8890** (sans token).

### Verification

```bash
# GPU
docker compose exec jupyter-gpu python -c \
  "import tensorflow as tf; print('GPUs:', tf.config.list_physical_devices('GPU'))"

# CPU — doit afficher les optimisations oneDNN au demarrage
docker compose exec jupyter-cpu python -c \
  "import tensorflow as tf; print('TF:', tf.__version__)"
```

### Arreter les conteneurs

```bash
docker compose down
```

## Verification rapide (hors Docker)

```bash
python -c "import tensorflow as tf; print('TF:', tf.__version__); print('Built with CUDA:', tf.test.is_built_with_cuda()); print('GPUs:', tf.config.list_physical_devices('GPU'))"
```

Si la liste `GPUs` est vide, verifie:

- le pilote NVIDIA (`nvidia-smi`)
- l'environnement Python actif (`which python`)
- le profil installe (`requirements-tf-gpu.txt` vs CPU)
