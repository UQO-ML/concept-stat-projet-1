# ============================================================
# Classification des actions d'un pare-feu avec SVM Multiclasse
# Basé sur : F. Ertam and M. Kaya (2018)
# Dataset : UCI - Internet Firewall Data
# https://www.kaggle.com/datasets/tunguz/internet-firewall-data-set
# ============================================================

# ==============================================================
# INSTALLATION DES LIBRAIRIES
# pip install -r requirement.txt
# ==============================================================
import glob
import os
import sys
import warnings

_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
_CODE_DIR = os.path.join(_ROOT_DIR, "code")
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import SVC

import firewall_visualization as _fw_viz

warnings.filterwarnings("ignore")

# ============================================================
# CONSTANTES GLOBALES
# ============================================================
FEATURE_NAMES = [
    "Source Port",
    "Destination Port",
    "NAT Source Port",
    "NAT Destination Port",
    "Elapsed Time (sec)",
    "Bytes",
    "Bytes Sent",
    "Bytes Received",
    "Packets",
    "pkts_sent",
    "pkts_received",
]
CLASS_NAMES = ["allow", "deny", "drop", "reset-both"]
RANDOM_STATE = 69

# Ordre des noyaux (identique à l'article : Linear, Polynomial, RBF, Sigmoid)
KERNEL_KEYS = ("linear", "poly", "rbf", "sigmoid")

KERNEL_DISPLAY_NAMES = {
    "linear": "SVM Linear",
    "poly": "SVM Polynomial",
    "rbf": "SVM RBF",
    "sigmoid": "SVM Sigmoid",
}

# Table III — F. Ertam & M. Kaya, valeurs numériques exactes de l'article (macro %)
ARTICLE_TABLE_III = {
    "SVM Linear": {"F1 Score": 75.4, "Precision": 67.5, "Recall": 85.3},
    "SVM Polynomial": {"F1 Score": 53.6, "Precision": 61.8, "Recall": 47.4},
    "SVM RBF": {"F1 Score": 76.4, "Precision": 63.0, "Recall": 97.1},
    "SVM Sigmoid": {"F1 Score": 74.8, "Precision": 60.3, "Recall": 98.5},
}


# ============================================================
# LOG (notebook : passer log=None pour utiliser print)
# ============================================================
def _print_log(msg: str = "") -> None:
    print(msg)


def _ensure_log(log):
    return log if log is not None else _print_log


# ============================================================
# UTILITAIRES : dossier de run + logging
# ============================================================
def create_run_folder():
    """
    Crée un dossier run-001, run-002, ... selon les runs déjà existants.
    Retourne le chemin du nouveau dossier.
    """
    existing = sorted(glob.glob("run-[0-9][0-9][0-9]"))
    next_id = len(existing) + 1
    folder = f"run-{next_id:03d}"
    os.makedirs(folder, exist_ok=True)
    return folder


def make_logger(run_folder):
    """
    Retourne une fonction log(msg) qui affiche dans le terminal et écrit dans run_folder/resultats.txt.
    """

    log_path = os.path.join(run_folder, "resultats.txt")
    log_file = open(log_path, "w", encoding="utf-8")

    def log(msg=""):
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    log.close = log_file.close
    return log


# ============================================================
# ÉTAPE 1 — Chargement des données
# ============================================================
def load_data(log, source=None):
    """
    Charge les données depuis :
      - source=None → données simulées (make_classification)
      - source="path/to.csv" → fichier CSV local
      - source="https://..." → URL vers un fichier CSV

    Le fichier CSV doit contenir les 11 features numériques et une colonne 'Action' avec les valeurs :
    allow / deny / drop / reset-both.

    Retourne un DataFrame df avec les features + colonne 'Action'.
    """
    log = _ensure_log(log)
    log("=" * 55)
    log("ÉTAPE 1 : Chargement des données")
    log("=" * 55)

    if source is None:
        log("Source : données simulées (make_classification)")
        N = 65532
        x_raw, y_raw = make_classification(
            n_samples=N,
            n_features=11,
            n_informative=8,
            n_redundant=2,
            n_classes=4,
            n_clusters_per_class=2,
            weights=[0.6, 0.2, 0.15, 0.05],
            random_state=RANDOM_STATE,
        )
        df = pd.DataFrame(x_raw, columns=FEATURE_NAMES)
        df["Action"] = [CLASS_NAMES[i] for i in y_raw]

    else:
        log(f"Source : {source}")
        df = pd.read_csv(source)

        missing = [c for c in FEATURE_NAMES + ["Action"] if c not in df.columns]
        if missing:
            raise ValueError(
                f"Colonnes manquantes dans le fichier : {missing}\n"
                f"Colonnes trouvées : {list(df.columns)}"
            )
        df = df[FEATURE_NAMES + ["Action"]].copy()
        df = df[df["Action"].isin(CLASS_NAMES)].reset_index(drop=True)

    log(f"Instances : {len(df)}")
    log(f"Features : {len(FEATURE_NAMES)}")
    log("\nDistribution des classes :")
    for line in df["Action"].value_counts().to_string().split("\n"):
        log(f"{line}")

    return df


# ============================================================
# ÉTAPE 2 — Sélection des features
# ============================================================
def select_features(log, df):
    """
    Sépare les features (x) de la variable cible (y).
    Encode y en entiers (0-3).
    Retourne x (array), y_encoded (array).
    """
    log = _ensure_log(log)
    log("\n" + "=" * 55)
    log("ÉTAPE 2 : Sélection des features")
    log("=" * 55)

    x = df[FEATURE_NAMES].values
    y = df["Action"].values
    y_encoded = np.array([CLASS_NAMES.index(c) for c in y])

    log(f"{len(FEATURE_NAMES)} features numériques sélectionnées")
    log(f"Variable cible : 'Action' ({len(CLASS_NAMES)} classes)")
    log("Données personnelles : exclues")

    return x, y_encoded


# ============================================================
# ÉTAPE 3 — Préparation des données
# ============================================================
def prepare_data(log, x, y_encoded):
    """
    Sépare en train (80%) / test (20%) et normalise avec StandardScaler.
    Retourne x_train, x_test, y_train, y_test.
    """
    log = _ensure_log(log)
    log("\n" + "=" * 55)
    log("ÉTAPE 3 : Préparation des données")
    log("=" * 55)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y_encoded,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_encoded,
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    log(f"Train : {len(x_train)} instances (80%)")
    log(f"Test : {len(x_test)} instances (20%)")
    log("Normalisation : StandardScaler appliquée")

    return x_train, x_test, y_train, y_test


# ============================================================
# Pipeline SVM — entraînement / évaluation (métrique macro)
# ============================================================
def build_svc_pipeline(kernel: str, random_state: int = RANDOM_STATE, **svc_overrides) -> Pipeline:
    """Pipeline StandardScaler + SVC, aligné sur l'article (4 noyaux) et le notebook."""
    params: dict = {
        "kernel": kernel,
        "probability": True,
        "random_state": random_state,
        "class_weight": "balanced",
    }
    if kernel == "poly":
        params["degree"] = svc_overrides.pop("degree", 3)
    params.update(svc_overrides)
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svm", SVC(**params)),
        ]
    )


def train_eval_kernel(
    kernel_key: str,
    x_train,
    y_train,
    x_test,
    y_test,
    random_state: int = RANDOM_STATE,
    **svc_overrides,
):
    """
    Entraîne un SVM pour un seul noyau (linear | poly | rbf | sigmoid).
    Retourne (pipeline, y_pred, metrics_dict) avec Precision, Recall, F1 Score en % (arrondi 0.1), moyenne macro.
    """
    if kernel_key not in KERNEL_DISPLAY_NAMES:
        raise ValueError(f"Noyau inconnu : {kernel_key!r}. Attendu : {list(KERNEL_DISPLAY_NAMES)}")
    pipeline = build_svc_pipeline(kernel_key, random_state=random_state, **svc_overrides)
    pipeline.fit(x_train, y_train)
    y_pred = pipeline.predict(x_test)
    p = precision_score(y_test, y_pred, average="macro", zero_division=0) * 100
    r = recall_score(y_test, y_pred, average="macro", zero_division=0) * 100
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0) * 100
    metrics = {
        "F1 Score": round(f1, 1),
        "Precision": round(p, 1),
        "Recall": round(r, 1),
    }
    return pipeline, y_pred, metrics


# ============================================================
# ÉTAPE 4 — Entraînement des 4 SVM
# ============================================================
def train_svms(log, x_train, y_train, x_test, y_test):
    """
    Entraîne les 4 classifieurs SVM (Linear, Poly, RBF, Sigmoid).
    Calcule F1, Précision et Rappel macro pour chacun.
    Retourne results (dict), models (dict), y_preds (dict).
    """
    log = _ensure_log(log)
    log("\n" + "=" * 55)
    log("ÉTAPE 4 : Entraînement des 4 classifieurs SVM")
    log("=" * 55)

    results = {}
    models = {}
    y_preds = {}

    for key in KERNEL_KEYS:
        name = KERNEL_DISPLAY_NAMES[key]
        log(f"\n {name} : ")
        model, y_pred, m = train_eval_kernel(key, x_train, y_train, x_test, y_test)
        results[name] = m
        models[name] = model
        y_preds[name] = y_pred
        log(f"F1={m['F1 Score']:.1f}%  Précision={m['Precision']:.1f}%  Rappel={m['Recall']:.1f}%")

    return results, models, y_preds


# ============================================================
# ÉTAPE 5 — Tableau comparatif
# ============================================================
def compare_results(log, results):
    """
    Affiche le tableau comparatif des 4 kernels et retourne le nom du meilleur modèle selon le F1 Score.
    """
    log = _ensure_log(log)
    log("\n" + "=" * 55)
    log("ÉTAPE 5 : Tableau comparatif des performances")
    log("=" * 55)

    results_df = pd.DataFrame(results).T
    log("\n" + results_df.to_string())

    best_name = results_df["F1 Score"].idxmax()
    log(
        f"\n  Meilleur modèle (F1) : {best_name} "
        f"— F1={results_df.loc[best_name, 'F1 Score']}%  "
        f"Précision={results_df.loc[best_name, 'Precision']}%  "
        f"Rappel={results_df.loc[best_name, 'Recall']}%"
    )

    return results_df, best_name


# ============================================================
# ÉTAPE 6 — Rapport détaillé du meilleur modèle
# ============================================================
def detailed_report(log, best_name, y_test, y_preds):
    """
    Affiche le rapport de classification complet (précision, rappel, F1 par classe) pour le meilleur modèle.
    """
    log = _ensure_log(log)
    log("\n" + "=" * 55)
    log(f"ÉTAPE 6 : Rapport détaillé — {best_name}")
    log("=" * 55)

    report = classification_report(
        y_test,
        y_preds[best_name],
        target_names=CLASS_NAMES,
    )
    log(report)


# ============================================================
# ÉTAPE 7 — Graphique comparatif
# ============================================================
def plot_comparison(log, run_folder, results_df):
    """Génère un graphique en barres comparant F1, Précision et Rappel pour les 4 kernels."""
    log = _ensure_log(log)
    path = os.path.join(run_folder, "comparaison_svm.png")
    _fw_viz.plot_comparison_bar(results_df, save_path=path)
    log(f"Sauvegardé : {path}")


# ============================================================
# ÉTAPE 8 — Matrice de confusion
# ============================================================
def plot_confusion_matrix(log, run_folder, best_name, y_test, y_preds):
    """Génère la matrice de confusion pour le meilleur modèle."""
    log = _ensure_log(log)
    y_pred = y_preds[best_name]
    path = os.path.join(run_folder, "matrice_confusion.png")
    _fw_viz.plot_confusion_kernel(
        y_test,
        y_pred,
        CLASS_NAMES,
        f"Matrice de confusion — {best_name}",
        save_path=path,
    )
    log(f"Sauvegardé : {path}")


# ============================================================
# ÉTAPE 9 — Courbes ROC
# ============================================================
def plot_roc_curves(log, run_folder, models, x_test, y_test):
    """Une figure par classe, toutes les courbes noyaux superposées (comme l'article)."""
    log = _ensure_log(log)
    log("\n" + "=" * 55)
    log("ÉTAPE 9 : Courbes ROC par classe")
    log("=" * 55)

    y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3])
    _fw_viz.plot_roc_all_kernels_per_class(
        models, x_test, y_test_bin, CLASS_NAMES, run_folder, log
    )


# ============================================================
# ÉTAPE 10 — Optimisation des hyperparamètres (GridSearchCV)
# ============================================================
def optimize_best_model(log, best_name, x_train, y_train, x_test, y_test):
    """
    Optimise les hyperparamètres du meilleur modèle via GridSearchCV avec validation croisée à 5 plis.
    Retourne le meilleur estimateur entraîné.
    """
    kernel = best_name.split(" ", 1)[1].lower()
    if kernel == "polynomial":
        kernel = "poly"

    param_grids = {
        "linear": {"C": [0.01, 0.1, 1, 10, 100]},
        "poly": {
            "C": [0.1, 1, 10, 100],
            "degree": [2, 3, 4],
            "coef0": [0.0, 0.5, 1.0],
        },
        "rbf": {
            "C": [0.1, 1, 10, 100],
            "gamma": ["scale", "auto", 0.001, 0.01, 0.1],
        },
        "sigmoid": {
            "C": [0.1, 1, 10, 100],
            "gamma": ["scale", "auto", 0.001, 0.01, 0.1],
            "coef0": [0.0, 0.5, 1.0],
        },
    }

    param_grid = param_grids[kernel]
    n_combinaisons = 1
    for v in param_grid.values():
        n_combinaisons *= len(v)

    log = _ensure_log(log)
    log("\n" + "=" * 55)
    log(f"ÉTAPE 10 : Optimisation hyperparamètres — {best_name}")
    log("=" * 55)
    log("")
    log(f"Kernel optimisé : {kernel}")
    log("Paramètres testés :")
    for param, values in param_grid.items():
        log(f"    • {param} : {values}")
    log(f"Nombre de combinaisons : {n_combinaisons}")
    log("Validation croisée : 5 plis (cv=5)")
    log("Métrique de sélection : F1 macro")
    log(f"Données d'entraînement : jeu complet ({len(x_train)} instances)")
    log("")

    grid = GridSearchCV(
        SVC(
            kernel=kernel,
            probability=True,
            random_state=RANDOM_STATE,
            class_weight="balanced",
        ),
        param_grid,
        cv=5,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=0,
    )
    grid.fit(x_train, y_train)

    best_p = grid.best_params_
    best_f1_cv = grid.best_score_ * 100

    log("Meilleurs paramètres trouvés :")
    for param, value in best_p.items():
        log(f"    {param} = {value}")
    log(f"F1 moyen (validation croisée) = {best_f1_cv:.1f}%")

    y_pred_opt = grid.best_estimator_.predict(x_test)
    p_opt = precision_score(y_test, y_pred_opt, average="macro", zero_division=0) * 100
    r_opt = recall_score(y_test, y_pred_opt, average="macro", zero_division=0) * 100
    f1_opt = f1_score(y_test, y_pred_opt, average="macro", zero_division=0) * 100

    log("\nRapport détaillé après optimisation :")
    report_opt = classification_report(
        y_test,
        y_pred_opt,
        target_names=CLASS_NAMES,
        zero_division=0,
    )
    log(report_opt)
    log(f"\nSur le jeu test (macro %) : P={p_opt:.1f} R={r_opt:.1f} F1={f1_opt:.1f}")

    return grid.best_estimator_


# ============================================================
# MAIN
# ============================================================
def main(data_source=None):
    """
    Exécute le pipeline complet de classification firewall SVM.

    Paramètre :
        data_source (str ou None) :
            - None → données simulées
            - "path/to/file" → fichier CSV local
            - "https://..." → URL vers un fichier CSV
    """
    run_folder = create_run_folder()
    log = make_logger(run_folder)

    log("=" * 55)
    log("  Classification Firewall — SVM Multiclasse")
    log(f"  Dossier de run : {run_folder}/")
    log("=" * 55)

    df = load_data(log, source=data_source)
    x, y_encoded = select_features(log, df)
    x_train, x_test, y_train, y_test = prepare_data(log, x, y_encoded)
    results, models, y_preds = train_svms(log, x_train, y_train, x_test, y_test)
    results_df, best_name = compare_results(log, results)

    detailed_report(log, best_name, y_test, y_preds)

    log("\n" + "=" * 55)
    log("ÉTAPES 7-8 : Graphiques")
    log("=" * 55)
    plot_comparison(log, run_folder, results_df)
    plot_confusion_matrix(log, run_folder, best_name, y_test, y_preds)
    plot_roc_curves(log, run_folder, models, x_test, y_test)

    optimize_best_model(log, best_name, x_train, y_train, x_test, y_test)

    log("\n" + "=" * 55)
    log(f"TERMINÉ — Résultats sauvegardés dans : {run_folder}/")
    log("=" * 55)

    log.close()


if __name__ == "__main__":
    # python firewall_svm.py          → données simulées
    # python firewall_svm.py log2.csv → fichier local
    source = sys.argv[1] if len(sys.argv) > 1 else None
    main(data_source=source)
