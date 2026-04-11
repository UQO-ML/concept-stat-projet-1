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
import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import SVC
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
    roc_curve, auc
)
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONSTANTES GLOBALES
# ============================================================
FEATURE_NAMES = [
    "Source Port", "Destination Port",
    "NAT Source Port", "NAT Destination Port",
    "Elapsed Time (sec)", "Bytes",
    "Bytes Sent", "Bytes Received",
    "Packets", "pkts_sent", "pkts_received"
]
CLASS_NAMES = ["allow", "deny", "drop", "reset-both"]
RANDOM_STATE = 69


# ============================================================
# UTILITAIRES : dossier de run + logging
# ============================================================
def create_run_folder():
    """
    Crée un dossier run-001, run-002, ... selon les runs déjà existants.
    Retourne le chemin du nouveau dossier.
    """
    existing = sorted(glob.glob("run-[0-9][0-9][0-9]"))
    next_id  = len(existing) + 1
    folder   = f"run-{next_id:03d}"
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
    log("=" * 55)
    log("ÉTAPE 1 : Chargement des données")
    log("=" * 55)

    if source is None:
        # --- Données simulées ---
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
            random_state=RANDOM_STATE
        )
        df = pd.DataFrame(x_raw, columns=FEATURE_NAMES)
        df["Action"] = [CLASS_NAMES[i] for i in y_raw]

    else:
        # --- Fichier CSV local ou URL ---
        log(f"Source : {source}")
        df = pd.read_csv(source)

        # Vérification des colonnes attendues
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
    log(f"\nDistribution des classes :")
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
    log("\n" + "=" * 55)
    log("ÉTAPE 2 : Sélection des features")
    log("=" * 55)

    x = df[FEATURE_NAMES].values
    y = df["Action"].values
    y_encoded = np.array([CLASS_NAMES.index(c) for c in y])

    log(f"{len(FEATURE_NAMES)} features numériques sélectionnées")
    log(f"Variable cible : 'Action' ({len(CLASS_NAMES)} classes)")
    log(f"Données personnelles : exclues")

    return x, y_encoded


# ============================================================
# ÉTAPE 3 — Préparation des données
# ============================================================
def prepare_data(log, x, y_encoded):
    """
    Sépare en train (80%) / test (20%) et normalise avec StandardScaler.
    Retourne x_train, x_test, y_train, y_test.
    """
    log("\n" + "=" * 55)
    log("ÉTAPE 3 : Préparation des données")
    log("=" * 55)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y_encoded,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_encoded
    )

    scaler  = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test  = scaler.transform(x_test)

    log(f"Train : {len(x_train)} instances (80%)")
    log(f"Test : {len(x_test)} instances (20%)")
    log(f"Normalisation : StandardScaler appliquée")

    return x_train, x_test, y_train, y_test


# ============================================================
# ÉTAPE 4 — Entraînement des 4 SVM
# ============================================================
def train_svms(log, x_train, y_train, x_test, y_test):
    """
    Entraîne les 4 classifieurs SVM (Linear, Poly, RBF, Sigmoid).
    Calcule F1, Précision et Rappel pour chacun.
    Retourne results (dict), models (dict), y_preds (dict).
    """
    log("\n" + "=" * 55)
    log("ÉTAPE 4 : Entraînement des 4 classifieurs SVM")
    log("=" * 55)

    kernels = {
        "SVM Linear": SVC(kernel="linear", probability=True, random_state=RANDOM_STATE, class_weight="balanced"),
        "SVM Polynomial": SVC(kernel="poly", probability=True, random_state=RANDOM_STATE, degree=3, class_weight="balanced"),
        "SVM RBF": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE, class_weight="balanced"),
        "SVM Sigmoid": SVC(kernel="sigmoid", probability=True, random_state=RANDOM_STATE, class_weight="balanced"),
    }

    results = {}
    models  = {}
    y_preds = {}

    for name, model in kernels.items():
        log(f"\n {name} : ")
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        p  = precision_score(y_test, y_pred, average="macro", zero_division=0) * 100
        r  = recall_score (y_test, y_pred, average="macro", zero_division=0) * 100
        f1 = f1_score (y_test, y_pred, average="macro", zero_division=0) * 100

        results[name] = {"F1 Score": round(f1,1), "Precision": round(p,1), "Recall": round(r,1)}
        models[name]  = model
        y_preds[name] = y_pred
        log(f"F1={f1:.1f}%  Précision={p:.1f}%  Rappel={r:.1f}%")

    return results, models, y_preds


# ============================================================
# ÉTAPE 5 — Tableau comparatif
# ============================================================
def compare_results(log, results):
    """
    Affiche le tableau comparatif des 4 kernels et retourne le nom du meilleur modèle selon le F1 Score.
    """
    log("\n" + "=" * 55)
    log("ÉTAPE 5 : Tableau comparatif des performances")
    log("=" * 55)

    results_df = pd.DataFrame(results).T
    log("\n" + results_df.to_string())

    best_name = results_df["F1 Score"].idxmax()
    log(f"\n  Meilleur modèle (F1) : {best_name} "
        f"— F1={results_df.loc[best_name,'F1 Score']}%  "
        f"Précision={results_df.loc[best_name,'Precision']}%  "
        f"Rappel={results_df.loc[best_name,'Recall']}%")

    return results_df, best_name


# ============================================================
# ÉTAPE 6 — Rapport détaillé du meilleur modèle
# ============================================================
def detailed_report(log, best_name, y_test, y_preds):
    """
    Affiche le rapport de classification complet (précision, rappel, F1 par classe) pour le meilleur modèle.
    """
    log("\n" + "=" * 55)
    log(f"ÉTAPE 6 : Rapport détaillé — {best_name}")
    log("=" * 55)

    report = classification_report(
        y_test, y_preds[best_name],
        target_names=CLASS_NAMES
    )
    log(report)


# ============================================================
# ÉTAPE 7 — Graphique comparatif
# ============================================================
def plot_comparison(log, run_folder, results_df):
    """
    Génère un graphique en barres comparant F1, Précision et Rappel pour les 4 kernels.
    Sauvegarde dans run_folder.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(results_df))
    w = 0.25
    colors_bar = ["#4C72B0", "#DD8452", "#55A868"]

    for i, (metric, color) in enumerate(zip(["F1 Score","Precision","Recall"], colors_bar)):
        bars = ax.bar(x + (i-1)*w, results_df[metric], w, label=metric, color=color)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(results_df.index, rotation=10, fontsize=10)
    ax.set_ylim(0, 115)
    ax.set_ylabel("Score (%)", fontsize=11)
    ax.set_title("Comparaison SVM — F1 Score, Précision, Rappel", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    path = os.path.join(run_folder, "comparaison_svm.png")
    plt.savefig(path, dpi=130)
    plt.close()
    log(f"Sauvegardé : {path}")


# ============================================================
# ÉTAPE 8 — Matrice de confusion
# ============================================================
def plot_confusion_matrix(log, run_folder, best_name, y_test, y_preds):
    """
    Génère la matrice de confusion pour le meilleur modèle.
    Sauvegarde dans run_folder.
    """
    cm = confusion_matrix(y_test, y_preds[best_name])
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_xlabel("Prédit", fontsize=11)
    ax.set_ylabel("Réel", fontsize=11)
    ax.set_title(f"Matrice de confusion — {best_name}", fontsize=12)
    plt.tight_layout()

    path = os.path.join(run_folder, "matrice_confusion.png")
    plt.savefig(path, dpi=130)
    plt.close()
    log(f"Sauvegardé : {path}")


# ============================================================
# ÉTAPE 9 — Courbes ROC
# ============================================================
def plot_roc_curves(log, run_folder, models, x_test, y_test):
    """
    Génère une courbe ROC par classe (allow/deny/drop/reset-both), avec une courbe par kernel SVM.
    Sauvegarde dans run_folder.
    """
    log("\n" + "=" * 55)
    log("ÉTAPE 9 : Courbes ROC par classe")
    log("=" * 55)

    y_test_bin    = label_binarize(y_test, classes=[0,1,2,3])
    kernel_colors = ["blue", "green", "red", "purple"]

    for class_idx, class_name in enumerate(CLASS_NAMES):
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot([0,1],[0,1],"k--", lw=1, label="Aléatoire")

        for (name, model), color in zip(models.items(), kernel_colors):
            y_score     = model.predict_proba(x_test)[:, class_idx]
            fpr, tpr, _ = roc_curve(y_test_bin[:, class_idx], y_score)
            roc_auc     = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=color, lw=1.8,
                    label=f"{name.split()[1]} (AUC={roc_auc:.3f})")

        ax.set_xlim([0,1]); ax.set_ylim([0,1.02])
        ax.set_xlabel("FP Rate (1 - Spécificité)", fontsize=11)
        ax.set_ylabel("TP Rate (Sensibilité)", fontsize=11)
        ax.set_title(f"Courbe ROC — Classe '{class_name}'", fontsize=12)
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(alpha=0.3)
        plt.tight_layout()

        fname = os.path.join(run_folder, f"roc_{class_name.replace('-','_')}.png")
        plt.savefig(fname, dpi=120)
        plt.close()
        log(f"Sauvegardé : {fname}")


# ============================================================
# ÉTAPE 10 — Optimisation des hyperparamètres (GridSearchCV)
# ============================================================
def optimize_best_model(log, best_name, x_train, y_train, x_test, y_test):
    """
    Optimise les hyperparamètres du meilleur modèle via GridSearchCV avec validation croisée à 5 plis.

    --- Explication ---
    Chaque kernel SVM a ses propres hyperparamètres à optimiser.
    On sélectionne la grille de paramètres selon le kernel gagnant.

    Hyperparamètre commun à tous les kernels :
    • C (régularisation) :
        Contrôle le compromis biais/variance.
        - C faible (ex: 0.1) → marge large, tolère des erreurs, meilleure généralisation mais risque de sous-apprentissage.
        - C élevé (ex: 100) → marge étroite, peu d'erreurs tolérées, risque de sur-apprentissage.

    Hyperparamètres spécifiques selon le kernel :
    • gamma (RBF et Sigmoid) :
        Définit l'influence de chaque point d'entraînement.
        - gamma faible → influence large, frontière de décision lisse.
        - gamma élevé  → influence locale, frontière très complexe.
        - "scale" = 1 / (n_features * Var(X))
        - "auto"  = 1 / n_features

    • degree (Polynomial uniquement) :
        Degré du polynôme utilisé pour la transformation des données.
        - degree=2 → séparation quadratique (moins complexe)
        - degree=3 → cubique (par défaut)
        - degree=4 → quartique (plus expressif mais plus lent)

    • coef0 (Polynomial et Sigmoid) :
        Terme indépendant dans la fonction du kernel.
        Influence les termes de degré inférieur (poly) ou le décalage de la fonction sigmoïde.

    On entraîne sur l'intégralité de x_train (pas un sous-ensemble), ce qui donne une estimation plus fiable des hyperparamètres.
    La validation croisée à 5 plis divise x_train en 5 blocs à chaque pli, 4 blocs servent à l'entraînement et 1 à la validation, en tournant 5 fois.
    Le score final est la moyenne des 5 évaluations, ce qui évite de choisir des paramètres qui ne fonctionnent bien que sur un seul découpage des données.

    Retourne le meilleur estimateur entraîné.
    """
    # Extrait le kernel depuis le nom (ex: "SVM RBF" → "rbf")
    kernel = best_name.split(" ", 1)[1].lower()  # "linear", "polynomial", "rbf", "sigmoid"
    if kernel == "polynomial":
        kernel = "poly"

    # Grilles de paramètres selon le kernel
    param_grids = {
        "linear": {
            "C": [0.01, 0.1, 1, 10, 100]
        },
        "poly": {
            "C":      [0.1, 1, 10, 100],
            "degree": [2, 3, 4],
            "coef0":  [0.0, 0.5, 1.0]
        },
        "rbf": {
            "C":     [0.1, 1, 10, 100],
            "gamma": ["scale", "auto", 0.001, 0.01, 0.1]
        },
        "sigmoid": {
            "C":     [0.1, 1, 10, 100],
            "gamma": ["scale", "auto", 0.001, 0.01, 0.1],
            "coef0": [0.0, 0.5, 1.0]
        },
    }

    param_grid   = param_grids[kernel]
    n_combinaisons = 1
    for v in param_grid.values():
        n_combinaisons *= len(v)

    log("\n" + "=" * 55)
    log(f"ÉTAPE 10 : Optimisation hyperparamètres — {best_name}")
    log("=" * 55)
    log("")
    log(f"Kernel optimisé : {kernel}")
    log(f"Paramètres testés :")
    for param, values in param_grid.items():
        log(f"    • {param} : {values}")
    log(f"Nombre de combinaisons : {n_combinaisons}")
    log(f"Validation croisée : 5 plis (cv=5)")
    log(f"Métrique de sélection : F1 macro")
    log(f"Données d'entraînement : jeu complet ({len(x_train)} instances)")
    log("")

    grid = GridSearchCV(
        SVC(kernel=kernel, probability=True, random_state=RANDOM_STATE, class_weight="balanced"),
        param_grid,
        cv=5,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=0
    )
    grid.fit(x_train, y_train)   # jeu d'entraînement COMPLET

    best_p     = grid.best_params_
    best_f1_cv = grid.best_score_ * 100

    log(f"Meilleurs paramètres trouvés :")
    for param, value in best_p.items():
        log(f"    {param} = {value}")
    log(f"F1 moyen (validation croisée) = {best_f1_cv:.1f}%")

    # Évaluation finale sur le test set
    y_pred_opt = grid.best_estimator_.predict(x_test)
    p_opt  = precision_score(y_test, y_pred_opt, average="macro", zero_division=0) * 100
    r_opt  = recall_score   (y_test, y_pred_opt, average="macro", zero_division=0) * 100
    f1_opt = f1_score       (y_test, y_pred_opt, average="macro", zero_division=0) * 100

    log(f"\nRapport détaillé après optimisation :")
    report_opt = classification_report(
        y_test, y_pred_opt,
        target_names=CLASS_NAMES,
        zero_division=0
    )
    log(report_opt)
 
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
    log        = make_logger(run_folder)

    log("=" * 55)
    log("  Classification Firewall — SVM Multiclasse")
    log(f"  Dossier de run : {run_folder}/")
    log("=" * 55)

    df = load_data(log, source=data_source)
    x, y_encoded  = select_features(log, df)
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
    # Utilisation depuis le terminal :
    #   python firewall_svm.py → données simulées
    #   python firewall_svm.py log2.csv → fichier CSV local
    #   python firewall_svm.py https://url/log2.csv → URL
    source = sys.argv[1] if len(sys.argv) > 1 else None
    main(data_source=source)