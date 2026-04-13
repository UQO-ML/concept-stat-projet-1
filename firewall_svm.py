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
import json
import os
import sys
import warnings
from datetime import datetime, timezone

_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
_CODE_DIR = os.path.join(_ROOT_DIR, "code")
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.datasets import make_classification
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_recall_curve,
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
RUN_MANIFEST_JSON = "run_manifest.json"
THRESHOLD_TUNING_CANDIDATES = (0.5, 0.3, 0.2, 0.1, 0.05, 0.02)
THRESHOLD_SELECTION_TARGET = "reset-both_f1"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


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
    Crée un dossier horodaté run-YYYYMMDDTHHMMSSZ (UTC), suffixe -01 si collision.
    Permet d'identifier sans ambiguïté la fraîcheur des résultats (voir run_manifest.json).
    """
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = f"run-{ts}"
    folder = base
    n = 1
    while os.path.exists(folder):
        folder = f"{base}-{n:02d}"
        n += 1
    os.makedirs(folder, exist_ok=True)
    return folder


def _write_run_manifest(run_folder: str, manifest: dict) -> None:
    path = os.path.join(run_folder, RUN_MANIFEST_JSON)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


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


def predict_with_reset_threshold(y_proba: np.ndarray, threshold: float, reset_class_index: int = 3) -> np.ndarray:
    """
    Prédit `reset-both` si sa probabilité dépasse un seuil dédié.
    Sinon, garde l'argmax parmi les autres classes.
    """
    pred_non_reset = np.argmax(y_proba[:, :reset_class_index], axis=1)
    return np.where(y_proba[:, reset_class_index] >= threshold, reset_class_index, pred_non_reset)


def _classification_report_dict(y_true, y_pred) -> dict:
    return classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )


def _score_threshold_results(report_dict: dict, target: str) -> float:
    if target == "reset-both_f1":
        return float(report_dict.get("reset-both", {}).get("f1-score", 0.0))
    if target == "reset-both_recall":
        return float(report_dict.get("reset-both", {}).get("recall", 0.0))
    if target == "macro_f1":
        return float(report_dict.get("macro avg", {}).get("f1-score", 0.0))
    raise ValueError(f"Cible de sélection de seuil non supportée: {target}")


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
    Optimise les hyperparamètres du meilleur modèle via GridSearchCV (cv=5),
    puis calibre un seuil dédié à la classe `reset-both` sur un split de validation.

    Retourne un dictionnaire contenant:
      - best_estimator (modèle ajusté sur x_train complet),
      - paramètres GridSearch,
      - résultats de calibration de seuil (validation),
      - résultats comparatifs sur test (seuil 0.50 vs seuil calibré).
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

    best_estimator = grid.best_estimator_

    # --- Évaluation de référence sur test (seuil multiclasses standard) ---
    y_pred_opt_default = best_estimator.predict(x_test)
    p_opt = precision_score(y_test, y_pred_opt_default, average="macro", zero_division=0) * 100
    r_opt = recall_score(y_test, y_pred_opt_default, average="macro", zero_division=0) * 100
    f1_opt = f1_score(y_test, y_pred_opt_default, average="macro", zero_division=0) * 100

    log("\nRapport détaillé après optimisation (seuil par défaut 0.50) :")
    report_opt_text = classification_report(
        y_test,
        y_pred_opt_default,
        target_names=CLASS_NAMES,
        zero_division=0,
    )
    log(report_opt_text)
    log(f"\nSur le jeu test (macro %) : P={p_opt:.1f} R={r_opt:.1f} F1={f1_opt:.1f}")

    # --- Calibration du seuil reset-both sur split validation (pas sur test) ---
    log("\n" + "=" * 55)
    log("ÉTAPE 10B : Calibration du seuil reset-both (sur validation)")
    log("=" * 55)

    x_train_cal, x_val_cal, y_train_cal, y_val_cal = train_test_split(
        x_train,
        y_train,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_train,
    )
    log(
        "Calibration interne: "
        f"train_cal={len(x_train_cal)} / val_cal={len(x_val_cal)} (stratifié)"
    )

    calib_model = clone(best_estimator)
    calib_model.fit(x_train_cal, y_train_cal)
    val_proba = calib_model.predict_proba(x_val_cal)
    val_reset_true = (y_val_cal == 3).astype(int)

    val_results: dict[str, dict] = {}
    for thr in THRESHOLD_TUNING_CANDIDATES:
        y_val_pred_thr = predict_with_reset_threshold(val_proba, float(thr))
        rep = _classification_report_dict(y_val_cal, y_val_pred_thr)
        val_results[f"{float(thr):.4f}"] = rep

    # Seuil auto issu de la courbe PR, calculé sur validation.
    p_curve, r_curve, th_curve = precision_recall_curve(val_reset_true, val_proba[:, 3])
    f1_curve = 2 * p_curve * r_curve / (p_curve + r_curve + 1e-12)
    if len(th_curve) > 0:
        # Le dernier couple precision/recall n'a pas de seuil associé.
        max_idx = int(np.argmax(f1_curve[:-1])) if len(f1_curve) > 1 else 0
        auto_thr = float(th_curve[max_idx])
        y_val_pred_auto = predict_with_reset_threshold(val_proba, auto_thr)
        val_results[f"auto({auto_thr:.4f})"] = _classification_report_dict(y_val_cal, y_val_pred_auto)
    else:
        auto_thr = 0.5

    best_thr_label, best_thr_report = max(
        val_results.items(),
        key=lambda kv: _score_threshold_results(kv[1], THRESHOLD_SELECTION_TARGET),
    )
    if best_thr_label.startswith("auto("):
        selected_threshold = auto_thr
    else:
        selected_threshold = float(best_thr_label)

    best_thr_score = _score_threshold_results(best_thr_report, THRESHOLD_SELECTION_TARGET)
    log(
        f"Seuil sélectionné ({THRESHOLD_SELECTION_TARGET}) : {selected_threshold:.4f} "
        f"(score validation={best_thr_score:.4f})"
    )

    # --- Évaluation finale sur test (seuil sélectionné) ---
    test_proba = best_estimator.predict_proba(x_test)
    y_pred_opt_selected = predict_with_reset_threshold(test_proba, selected_threshold)

    default_report_dict = _classification_report_dict(y_test, y_pred_opt_default)
    selected_report_dict = _classification_report_dict(y_test, y_pred_opt_selected)

    log("\nRapport détaillé sur test — seuil sélectionné :")
    log(
        classification_report(
            y_test,
            y_pred_opt_selected,
            target_names=CLASS_NAMES,
            zero_division=0,
        )
    )

    log("\nComparaison test (macro avg):")
    log(
        f"  défaut 0.50    -> P={default_report_dict['macro avg']['precision']*100:.1f} "
        f"R={default_report_dict['macro avg']['recall']*100:.1f} "
        f"F1={default_report_dict['macro avg']['f1-score']*100:.1f}"
    )
    log(
        f"  seuil {selected_threshold:.4f} -> P={selected_report_dict['macro avg']['precision']*100:.1f} "
        f"R={selected_report_dict['macro avg']['recall']*100:.1f} "
        f"F1={selected_report_dict['macro avg']['f1-score']*100:.1f}"
    )
    log(
        f"  reset-both (F1) défaut={default_report_dict.get('reset-both',{}).get('f1-score',0)*100:.2f} "
        f"/ seuil={selected_report_dict.get('reset-both',{}).get('f1-score',0)*100:.2f}"
    )

    return {
        "best_estimator": best_estimator,
        "grid_best_params": best_p,
        "grid_best_f1_cv_pct": round(float(best_f1_cv), 2),
        "threshold_selection_target": THRESHOLD_SELECTION_TARGET,
        "threshold_candidates": [float(t) for t in THRESHOLD_TUNING_CANDIDATES],
        "selected_reset_both_threshold": float(selected_threshold),
        "validation_threshold_results": val_results,
        "test_threshold_results": {
            "default_0.5000": default_report_dict,
            f"selected_{selected_threshold:.4f}": selected_report_dict,
        },
    }


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

    Écrit ``run_manifest.json`` dans le dossier de run (horodatage UTC, effectifs,
    métriques macro par noyau, F1 par classe pour le meilleur noyau et après GridSearch).
    """
    run_folder = create_run_folder()
    log = make_logger(run_folder)

    manifest: dict = {
        "run_folder": run_folder,
        "started_at_utc": _utc_now_iso(),
        "data_source": data_source,
        "random_state": RANDOM_STATE,
        "class_names": list(CLASS_NAMES),
    }

    log("=" * 55)
    log("  Classification Firewall — SVM Multiclasse")
    log(f"  Dossier de run : {run_folder}/")
    log(f"  Horodatage début (UTC) : {manifest['started_at_utc']}")
    log("=" * 55)

    df = load_data(log, source=data_source)
    x, y_encoded = select_features(log, df)
    x_train, x_test, y_train, y_test = prepare_data(log, x, y_encoded)

    full_counts = df["Action"].value_counts().reindex(CLASS_NAMES, fill_value=0)
    manifest["class_counts_full"] = {str(k): int(v) for k, v in full_counts.items()}

    manifest["class_counts_train"] = {
        CLASS_NAMES[i]: int((y_train == i).sum()) for i in range(len(CLASS_NAMES))
    }
    manifest["class_counts_test"] = {
        CLASS_NAMES[i]: int((y_test == i).sum()) for i in range(len(CLASS_NAMES))
    }
    _tv = list(manifest["class_counts_test"].values())
    manifest["test_support_ratio"] = float(max(_tv) / max(min(_tv), 1))

    results, models, y_preds = train_svms(log, x_train, y_train, x_test, y_test)
    results_df, best_name = compare_results(log, results)

    manifest["kernel_results_macro_pct"] = {k: dict(v) for k, v in results.items()}
    manifest["best_kernel_display_name"] = best_name

    y_best = y_preds[best_name]
    f1_pc = f1_score(y_test, y_best, average=None, zero_division=0)
    manifest["f1_per_class_best_kernel_pct"] = {
        CLASS_NAMES[i]: round(float(f1_pc[i]) * 100, 2) for i in range(len(CLASS_NAMES))
    }

    detailed_report(log, best_name, y_test, y_preds)

    log("\n" + "=" * 55)
    log("ÉTAPES 7-8 : Graphiques")
    log("=" * 55)
    plot_comparison(log, run_folder, results_df)
    plot_confusion_matrix(log, run_folder, best_name, y_test, y_preds)
    plot_roc_curves(log, run_folder, models, x_test, y_test)

    opt_payload = optimize_best_model(log, best_name, x_train, y_train, x_test, y_test)
    opt_est = opt_payload["best_estimator"]
    y_pred_opt = opt_est.predict(x_test)
    f1_pc_opt = f1_score(y_test, y_pred_opt, average=None, zero_division=0)
    manifest["f1_per_class_after_gridsearch_pct"] = {
        CLASS_NAMES[i]: round(float(f1_pc_opt[i]) * 100, 2) for i in range(len(CLASS_NAMES))
    }
    manifest["threshold_tuning"] = {
        "selection_target": opt_payload["threshold_selection_target"],
        "candidates": opt_payload["threshold_candidates"],
        "selected_reset_both_threshold": opt_payload["selected_reset_both_threshold"],
        "grid_best_f1_cv_pct": opt_payload["grid_best_f1_cv_pct"],
        "grid_best_params": opt_payload["grid_best_params"],
        "validation_threshold_results": opt_payload["validation_threshold_results"],
        "test_threshold_results": opt_payload["test_threshold_results"],
    }

    selected_key = f"selected_{opt_payload['selected_reset_both_threshold']:.4f}"
    selected_report = opt_payload["test_threshold_results"][selected_key]
    manifest["f1_per_class_after_threshold_tuning_pct"] = {
        cls: round(float(selected_report.get(cls, {}).get("f1-score", 0.0)) * 100, 2)
        for cls in CLASS_NAMES
    }

    manifest["finished_at_utc"] = _utc_now_iso()
    _write_run_manifest(run_folder, manifest)
    log(f"\nManifest JSON : {os.path.join(run_folder, RUN_MANIFEST_JSON)}")

    log("\n" + "=" * 55)
    log(f"TERMINÉ — Résultats sauvegardés dans : {run_folder}/")
    log("=" * 55)

    log.close()


if __name__ == "__main__":
    # python firewall_svm.py              → utilise log2.csv si trouvé, sinon simulation
    # python firewall_svm.py log2.csv     → fichier local explicite
    # python firewall_svm.py https://...  → URL explicite
    if len(sys.argv) > 1:
        source = sys.argv[1]
    else:
        candidates = [
            "log2.csv",
            os.path.join(_ROOT_DIR, "log2.csv"),
        ]
        source = next((p for p in candidates if os.path.exists(p)), None)
        if source is None:
            print("[firewall_svm] Aucun log2.csv trouvé; bascule sur données simulées.")
        else:
            print(f"[firewall_svm] Source auto-détectée: {source}")
    main(data_source=source)
