"""Conseils sur les hyperparamètres SVM utilisés (C, gamma, degree, coef0, class_weight)."""

from __future__ import annotations

from typing import Any, Mapping, Optional


# Valeurs sklearn par défaut (référence) pour contextualiser nos choix
_SKLEARN_DEFAULTS = """
Référence sklearn.svm.SVC (extraits) :
  C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0,
  class_weight=None, probability=False (nous utilisons probability=True pour les ROC).
"""


def describe_current_svc_params(pipeline: Any) -> Optional[Mapping[str, Any]]:
    """Extrait les paramètres du SVC depuis un Pipeline sklearn [('scaler', _), ('svm', SVC)]."""
    if pipeline is None or not hasattr(pipeline, "named_steps"):
        return None
    svm = pipeline.named_steps.get("svm")
    if svm is None:
        return None
    return svm.get_params()


def print_kernel_hyperparam_advice(kernel_key: str, pipeline: Any | None = None) -> None:
    """
    Évalue (verbalement) les hyperparamètres typiques du noyau et donne des pistes d'ajustement.
    Ne lance pas de recherche sur grille (voir optimize_best_model dans firewall_svm.py).
    """
    print("\n" + "-" * 60)
    print(f"  Conseils hyperparamètres — noyau « {kernel_key} »")
    print("-" * 60)
    print(_SKLEARN_DEFAULTS.strip())
    params = describe_current_svc_params(pipeline)
    if params:
        print("\nParamètres effectifs du modèle dans cette cellule :")
        keys = ("C", "kernel", "degree", "gamma", "coef0", "class_weight", "random_state")
        for k in keys:
            if k in params:
                print(f"  • {k} = {params[k]!r}")

    print("\n--- Conseils par noyau ---")
    if kernel_key == "linear":
        print(
            "• Linear : seul C et class_weight influencent fortement. "
            "C plus grand → marge plus étroite, risque de sur-apprentissage ; "
            "C plus petit → marge large, peut sous-ajuster. "
            "Avec classes déséquilibrées, class_weight='balanced' aide le rappel des classes rares."
        )
    elif kernel_key == "poly":
        print(
            "• Polynomial : degree (souvent 2–4), C, coef0. "
            "Un degré élevé augmente la complexité ; si F1 macro chute, tester degree=2 "
            "ou réduire C avant d'augmenter la complexité."
        )
    elif kernel_key == "rbf":
        print(
            "• RBF : C et gamma dominent. gamma contrôle la portée locale du noyau ; "
            "'scale' ou 'auto' est un bon point de départ. "
            "gamma trop grand → frontière très oscillante ; trop petit → presque linéaire."
        )
    elif kernel_key == "sigmoid":
        print(
            "• Sigmoid : sensible à gamma et coef0 (similarité avec réseaux). "
            "Peut imiter des comportements saturés ; si instable, rapprocher gamma de 'scale' "
            "ou ajuster coef0."
        )
    else:
        print("• Noyau non reconnu pour des conseils spécifiques.")

    print(
        "\nPour une recherche systématique : utiliser GridSearchCV (fonction "
        "`optimize_best_model` dans `firewall_svm.py`) sur le jeu d'entraînement, "
        "métrique f1_macro, sans toucher au jeu test jusqu'à l'évaluation finale."
    )
