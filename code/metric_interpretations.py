"""Interprétations textuelles des métriques (affichage notebook / terminal)."""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report


def print_kernel_macro_summary(
    kernel_display_name: str,
    metrics_pct: Mapping[str, float],
    *,
    average_used: str = "macro",
) -> None:
    """Affiche le bloc métriques aligné sur l'article (F1, précision, rappel en %)."""
    f1 = metrics_pct.get("F1 Score", metrics_pct.get("F1", 0.0))
    p = metrics_pct.get("Precision", metrics_pct.get("Précision", 0.0))
    r = metrics_pct.get("Recall", metrics_pct.get("Rappel", 0.0))
    print("\n" + "=" * 60)
    print(f"  {kernel_display_name} — scores globaux (moyenne « {average_used} »)")
    print("=" * 60)
    print(f"  F1-score (harmonique P/R) : {f1:.1f} %")
    print(f"  Précision                 : {p:.1f} %")
    print(f"  Rappel (sensibilité)      : {r:.1f} %")
    print(
        "  (L'article Ertam & Kaya rapporte ces trois indicateurs par noyau ; "
        "F1 résume le compromis entre précision et rappel.)"
    )


def print_kernel_interpretation(
    kernel_display_name: str,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    class_names: Sequence[str],
    metrics_pct: Mapping[str, float],
) -> None:
    """Interprétation qualitative à afficher à côté des graphiques."""
    print_kernel_macro_summary(kernel_display_name, metrics_pct)
    print("\n--- Interprétation ---")
    p = metrics_pct.get("Precision", 0.0) / 100.0
    r = metrics_pct.get("Recall", 0.0) / 100.0
    if p > r + 0.05:
        print(
            "• La précision macro domine le rappel : en moyenne sur les classes, "
            "les prédictions positives sont plutôt fiables, mais le modèle "
            "peut manquer certaines instances réelles (faux négatifs)."
        )
    elif r > p + 0.05:
        print(
            "• Le rappel macro domine la précision : le modèle retrouve "
            "davantage d'exemples réels par classe, au prix possible de "
            "plus de faux positifs."
        )
    else:
        print(
            "• Précision et rappel macro sont proches : compromis relativement "
            "équilibré entre faux positifs et faux négatifs au niveau global."
        )

    # Classes rares
    support = np.bincount(y_test, minlength=len(class_names))
    rare = [(class_names[i], support[i]) for i in range(len(class_names)) if support[i] < max(support) * 0.05]
    if rare:
        print(
            "\n• Classes peu représentées dans le jeu de test (exemples) : "
            + ", ".join(f"{n} (n={c})" for n, c in rare)
            + ". Leur F1 par classe peut peser peu dans la moyenne macro mais "
              "être critique pour la sécurité (ex. reset-both)."
        )

    print("\n--- Rapport par classe (support = effectifs test) ---")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=list(class_names),
            zero_division=0,
        )
    )


def build_results_dataframe(
    results_by_kernel: Mapping[str, Mapping[str, float]],
    display_names: Mapping[str, str],
    kernel_order: Optional[Sequence[str]] = None,
):
    """Construit le DataFrame bilan (index = libellés article, ordre = kernel_order ou clés de display_names)."""
    order = list(kernel_order) if kernel_order is not None else list(display_names.keys())
    rows = []
    for k in order:
        disp = display_names[k]
        m = results_by_kernel[k]
        rows.append(
            {
                "Méthode": disp,
                "F1 Score": m["F1 Score"],
                "Precision": m["Precision"],
                "Recall": m["Recall"],
            }
        )
    df = pd.DataFrame(rows).set_index("Méthode")
    return df


def print_article_vs_experiment(
    results_df: pd.DataFrame,
    article_table: Mapping[str, Mapping[str, float]],
    *,
    article_label: str = "Ertam & Kaya (2018), Table III",
) -> None:
    """
    Tableau comparatif : valeurs expérimentales vs références article (mêmes colonnes).
    Les écarts sont normaux si le jeu (UCI / Firat) ou le tirage diffère.
    """
    print("\n" + "=" * 70)
    print(f"  Bilan harmonisé — métriques macro (%) vs {article_label}")
    print("=" * 70)
    metrics = ["F1 Score", "Precision", "Recall"]
    for method in results_df.index:
        art = article_table.get(method)
        exp = results_df.loc[method]
        print(f"\n{method}")
        print(f"  {'':12} {'Article':>10} {'Nos rés.':>10} {'Δ (nos−art)':>14}")
        if art is None:
            print("  (pas de ligne article pour ce libellé)")
            continue
        for col in metrics:
            a = art[col]
            e = float(exp[col])
            print(f"  {col:12} {a:10.1f} {e:10.1f} {e - a:+14.1f}")
    print(
        "\nNote : l'article utilise 65 532 instances et 11 attributs numériques ; "
        "la reproduction exacte des pourcentages n'est pas requise si la source "
        "CSV ou le partitionnement diffère, mais la tendance (ex. poly souvent plus faible, "
        "RBF/sigmoid avec rappel élevé) est ce qu'il faut commenter."
    )
