"""Interprétations textuelles des métriques (affichage notebook / terminal)."""

from __future__ import annotations

import math
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.metrics import classification_report

from run_timestamps import discover_manifest_runs, runs_for_source


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


def print_crossrun_class_imbalance_bilan(
    repo_root: Path | str,
    data_source_filter: str | None = None,
) -> None:
    """
    Compare les runs CLI terminés : pire vs meilleur sur la classe minoritaire
    « reset-both » (F1 % du meilleur noyau avant grille), et rappelle le déséquilibre des supports test.
    """
    root = Path(repo_root).resolve()
    runs = (
        runs_for_source(root, data_source_filter)
        if data_source_filter is not None
        else discover_manifest_runs(root)
    )
    print("\n" + "=" * 70)
    print("  Bilan multi-runs — déséquilibre des classes et sort « reset-both »")
    print("=" * 70)
    if len(runs) < 1:
        print("  Aucun run manifesté : exécuter `python firewall_svm.py` pour générer des métadonnées.")
        return

    def reset_both_f1(m: dict) -> float:
        d = m.get("f1_per_class_best_kernel_pct") or {}
        v = d.get("reset-both")
        return float(v) if v is not None else float("nan")

    scored: list[tuple[float, Path, dict]] = []
    for folder, man in runs:
        s = reset_both_f1(man)
        scored.append((s, folder, man))

    valid = [t for t in scored if not math.isnan(t[0])]
    if not valid:
        print("  Manifests sans f1_per_class_best_kernel_pct : régénérer les runs avec la version actuelle.")
        return

    worst = min(valid, key=lambda x: x[0])
    best = max(valid, key=lambda x: x[0])

    print(
        "\n• Déséquilibre typique (jeu de test) : la classe « reset-both » a un support "
        "très faible par rapport à « allow » ; le ratio max/min sur le test est indiqué "
        "dans chaque manifest (`test_support_ratio`)."
    )

    def _block(label: str, folder: Path, man: dict) -> None:
        ct = man.get("class_counts_test") or {}
        ratio = man.get("test_support_ratio")
        f1s = man.get("f1_per_class_best_kernel_pct") or {}
        print(f"\n--- {label} : {folder.name} (fin UTC {man.get('finished_at_utc', '?')}) ---")
        print(f"  Effectifs test : {ct}")
        if ratio is not None:
            print(f"  Ratio max(support)/min(support) sur le test : {ratio:.1f}")
        print(f"  F1 par classe (meilleur noyau avant GridSearch) % : {f1s}")
        rb = f1s.get("reset-both")
        if rb is not None:
            print(f"  → Indicateur « reset-both » retenu pour le tri : F1 = {rb} %")

    if len(valid) >= 2:
        ordered = sorted(valid, key=lambda x: x[0])
        worst = ordered[0]
        best = ordered[-1]
        _block("Run le moins favorable pour « reset-both » (F1 plus bas)", worst[1], worst[2])
        _block("Run le plus favorable pour « reset-both » (F1 plus haut)", best[1], best[2])
        print(
            "\n• Interprétation : avec ~54 occurrences totales de « reset-both » dans le CSV, "
            "le découpage stratifié 80/20 ne laisse qu'une dizaine d'exemples en test ; la variance "
            "entre runs (même code) peut donc être grande. `class_weight='balanced'` aide au rappel "
            "mais ne garantit pas un F1 élevé sur une classe aussi rare."
        )
    else:
        _block("Seul run manifesté disponible", valid[0][1], valid[0][2])
        print(
            "\n• Pour une comparaison pire/meilleur run sur « reset-both », il faut au moins deux "
            "exécutions complètes de `firewall_svm.py` (deux dossiers run-* avec manifest) "
            "dans le périmètre de source sélectionné."
        )


def print_threshold_tuning_bilan(manifest: Mapping[str, object]) -> None:
    """
    Affiche le résumé de calibration de seuil `reset-both` à partir d'un run_manifest.
    """
    t = manifest.get("threshold_tuning") if isinstance(manifest, Mapping) else None
    if not isinstance(t, Mapping):
        print("\n[threshold] Aucun bloc `threshold_tuning` dans ce manifest.")
        return

    selected = t.get("selected_reset_both_threshold")
    target = t.get("selection_target")
    test_results = t.get("test_threshold_results") or {}
    if not isinstance(test_results, Mapping):
        test_results = {}

    print("\n" + "=" * 70)
    print("  Calibration du seuil reset-both — bilan")
    print("=" * 70)
    print(f"  Cible de sélection          : {target!r}")
    if selected is not None:
        print(f"  Seuil retenu                : {float(selected):.4f}")

    default_rep = test_results.get("default_0.5000", {})
    selected_rep = test_results.get(f"selected_{float(selected):.4f}", {}) if selected is not None else {}
    if not isinstance(default_rep, Mapping) or not isinstance(selected_rep, Mapping):
        print("  Résultats test incomplets dans le manifest.")
        return

    def _val(rep: Mapping[str, object], block: str, metric: str) -> float:
        sub = rep.get(block, {})
        if isinstance(sub, Mapping):
            return float(sub.get(metric, 0.0))
        return 0.0

    rows = [
        ("Macro F1", _val(default_rep, "macro avg", "f1-score"), _val(selected_rep, "macro avg", "f1-score")),
        ("Macro Recall", _val(default_rep, "macro avg", "recall"), _val(selected_rep, "macro avg", "recall")),
        ("Macro Precision", _val(default_rep, "macro avg", "precision"), _val(selected_rep, "macro avg", "precision")),
        (
            "reset-both F1",
            _val(default_rep, "reset-both", "f1-score"),
            _val(selected_rep, "reset-both", "f1-score"),
        ),
        (
            "reset-both Recall",
            _val(default_rep, "reset-both", "recall"),
            _val(selected_rep, "reset-both", "recall"),
        ),
        (
            "reset-both Precision",
            _val(default_rep, "reset-both", "precision"),
            _val(selected_rep, "reset-both", "precision"),
        ),
    ]

    print(f"\n  {'Métrique':24} {'Défaut(0.50)':>14} {'Seuil sélectionné':>18} {'Δ':>10}")
    print("  " + "-" * 68)
    for label, a, b in rows:
        print(f"  {label:24} {a*100:14.2f} {b*100:18.2f} {(b-a)*100:10.2f}")
