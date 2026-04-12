"""Graphiques pour le projet pare-feu SVM (notebook et sauvegarde run-XXX)."""

from __future__ import annotations

import os
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import auc, confusion_matrix, roc_curve


def plot_confusion_kernel(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    class_names: Sequence[str],
    title: str,
    *,
    save_path: Optional[str] = None,
    figsize: tuple[float, float] = (6.0, 5.0),
) -> None:
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Prédit", fontsize=11)
    ax.set_ylabel("Réel", fontsize=11)
    ax.set_title(title, fontsize=12)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130)
        plt.close(fig)
    else:
        plt.show()


def plot_roc_multiclass_single_kernel(
    y_test: np.ndarray,
    y_score: np.ndarray,
    class_names: Sequence[str],
    kernel_title: str,
    *,
    save_path: Optional[str] = None,
    figsize: tuple[float, float] = (9.0, 8.0),
) -> None:
    """Une figure : sous-graphiques ROC (une courbe par classe), article Figs. 2–5 regroupées."""
    n = len(class_names)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes_flat = np.atleast_1d(axes).ravel()
    for i, cname in enumerate(class_names):
        ax = axes_flat[i]
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Aléatoire")
        fpr, tpr, _ = roc_curve(y_test == i, y_score[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2.0, label=f"AUC = {roc_auc:.3f}")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.02])
        ax.set_xlabel("FP rate (1 − spécificité)", fontsize=10)
        ax.set_ylabel("TP rate (sensibilité)", fontsize=10)
        ax.set_title(f"ROC — {cname}", fontsize=11)
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(alpha=0.3)
    for j in range(len(class_names), len(axes_flat)):
        axes_flat[j].set_visible(False)
    fig.suptitle(f"Courbes ROC par classe — {kernel_title}", fontsize=13, y=1.02)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_comparison_bar(
    results_df,
    *,
    title: str = "Comparaison SVM — F1, Précision, Rappel (macro %)",
    ylim: tuple[float, float] = (0, 115),
    save_path: Optional[str] = None,
    figsize: tuple[float, float] = (10.0, 5.0),
) -> None:
    """results_df : index = noms méthodes, colonnes F1 Score, Precision, Recall (valeurs 0–100)."""
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(results_df))
    w = 0.25
    colors_bar = ["#4C72B0", "#DD8452", "#55A868"]
    for i, (metric, color) in enumerate(
        zip(["F1 Score", "Precision", "Recall"], colors_bar)
    ):
        bars = ax.bar(x + (i - 1) * w, results_df[metric], w, label=metric, color=color)
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{bar.get_height():.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(results_df.index, rotation=10, fontsize=10)
    ax.set_ylim(ylim)
    ax.set_ylabel("Score (%)", fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=130)
        plt.close(fig)
    else:
        plt.show()


def plot_roc_all_kernels_per_class(
    models_by_name: dict,
    x_test: np.ndarray,
    y_test_bin: np.ndarray,
    class_names: Sequence[str],
    run_folder: str,
    log,
) -> None:
    """Équivalent pipeline article : une figure par classe, toutes les courbes (noyaux) superposées."""
    kernel_colors = ["blue", "green", "red", "purple"]
    for class_idx, class_name in enumerate(class_names):
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Aléatoire")
        for (name, model), color in zip(models_by_name.items(), kernel_colors):
            y_score = model.predict_proba(x_test)[:, class_idx]
            fpr, tpr, _ = roc_curve(y_test_bin[:, class_idx], y_score)
            roc_auc = auc(fpr, tpr)
            short = name.split()[-1] if " " in name else name
            ax.plot(
                fpr,
                tpr,
                color=color,
                lw=1.8,
                label=f"{short} (AUC={roc_auc:.3f})",
            )
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.02])
        ax.set_xlabel("FP Rate (1 - Spécificité)", fontsize=11)
        ax.set_ylabel("TP Rate (Sensibilité)", fontsize=11)
        ax.set_title(f"Courbe ROC — Classe '{class_name}'", fontsize=12)
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        fname = os.path.join(run_folder, f"roc_{class_name.replace('-', '_')}.png")
        fig.savefig(fname, dpi=120)
        plt.close(fig)
        log(f"Sauvegardé : {fname}")
