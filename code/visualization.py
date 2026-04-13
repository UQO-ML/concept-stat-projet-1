from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Any
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

CLASS_NAMES = ['avion','auto','oiseau','chat','cerf',
            'chien','grenouille','cheval','navire','camion']

    # Exemples d'images CIFAR-10 par classe
def image_sample_by_class(x_train, y_train) -> Any:
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for i, ax in enumerate(axes.flat):
        idx = np.where(y_train.flatten() == i)[0][0]
        ax.imshow(x_train[idx])
        ax.set_title(CLASS_NAMES[i])
        ax.axis('off')
    fig.suptitle("1 exemple par classe CIFAR-10")
    plt.tight_layout()
    return plt.show()


# Distribution des classes (train + test)
def distribution(y_train, y_test) -> Any:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.bar(CLASS_NAMES, np.bincount(y_train.flatten()), color='steelblue')
    ax1.set_title("Distribution train")
    ax1.tick_params(axis='x', rotation=45)

    ax2.bar(CLASS_NAMES, np.bincount(y_test.flatten()), color='coral')
    ax2.set_title("Distribution test")
    ax2.tick_params(axis='x', rotation=45)
    fig.suptitle("Distribution")

    plt.tight_layout()
    return plt.show()

# Matrice de confusion du modèle de base
def confusion_matrices(y_pred, y_test) -> Any:
    cm = confusion_matrix(y_test.flatten(), y_pred)
    fig, ax = plt.subplots(figsize=(8, 8))
    disp = ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES)
    disp.plot(ax=ax, cmap='Blues', xticks_rotation=45, values_format='d')
    ax.set_title("Matrice de confusion - Modèle de base (1 epoch)")
    fig.suptitle("Matrice de confusion")
    plt.tight_layout()
    return plt.show()


# Exemples d\'erreurs de classification
def classification_error_samples(y_pred, x_test, y_test) -> Any:
    errors = np.where(y_pred != y_test.flatten())[0]
    sample = np.random.choice(errors, 10, replace=False)
    fig, axes = plt.subplots(2, 5, figsize=(14, 6))
    for ax, idx in zip(axes.flat, sample):
        ax.imshow(x_test[idx])
        ax.set_title(f"Vrai: {CLASS_NAMES[y_test[idx,0]]}\nPrédit: {CLASS_NAMES[y_pred[idx]]}", fontsize=9)
        ax.axis('off')
    fig.suptitle("Exemples mal classifiés")
    plt.tight_layout()
    return plt.show()

# Confiance du modèle sur les erreurs
def error_confidence(y_proba, y_pred, y_test) -> Any:
    confidence = y_proba.max(axis=1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    correct = y_pred == y_test.flatten()
    ax1.hist(confidence[correct], bins=30, alpha=0.7, label='Correct', color='green')
    ax1.hist(confidence[~correct], bins=30, alpha=0.7, label='Erreur', color='red')
    ax1.set_xlabel('Confiance (max softmax)')
    ax1.set_ylabel('Nombre')
    ax1.set_title('Distribution de confiance')
    ax1.legend()

    # Accuracy par classe
    cm = confusion_matrix(y_test.flatten(), y_pred)
    acc_per_class = cm.diagonal() / cm.sum(axis=1)
    ax2.barh(CLASS_NAMES, acc_per_class, color='steelblue')
    ax2.set_xlim(0, 1)
    ax2.set_xlabel('Accuracy')
    ax2.set_title('Accuracy par classe')

    fig.suptitle("Confiance Erreur")
    plt.tight_layout()
    return plt.show()

def learning_curve(history_fit) -> Any:
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history_fit.history['loss'], label='Perte d\'entraînement')
    plt.plot(history_fit.history['val_loss'], label='Perte de validation')
    plt.legend()
    plt.title('Courbe de Perte')

    plt.subplot(1, 2, 2)
    plt.plot(history_fit.history['accuracy'], label='Précision d\'entraînement')
    plt.plot(history_fit.history['val_accuracy'], label='Précision de validation')
    plt.legend()
    plt.title('Courbe de Précision')

    return plt.show()


# ---------------------------------------------------------------------------
# Firewall log EDA (projet SVM multiclasse)
# ---------------------------------------------------------------------------

def set_firewall_plot_style() -> None:
    """Style cohérent pour les figures du projet firewall."""
    sns.set_theme(style="whitegrid", context="notebook", font_scale=1.0)


def plot_firewall_action_distribution(df: pd.DataFrame, action_col: str = "Action") -> Any:
    """
    Distribution des classes avec annotation du pourcentage.
    """
    set_firewall_plot_style()
    counts = df[action_col].value_counts().sort_values(ascending=False)
    total = counts.sum()
    _, ax = plt.subplots(figsize=(8, 4.5))
    colors = sns.color_palette("Set2", n_colors=len(counts))
    bars = ax.bar(counts.index, counts.values, color=colors, edgecolor="black", linewidth=0.5)
    for bar, v in zip(bars, counts.values):
        pct = 100.0 * v / max(total, 1)
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{v}\n({pct:.2f}%)",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_title("Distribution des classes (Action)")
    ax.set_xlabel("Classe")
    ax.set_ylabel("Nombre d'instances")
    plt.tight_layout()
    return plt.show()


def plot_firewall_feature_boxplots(
    df: pd.DataFrame,
    features: list[str],
    action_col: str = "Action",
    ncols: int = 3,
) -> Any:
    """
    Boxplots par classe pour chaque feature numérique.
    Les features sont affichées en échelle log sur Y pour améliorer la lisibilité.
    """
    set_firewall_plot_style()
    n = len(features)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.6 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for i, feat in enumerate(features):
        ax = axes[i]
        sns.boxplot(data=df, x=action_col, y=feat, ax=ax, showfliers=False, palette="Set3")
        ax.set_title(feat)
        ax.set_yscale("log")
        ax.tick_params(axis="x", rotation=20)
    for j in range(n, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Distribution des variables numériques par classe (boxplot, échelle log)", y=1.02)
    plt.tight_layout()
    return plt.show()


def plot_firewall_correlation_heatmap(df: pd.DataFrame, features: list[str]) -> Any:
    """
    Heatmap des corrélations entre variables numériques.
    """
    set_firewall_plot_style()
    corr = df[features].corr(method="spearman")
    _, ax = plt.subplots(figsize=(10.5, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr,
        mask=mask,
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        center=0,
        annot=False,
        square=True,
        linewidths=0.4,
        cbar_kws={"shrink": 0.75},
        ax=ax,
    )
    ax.set_title("Corrélations (Spearman) entre features")
    plt.tight_layout()
    return plt.show()


def plot_firewall_reset_both_focus(
    df: pd.DataFrame,
    features: list[str],
    action_col: str = "Action",
    max_features: int = 4,
) -> Any:
    """
    Compare `reset-both` vs `autres classes` sur quelques variables.
    """
    set_firewall_plot_style()
    feats = features[:max_features]
    tmp = df.copy()
    tmp["Action_group"] = np.where(tmp[action_col].eq("reset-both"), "reset-both", "others")
    n = len(feats)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.2))
    axes = np.atleast_1d(axes)
    for ax, feat in zip(axes, feats):
        sns.violinplot(
            data=tmp,
            x="Action_group",
            y=feat,
            inner="quartile",
            cut=0,
            ax=ax,
            palette=["#dd8452", "#4c72b0"],
        )
        ax.set_yscale("log")
        ax.set_title(feat)
    fig.suptitle("Focus classe rare: reset-both vs autres (violin, échelle log)", y=1.03)
    plt.tight_layout()
    return plt.show()