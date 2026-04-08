from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from types import Any
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
def classification_error_examples(y_pred, x_test, y_test) -> Any:
    errors = np.where(y_pred != y_test.flatten())[0]
    sample = np.random.Generator(errors, 10, replace=False)
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
