"""
Visualization module: training curves, phase profiles, confusion matrix, E-field plots.

Reference:
Marzban, M.-R., Zarei, S., & Khavasi, A. (2020).
Optics Express, 28(24), 36668. https://doi.org/10.1364/OE.404386
"""

import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from sklearn.metrics import confusion_matrix

from .data_utils import DIM


def plot_training_curves(cost_history, test_acc_history, save_path=None):
    """
    Plot training cost (left axis) and test accuracy (right axis) vs. epoch.

    Reproduces Fig. 4 of the paper.

    Args:
        cost_history:     List of per-epoch cost values
        test_acc_history: List of per-epoch test accuracy (%)
        save_path:        Optional path to save the figure (PNG)
    """
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(111, label="cost")
    ax2 = fig.add_subplot(111, label="acc", frame_on=False)

    ax1.plot(cost_history, color="C0", linewidth=3, label="Cost")
    ax1.set_xlabel("Epochs", fontsize=18)
    ax1.set_ylabel("Cost", fontsize=18)

    ax2.plot(
        [a / 100 for a in test_acc_history],
        color="C3",
        linewidth=3,
        label="Test Accuracy",
    )
    ax2.yaxis.tick_right()
    ax2.set_ylabel("Accuracy [%]", fontsize=18)
    ax2.yaxis.set_label_position("right")
    plt.setp(ax2.get_xticklabels(), visible=False)

    plt.rcParams.update({"font.size": 18})

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def visualize_phase_profiles(Phasetarget_1, Phasetarget_2, Phasetarget_3,
                              dim=DIM, save_dir=None):
    """
    Plot the optimised phase profile of each metaline layer as a heatmap.

    Reproduces Fig. 5 of the paper.

    Args:
        Phasetarget_1/2/3: Phase arrays (rad), shape (dim²,)
        dim:               Spatial dimension
        save_dir:          If provided, save PNGs here
    """
    layers = [Phasetarget_1, Phasetarget_2, Phasetarget_3]
    titles = ["Phase Profile (Layer 1)", "Phase Profile (Layer 2)", "Phase Profile (Layer 3)"]

    for i, (ph, title) in enumerate(zip(layers, titles), start=1):
        fig, ax = plt.subplots(figsize=(5, 5))
        sns.heatmap(
            -ph.reshape(dim, dim),
            linewidths=0.0,
            cmap="jet",
            vmin=0,
            vmax=2 * math.pi,
            yticklabels=False,
            xticklabels=False,
            ax=ax,
        )
        ax.set_title(title, fontsize=18)
        plt.tight_layout()
        if save_dir:
            fig.savefig(Path(save_dir) / f"Phasetarget_{i}.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()


def plot_confusion_matrix(
    y_true,
    y_pred,
    classes,
    normalize=True,
    title="Normalized confusion matrix",
    cmap=plt.cm.Blues,
    save_path=None,
):
    """
    Plot (and optionally save) a confusion matrix.

    Reproduces Fig. 6(c) of the paper.

    Args:
        y_true:     Ground-truth labels
        y_pred:     Predicted labels
        classes:    Class-name list (e.g. ['0','1',...,'9'])
        normalize:  If True, normalise each row to [0, 1]
        title:      Figure title
        cmap:       Colormap
        save_path:  Optional path to save PNG
    """
    cm_mat = confusion_matrix(y_true, y_pred)
    if normalize:
        cm_mat = cm_mat.astype("float") / cm_mat.sum(axis=1, keepdims=True)
        cm_mat = np.round(cm_mat, decimals=2)

    fig, ax = plt.subplots(figsize=(9.5, 9.5))
    im = ax.imshow(cm_mat, interpolation="nearest", cmap=cmap)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        xticks=np.arange(cm_mat.shape[1]),
        yticks=np.arange(cm_mat.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        title=title,
        ylabel="True label",
        xlabel="Predicted label",
    )
    plt.ylabel("True label", fontsize=20)
    plt.xlabel("Predicted label", fontsize=20)
    plt.title(title, fontsize=20, y=1.02)
    plt.setp(ax.get_xticklabels(), rotation=0, ha="right", rotation_mode="anchor")

    fmt = ".2f" if normalize else "d"
    thresh = cm_mat.max() / 2.0
    for i in range(cm_mat.shape[0]):
        for j in range(cm_mat.shape[1]):
            ax.text(
                j, i, format(cm_mat[i, j], fmt),
                ha="center", va="center",
                color="white" if cm_mat[i, j] > thresh else "black",
            )
    fig.tight_layout()
    plt.xlim(-0.5, len(classes) - 0.5)
    plt.ylim(len(classes) - 0.5, -0.5)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
    return ax


def visualize_efield(E_image, title="E-field Distribution", save_path=None):
    """
    Visualise the x-y electric field distribution from a Lumerical simulation.

    Reproduces Fig. 10/12 of the paper.

    Args:
        E_image:    Dict with keys 'x', 'y', 'E' (as returned by Lumerical MODE API)
        title:      Figure title
        save_path:  Optional path to save PNG
    """
    x = E_image["x"] * 1e6
    y = E_image["y"] * 1e6
    Ex = E_image["E"][:, :, 0, 0, 0]
    Ey = E_image["E"][:, :, 0, 0, 1]
    Ez = E_image["E"][:, :, 0, 0, 2]
    E = np.sqrt(np.sqrt(np.abs(Ex)) + np.sqrt(np.abs(Ey)) + np.sqrt(np.abs(Ez)))

    fig, ax = plt.subplots(figsize=(15, 10))
    im = ax.imshow(
        np.transpose(E),
        aspect="equal",
        interpolation="bicubic",
        cmap=cm.bwr,
        origin="lower",
        extent=[x.min(), x.max(), y.min(), y.max()],
        vmax=0.82 * E.max(),
        vmin=0.91,
    )
    plt.colorbar(im, ax=ax)
    ax.set_xlabel("x (µm)", fontsize=30)
    ax.set_ylabel("y (µm)", fontsize=30)
    ax.set_title(title, fontsize=24)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
