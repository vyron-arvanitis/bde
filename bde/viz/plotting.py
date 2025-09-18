"""This here has some plotters!"""

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve, auc
import seaborn as sns
import numpy as np
import os
import matplotlib.gridspec as gridspec



def plot_pred_vs_true(y_pred, y_true, title, savepath,y_pred_err=None,):
    """
    #TODO: documentation
    Parameters
    ----------
    y_pred
    y_true
    y_pred_err
    title
    savepath

    Returns
    -------

    """

    def _to_1d(a):
        a = np.asarray(a)  # handles JAX DeviceArray too
        a = np.squeeze(a)  # drop singleton dims
        return a.reshape(-1)  # ensure 1D

    y_true_1d = _to_1d(y_true)
    y_pred_1d = _to_1d(y_pred)

    yerr = None
    if y_pred_err is not None:
        err = np.asarray(y_pred_err)
        # if it's (N,1) or (N,), squeeze to 1D
        err = np.squeeze(err)
        # if it looks like variance (non-negative), take sqrt to get std
        if np.all(err >= 0):
            err = np.sqrt(err)
        yerr = err.reshape(-1)

    fig = plt.figure(figsize=(7, 7))
    gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1], hspace=0.05)
    ax_main = plt.subplot(gs[0])

    ax_main.errorbar(y_true_1d, y_pred_1d, yerr=yerr, fmt='o', color="#539ecd", alpha=0.5)

    lim_min = float(min(y_true_1d.min(), y_pred_1d.min()))
    lim_max = float(max(y_true_1d.max(), y_pred_1d.max()))
    ax_main.plot([lim_min, lim_max], [lim_min, lim_max], "r--", linewidth=1, label="y = x")

    ax_main.set_xlabel("True values")
    ax_main.set_ylabel("Predicted values")
    ax_main.set_title(title)
    ax_main.legend()
    ax_main.grid(True)

    ax_res = plt.subplot(gs[1], sharex=ax_main)
    pull = (y_pred_1d - y_true_1d) / yerr
    ax_res.axhline(0, color="black", linestyle="--")
    ax_res.scatter(y_true, pull, alpha=0.5, s=10)
    # ax_res.errorbar(y_true, pull, yerr=np.std(pull))
    ax_res.set_xlabel(r"True values")
    ax_res.set_ylim(-2, 2)
    ax_res.set_ylabel("Pull")
    ax_res.grid(True)

    if savepath is not None:
        os.makedirs(savepath, exist_ok=True)
        plt.savefig(os.path.join(savepath, f"{title}.png"), bbox_inches="tight")
    plt.close(fig)

def plot_confusion_matrix(y_true, y_pred, classes, title, savepath=None):
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    fig, ax = plt.subplots(figsize=(7, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)

    if savepath is not None:
        os.makedirs(savepath, exist_ok=True)
        plt.savefig(os.path.join(savepath, f"{title}_confusion.png"), bbox_inches="tight")
    plt.close(fig)

def plot_reliability_curve(y_true, y_proba, n_bins=10, title="", savepath=None):
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=n_bins)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(prob_pred, prob_true, "s-", label="Calibration curve")
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("True fraction of positives")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)

    if savepath is not None:
        os.makedirs(savepath, exist_ok=True)
        plt.savefig(os.path.join(savepath, f"{title}_calibration.png"), bbox_inches="tight")
    plt.close(fig)


def plot_roc_curve(y_true, y_proba, title="", savepath=None):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
    ax.plot([0, 1], [0, 1], "k--", lw=2)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")

    if savepath is not None:
        os.makedirs(savepath, exist_ok=True)
        plt.savefig(os.path.join(savepath, f"{title}_roc.png"), bbox_inches="tight")
    plt.close(fig)