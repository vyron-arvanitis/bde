"""This here has some plotters!"""

import matplotlib.pyplot as plt
import os

def plot_pred_vs_true(y_pred, y_true, title, savepath):
    """
    #TODO: documentation
    Parameters
    ----------
    y_pred
    y_true
    title
    savepath

    Returns
    -------

    """

    fig, ax = plt.subplots()
    ax.scatter(y_true, y_pred, s=15, alpha=0.6, edgecolors="k", linewidths=0.2)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", linewidth=1, label="y = x")
    ax.set_xlabel("True values")
    ax.set_ylabel("Predicted values")
    ax.set_title(title)
    ax.legend()

    os.makedirs(savepath, exist_ok=True)

    if savepath is not None:
        plt.savefig(os.path.join(savepath, title), bbox_inches="tight")


