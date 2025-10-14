"""
Minimal Example
===============

This example demonstrates a simple usage of the BDE package.
"""

import logging
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logging.getLogger("bde").setLevel(logging.INFO)

import jax.numpy as jnp
from sklearn.datasets import fetch_openml, load_iris
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split

from bde import BdeClassifier, BdeRegressor
from bde.loss.loss import CategoricalCrossEntropy, GaussianNLL
from bde.viz.plotting import (
    plot_pred_vs_true,
)


def regression_example():
    data = fetch_openml(name="airfoil_self_noise", as_frame=True)

    X = data.data.values  # shape (1503, 5)
    y = data.target.values.reshape(-1, 1)  # shape (1503, 1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    Xmu, Xstd = jnp.mean(X_train, 0), jnp.std(X_train, 0) + 1e-8
    Ymu, Ystd = jnp.mean(y_train, 0), jnp.std(y_train, 0) + 1e-8

    Xtr = (X_train - Xmu) / Xstd
    Xte = (X_test - Xmu) / Xstd
    ytr = (y_train - Ymu) / Ystd
    yte = (y_test - Ymu) / Ystd

    regressor = BdeRegressor(
        hidden_layers=[16, 16],
        n_members=20,
        seed=0,
        loss=GaussianNLL(),
        epochs=200,
        lr=1e-3,
        warmup_steps=500,
        n_samples=100,
        n_thinning=1,
        patience=10,
    )

    print(f"the params are {regressor.get_params()}")  # get_params is from sk learn!
    regressor.fit(x=Xtr, y=ytr)

    means, sigmas = regressor.predict(Xte, mean_and_std=True)

    print("RSME: ", root_mean_squared_error(y_true=yte, y_pred=means))
    plot_pred_vs_true(
        y_pred=means,
        y_true=yte,
        y_pred_err=sigmas,
        title="trial",
        savepath="plots_regression",
    )

    mean, intervals = regressor.predict(Xte, credible_intervals=[0.9, 0.95])
    raw = regressor.predict(Xte, raw=True)
    print(
        f"The shape of the raw predictions are {raw.shape}"
    )  # (ensemble members, n_samples, n_data, (mu,sigma))

    print("Credible intervals shape:", intervals.shape)  # (len(q), N)

    # for plotting, pick the 95% interval
    lower = intervals[0]  # q=0.9 or 0.95 depending on order
    upper = intervals[1]  # if you asked for 2 quantiles

    plot_pred_vs_true(
        y_pred=means,
        y_true=yte,
        y_pred_err=(upper - lower) / 2,  # approx half-width as "sigma"
        title="trial_with_intervals",
        savepath="plots_regression",
    )

    score = regressor.score(Xtr, ytr)
    print(f"the sklearn score is {score}")


def classification_example():
    iris = load_iris()
    X = iris.data.astype("float32")
    y = iris.target.astype("int32").ravel()  # 0, 1, 2

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Convert to JAX
    Xtr, Xte = jnp.array(X_train), jnp.array(X_test)
    ytr, yte = jnp.array(y_train), jnp.array(y_test)

    classifier = BdeClassifier(
        n_members=2,
        hidden_layers=[16, 16],
        seed=0,
        loss=CategoricalCrossEntropy(),
        activation="relu",
        epochs=4,
        lr=1e-3,
        warmup_steps=50,
        n_samples=2,
        n_thinning=1,
        patience=2,
    )

    classifier.fit(x=Xtr, y=ytr)

    preds = classifier.predict(Xte)
    probs = classifier.predict_proba(Xte)
    print("Predicted class probabilities:\n", probs)
    print("Predicted class labels:\n", preds)
    print("True labels:\n", yte)
    score = classifier.score(Xtr, ytr)
    print(f"the sklearn score is {score}")
    raw = classifier.predict(Xte, raw=True)
    print(
        f"The shape of the raw predictions are {raw.shape}"
    )  # (ensemble members, n_samples, n_data, n_classes))


if __name__ == "__main__":
    classification_example()
    regression_example()
