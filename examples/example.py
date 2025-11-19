"""
Usage Examples
===============

These examples demonstrate a simple usage of the BDE package both for
regression and classification tasks.
"""

import logging
import os
import sys

import pandas as pd
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logging.getLogger("bde").setLevel(logging.INFO)

import jax
import jax.numpy as jnp
from jax.scipy.stats import norm
from sklearn.datasets import fetch_openml, load_iris
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split

from bde import BdeClassifier, BdeRegressor
from bde.loss.loss import CategoricalCrossEntropy, GaussianNLL


def regression_example():
    print("-" * 20)
    print("Regression example")
    print("-" * 20)
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
        n_members=8,
        seed=0,
        loss=GaussianNLL(),
        epochs=200,
        lr=1e-3,
        warmup_steps=5000,  # 50k in the original paper
        n_samples=2000,  # 10k in the original paper
        n_thinning=2,
        patience=10,
    )

    print(f"the params are {regressor.get_params()}")  # get_params is from sklearn!
    regressor.fit(x=Xtr, y=ytr)

    means, sigmas = regressor.predict(Xte, mean_and_std=True)

    print("RSME: ", root_mean_squared_error(y_true=yte, y_pred=means))
    mean, intervals = regressor.predict(Xte, credible_intervals=[0.1, 0.9])
    raw = regressor.predict(Xte, raw=True)
    print(
        f"The shape of the raw predictions are {raw.shape}"
    )  # (ensemble members, n_samples/n_thinning, n_data, (mu,sigma))

    # use the raw predictions to compute log pointwise predictive density (lppd)

    n_data = yte.shape[0]
    log_likelihoods = norm.logpdf(
        yte.reshape(1, 1, n_data),
        loc=raw[:, :, :, 0],
        scale=jax.nn.softplus(raw[..., 1]) + 1e-6,  # map raw scale via softplus
    )  # (E,T,N)
    b = 1 / jnp.prod(jnp.array(log_likelihoods.shape[:-1]))  # 1/ET
    axis = tuple(range(len(log_likelihoods.shape) - 1))
    log_likelihoods = jax.scipy.special.logsumexp(log_likelihoods, b=b, axis=axis)
    lppd = jnp.mean(log_likelihoods)
    print(f"The log pointwise predictive density (lppd) is {lppd}")

    print("Quantiles shape:", intervals.shape)  # (len(q), N)
    # calculate the coverage of the 80% credible interval
    lower = intervals[0]
    upper = intervals[1]
    coverage = jnp.mean((yte.ravel() >= lower) & (yte.ravel() <= upper))
    print(f"Coverage of the 80% credible interval: {coverage * 100:.2f}%")

    score = regressor.score(Xte, yte)
    print(f"The sklearn test score is {score}")


def classification_example():
    print("-" * 20)
    print("Classification example")
    print("-" * 20)
    iris = load_iris()
    X = iris.data.astype("float32")
    y = iris.target.astype("int32").ravel()  # 0, 1, 2

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    classifier = BdeClassifier(
        n_members=4,
        hidden_layers=[16, 16],
        seed=0,
        loss=CategoricalCrossEntropy(),
        activation="relu",
        epochs=100,
        lr=1e-3,
        warmup_steps=400,  # very few steps required for this simple dataset
        n_samples=100,
        n_thinning=1,
        patience=10,
    )

    classifier.fit(x=X_train, y=y_train)

    preds = classifier.predict(X_test)
    probs = classifier.predict_proba(X_test)
    print("Predicted class probabilities shape:\n", probs.shape)
    accuracy = jnp.mean(preds == y_test)
    print(f"Test accuracy: {accuracy * 100:.2f}%")
    score = classifier.score(X_train, y_train)
    print(f"The sklearn score is {score}")
    raw = classifier.predict(X_test, raw=True)
    print(
        f"The shape of the raw predictions are {raw.shape}"
    )  # (ensemble members, n_samples, n_test_data, n_classes))


def concrete_data_example():
    print("-" * 20)
    print("Regression example for concrete data")
    print("-" * 20)
    scaler = StandardScaler()
    data = pd.read_csv("bde/data/concrete.data", sep=" ", header=None)
    data_norm = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

    X = data_norm.iloc[:, :-1]  # shape (1038, 8)
    y = data_norm.iloc[:, -1]  # shape (1038, )
    X = jnp.array(X)
    y = jnp.array(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    regressor = BdeRegressor(
        hidden_layers=[16, 16],
        n_members=8,
        seed=0,
        loss=GaussianNLL(),
        epochs=200,
        lr=1e-3,
        warmup_steps=50000,  # 50k in the original paper
        n_samples=10000,  # 10k in the original paper
        n_thinning=10,
        patience=10,
    )

    print(f"the params are {regressor.get_params()}")  # get_params is from sklearn!
    regressor.fit(x=X_train, y=y_train)

    means, sigmas = regressor.predict(X_test, mean_and_std=True)

    print("RSME: ", root_mean_squared_error(y_true=y_test, y_pred=means))
    mean, intervals = regressor.predict(X_test, credible_intervals=[0.1, 0.9])
    raw = regressor.predict(X_test, raw=True)
    print(
        f"The shape of the raw predictions are {raw.shape}"
    )  # (ensemble members, n_samples/n_thinning, n_data, (mu,sigma))

    # use the raw predictions to compute log pointwise predictive density (lppd)
    n_data = y_test.shape[0]
    log_likelihoods = norm.logpdf(
        y_test.reshape(1, 1, n_data),
        loc=raw[:, :, :, 0],
        scale=jax.nn.softplus(raw[..., 1]) + 1e-6,  # map raw scale via softplus
    )  # (E,T,N)
    b = 1 / jnp.prod(jnp.array(log_likelihoods.shape[:-1]))  # 1/ET
    axis = tuple(range(len(log_likelihoods.shape) - 1))
    log_likelihoods = jax.scipy.special.logsumexp(log_likelihoods, b=b, axis=axis)
    lppd = jnp.mean(log_likelihoods)
    print(f"The log pointwise predictive density (lppd) is {lppd}")

    print("Quantiles shape:", intervals.shape)  # (len(q), N)
    # calculate the coverage of the 80% credible interval
    lower = intervals[0]
    upper = intervals[1]
    coverage = jnp.mean((y_test.ravel() >= lower) & (y_test.ravel() <= upper))
    print(f"Coverage of the 80% credible interval: {coverage * 100:.2f}%")

    score = regressor.score(X_test, y_test)
    print(f"The sklearn score is {score}")


if __name__ == "__main__":
    # classification_example()
    # regression_example()
    concrete_data_example()
