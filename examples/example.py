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
from sklearn.datasets import load_iris
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from bde import BdeClassifier, BdeRegressor
from bde.loss.loss import CategoricalCrossEntropy, GaussianNLL
from bde.sampler.prior import PriorDist


def regression_airfoil_example():
    print("-" * 20)
    print("Regression example")
    print("-" * 20)

    data = pd.read_csv("bde/data/airfoil.csv", sep=",", header=0)

    X = data.iloc[:, :-1].values  # (1503, 5)
    y = data.iloc[:, -1].values  # (1503,)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_train = x_scaler.fit_transform(X_train)
    X_test = x_scaler.transform(X_test)

    y_train = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_test = y_scaler.transform(y_test.reshape(-1, 1)).ravel()

    regressor = BdeRegressor(
        hidden_layers=[16, 16],
        n_members=8,
        seed=0,
        loss=GaussianNLL(),
        epochs=1000,
        validation_split=0.15,
        lr=1e-3,
        weight_decay=1e-3,
        warmup_steps=5000,  # 50k in the original paper
        n_samples=1000,  # 10k in the original paper
        n_thinning=1,
        patience=20,
    )

    print(f"the params are {regressor.get_params()}")  # get_params is from sklearn!
    regressor.fit(x=jnp.array(X_train), y=jnp.array(y_train))

    means, sigmas = regressor.predict(jnp.array(X_test), mean_and_std=True)

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
    print(f"The sklearn test score is {score}")

    # print(f"This is the history of the regressor\n {regressor.history()}")


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
        epochs=1000,
        validation_split=0.15,
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
    )  # (ensemble members, n_samples/n_thinning, n_test_data, n_classes)


def regression_concrete_example():
    print("-" * 20)
    print("Regression example for concrete data")
    print("-" * 20)

    data = pd.read_csv("bde/data/concrete.data", sep=" ", header=None)

    X = data.iloc[:, :-1].values  # (1030, 8)
    y = data.iloc[:, -1].values  # (1030,)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_train = x_scaler.fit_transform(X_train)
    X_test = x_scaler.transform(X_test)

    y_train = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_test = y_scaler.transform(y_test.reshape(-1, 1)).ravel()

    regressor = BdeRegressor(
        hidden_layers=[16, 16],
        n_members=8,
        seed=0,
        loss=GaussianNLL(),
        epochs=1000,
        validation_split=0.15,
        lr=1e-3,
        weight_decay=1e-4,
        warmup_steps=50000,
        n_samples=10000,
        n_thinning=0,
        patience=10,
        prior_family=PriorDist.NORMAL,
        prior_kwargs={"loc": 0, "scale": 1.5},
    )

    print(f"The params are {regressor.get_params()}")
    regressor.fit(x=jnp.array(X_train), y=jnp.array(y_train))

    means, sigmas = regressor.predict(jnp.array(X_test), mean_and_std=True)
    rmse = root_mean_squared_error(y_true=y_test, y_pred=means)
    print("RMSE:", rmse)
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


def sklearn_pipeline():
    print("-" * 20)
    print("Sklearn Pipeline with GridSearchCV example")
    print("-" * 20)

    base_estimator = BdeClassifier(
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

    pipe = Pipeline(
        [
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("clf", base_estimator),
        ]
    )

    # Explore relevant BDE hyperparameters while keeping the search lightweight.
    param_grid = {
        "clf__n_members": [4],
        "clf__hidden_layers": [[8, 8], [16, 16]],
        "clf__epochs": [20],
        "clf__lr": [1e-3],
        "clf__warmup_steps": [400],
        "clf__n_samples": [100],
        "scaler__with_std": [True],
    }

    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring={"accuracy": "accuracy", "f1_macro": "f1_macro"},
        refit="f1_macro",
        n_jobs=None,  # single-threaded for clarity; set to -1 if you want parallel
        cv=3,
        verbose=1,
        return_train_score=True,
    )
    iris = load_iris()
    X = iris.data.astype("float32")
    y = iris.target.astype("int32").ravel()  # 0, 1, 2

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    grid.fit(X_train, y_train)

    print("Best params:", grid.best_params_)
    print("Best CV f1_macro: {:.4f}".format(grid.best_score_))


if __name__ == "__main__":
    regression_airfoil_example()
    classification_example()
    regression_concrete_example()
    sklearn_pipeline()
