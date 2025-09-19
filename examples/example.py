import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=5"

from bde.bde import BdeRegressor, BdeClassifier
from bde.task import TaskType
from bde.loss.loss import *
from sklearn.datasets import fetch_openml, load_iris
from sklearn.model_selection import train_test_split
import jax.numpy as jnp

from bde.viz.plotting import plot_pred_vs_true, plot_confusion_matrix, plot_reliability_curve, plot_roc_curve


def regression_example():
    data = fetch_openml(name="airfoil_self_noise", as_frame=True)

    X = data.data.values  # shape (1503, 5)
    y = data.target.values.reshape(-1, 1)  # shape (1503, 1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Convert to JAX arrays
    X_train = jnp.array(X_train, dtype=jnp.float32)
    y_train = jnp.array(y_train, dtype=jnp.float32)
    X_test = jnp.array(X_test, dtype=jnp.float32)
    y_test = jnp.array(y_test, dtype=jnp.float32)

    Xmu, Xstd = jnp.mean(X_train, 0), jnp.std(X_train, 0) + 1e-8
    Ymu, Ystd = jnp.mean(y_train, 0), jnp.std(y_train, 0) + 1e-8

    Xtr = (X_train - Xmu) / Xstd
    Xte = (X_test - Xmu) / Xstd
    ytr = (y_train - Ymu) / Ystd
    yte = (y_test - Ymu) / Ystd

    sizes = [5, 16, 16, 2]  # TODO: [@later] allow user to configure only hidden layers\ -> this is done

    regressor = BdeRegressor(
        hidden_layers=[16, 16],
        n_members=11,
        seed=0,
        loss=GaussianNLL(),
        epochs=100,
        lr=1e-3,
        warmup_steps=500,
        n_samples=100,
        n_thinning=10,
    )

    print(f"the params are {regressor.get_params()}")  # get_params is from sk learn!!
    regressor.fit(X=Xtr, y=ytr)

    means, sigmas = regressor.predict(Xte, mean_and_std=True)

    print("RSME: ", jnp.sqrt(jnp.mean((means - yte) ** 2)))
    plot_pred_vs_true(
        y_pred=means,
        y_true=yte,
        y_pred_err=sigmas,
        title="trial",
        savepath="plots_regression"
    )

    mean, intervals = regressor.predict(Xte, credible_intervals=[0.9, 0.95])

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
    y = iris.target.astype("int32")  # 0, 1, 2

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert to JAX
    Xtr, Xte = jnp.array(X_train), jnp.array(X_test)
    ytr, yte = jnp.array(y_train), jnp.array(y_test)

    classifier = BdeClassifier(
        n_members=5,
        hidden_layers=[16, 16],
        seed=0,
        loss=CategoricalCrossEntropy(),
        activation="relu",
        epochs=50,
        lr=1e-3,
        warmup_steps=200,
        n_samples=50,
        n_thinning=5
    )

    classifier.fit(X=Xtr, y=ytr)

    preds = classifier.predict(Xte)
    probs = classifier.predict_proba(Xte)
    print("Predicted class probabilities:\n", probs)
    print("Predicted class labels:\n", preds)
    print("True labels:\n", yte)

    savepath = "plots_classification"
    classes = list(range(3))  # [0,1,2]

    # 1. Confusion matrix
    plot_confusion_matrix(
        y_true=jnp.array(yte),
        y_pred=jnp.array(preds),
        classes=classes,
        title="Iris Confusion Matrix",
        savepath=savepath,
    )

    # 2. Reliability curve (per class, e.g. class 0)
    plot_reliability_curve(
        y_true=(jnp.array(yte) == 0).astype(int),
        y_proba=jnp.array(probs)[:, 0],  # probability of class 0
        n_bins=10,
        title="Iris Calibration Curve (class 0)",
        savepath=savepath,
    )

    # 3. ROC curve (per class, e.g. class 0)
    plot_roc_curve(
        y_true=(jnp.array(yte) == 0).astype(int),
        y_proba=jnp.array(probs)[:, 0],
        title="Iris ROC Curve (class 0 vs rest)",
        savepath=savepath,
    )

    score = classifier.score(Xtr, ytr)
    print(f"the sklearn score is {score}")


if __name__ == "__main__":
    # regression_example()
    classification_example()
