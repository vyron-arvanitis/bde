.. _quick_start:

###############
Quick start
###############

`bde` implements Bayesian Deep Ensembles that plug directly into scikit-learn
pipelines while running training and sampling in JAX. This page walks through
setting up the environment, running your first estimators, and validating the
installation.

Installation
============

The package will be published on PyPI soon. Until then you can install the
current source build:

.. prompt:: bash $

  pip install git+https://github.com/scikit-learn-contrib/bde.git

If you prefer reproducible environments with GPU or multi-device CPU support,
consider using `pixi <https://prefix.dev/docs/pixi/overview>`__ as described in
:ref:`user_guide`. The repo already ships with a ``pixi.toml`` that pins the
required versions of JAX, scikit-learn, and the CUDA toolchain when available.

Set the JAX device count
========================

When you run the estimators outside of the examples shipped in ``examples/``,
ensure JAX can see enough host devices. The environment variable below makes
CPU-only runs allocate eight virtual devices; tweak the value to match your
hardware:

.. prompt:: bash $

  export XLA_FLAGS="--xla_force_host_platform_device_count=8"

You can set it once per shell session or inject it programmatically before
importing JAX in your Python scripts.

Run a regression model
======================

The snippet below creates a toy regression dataset and fits a
:class:`bde.BdeRegressor` instance. It mirrors the script in
``examples/plot_regression.py``.

.. code-block:: python

   import os

   os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

   import jax.numpy as jnp
   from sklearn.datasets import fetch_openml
   from sklearn.metrics import root_mean_squared_error
   from bde import BdeRegressor
   from bde.loss import GaussianNLL


   data = fetch_openml(name="airfoil_self_noise", as_frame=True)

    X = data.data.values  # shape (1503, 5)
    y = data.target.values.reshape(-1, 1)  # shape (1503, 1)

   X_train, X_test, y_train, y_test = train_test_split(
       X,
       y,
       test_size=0.2,
       random_state=0,
   )
   # Normalize data

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

    regressor.fit(x=Xtr, y=ytr)

   mean, std = reg.predict(jnp.array(X_test), mean_and_std=True)
   mu, intervals = regressor.predict(Xte, credible_intervals=[0.9, 0.95])
   raw = regressor.predict(Xte, raw=True)
   print("RSME: ", root_mean_squared_error(y_true=yte, y_pred=means))
   score = regressor.score(Xtr, ytr)
   print(f"the sklearn score is {score}")


Run a classification model
==========================

:class:`bde.BdeClassifier` follows the same API and works with standard
datasets such as Iris:

.. code-block:: python

   import os

   os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

   from sklearn.datasets import load_iris
   from sklearn.model_selection import train_test_split

   from bde import BdeClassifier
   from bde.loss import CategoricalCrossEntropy

   iris = load_iris()
   X = iris.data.astype("float32")
   y = iris.target.astype("int32").ravel()
   X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
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
        patience=2
        )
   classifier.fit(x=X_train, y=y_train)
   preds = classifier.predict(X_test)
   probs = classifier.predict_proba(X_test)
   score = classifier.score(X_train, y_train)
   raw = classifier.predict(X_test, raw=True)
   print("Predicted class probabilities:\n", probs)
   print("Predicted class labels:\n", preds)
   print("True labels:\n", y_test)
   print(f"the sklearn score is {score}")
   print(f"The shape of the raw predictions are {raw.shape}")

Work with the development environment
=====================================

Once you clone the repository, bootstrap the dev dependencies and tooling with:

.. prompt:: bash $

  pixi install

You can then run the standard tasks:

* ``pixi run lint`` for style checks (``ruff`` and ``black``)
* ``pixi run test`` for the pytest suite, including scikit-learn compatibility
  checks in ``tests/test_common.py``
* ``pixi run build-doc`` to render this documentation locally with Sphinx and
  sphinx-gallery examples

If you prefer calling Python tooling directly, enter the environment shell:

.. prompt:: bash $

  pixi shell -e dev

From there ``pytest``, ``ruff`` and ``sphinx-build`` are available on ``PATH``.

Next steps
==========

* Dive into :ref:`user_guide` for a walkthrough of the estimator internals and
  configuration knobs.
* Explore the auto-generated API reference in :ref:`api` to see every public
  class, function, and dataclass.
* Run the gallery in :ref:`general_examples` to compare regression and
  classification behaviours on real-world datasets.
