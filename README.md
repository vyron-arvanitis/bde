bde: Bayesian Deep Ensembles for scikit-learn and JAX
====================================================

![tests](https://github.com/scikit-learn-contrib/bde/actions/workflows/python-app.yml/badge.svg)
[![codecov](https://codecov.io/gh/scikit-learn-contrib/bde/graph/badge.svg?token=L0XPWwoPLw)](https://codecov.io/gh/scikit-learn-contrib/bde)
![doc](https://github.com/scikit-learn-contrib/bde/actions/workflows/deploy-gh-pages.yml/badge.svg)

Introduction
------------

**bde** is a user-friendly implementation of Bayesian Deep Ensembles compatible with
both scikit-learn and JAX. It exposes estimators that plug into scikit-learn
pipelines while leveraging JAX for accelerator-backed training, sampling, and
uncertainty estimation.

Installation
------------

```
pip install --index-url <pending-release-url> bde
```

The public package index is not published yet; the command above is a placeholder
for the upcoming release.

Dependency Management
---------------------

We recommend using [pixi](https://prefix.dev/docs/pixi/overview) to create a
deterministic development environment:

```
pixi install
pixi run python -m examples.example
```

Pixi ensures the correct JAX, CUDA (when needed), and scikit-learn versions are
selected automatically. See `pixi.toml` for channel and platform details.

Example Usage
-------------

Minimal runnable scripts live in `examples/`, and the snippets below highlight the
most common regression and classification workflows.

### Regression Example

```python
import jax.numpy as jnp
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

from bde import BdeRegressor
from bde.loss import GaussianNLL

# Generate and split a toy regression problem
X, y = make_regression(
    n_samples=500,
    n_features=8,
    noise=0.3,
    random_state=42,
)
X = X.astype("float32")
y = y.astype("float32")
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=0,
)

# Convert to JAX arrays expected by the estimators
X_train = jnp.array(X_train)
y_train = jnp.array(y_train)
X_test = jnp.array(X_test)
y_test = jnp.array(y_test)

regressor = BdeRegressor(
    n_members=4,
    hidden_layers=[64, 64],
    seed=123,
    loss=GaussianNLL(),
    activation="relu",
    epochs=100,
    patience=10, #Note: Early stopping is always implemented
    n_samples=20,
    warmup_steps=100,
    lr=5e-4,
    n_thinning=2,
    desired_energy_var_start=0.5,
    desired_energy_var_end=0.1,
    step_size_init=0.01,
)

regressor.fit(x=X_train, y=y_train)
mean, std = regressor.predict(X_test, mean_and_std=True)
print("RMSE:", jnp.sqrt(jnp.mean((mean - y_test) ** 2)))
```

### Classification Example

```python
import jax.numpy as jnp
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from bde import BdeClassifier
from bde.loss import CategoricalCrossEntropy

# Prepare the Iris dataset
X, y = load_iris(return_X_y=True)
X = X.astype("float32")
y = y.astype("int32")
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=0,
    stratify=y,
)

# Convert to JAX arrays expected by the estimators
X_train = jnp.array(X_train)
y_train = jnp.array(y_train)
X_test = jnp.array(X_test)
y_test = jnp.array(y_test)

classifier = BdeClassifier(
    n_members=3,
    hidden_layers=[32, 32],
    seed=456,
    loss=CategoricalCrossEntropy(),
    activation="relu",
    epochs=50,
    patience=8, #Note: Early stopping is always implemented
    n_samples=15,
    warmup_steps=80,
    lr=1e-3,
    n_thinning=2,
    desired_energy_var_start=0.5,
    desired_energy_var_end=0.1,
    step_size_init=0.01,
)

classifier.fit(x=X_train, y=y_train)
preds = classifier.predict(X_test)
probs = classifier.predict_proba(X_test)
print("Predicted probabilities shape:", probs.shape)
accuracy = jnp.mean((jnp.array(preds) == y_test).astype(jnp.float32))
print("Accuracy:", float(accuracy))
```

Mathematical Background
-----------------------

