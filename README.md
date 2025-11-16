bde: Bayesian Deep Ensembles for scikit-learn
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

In particular, **bde** implements **Microcanonical Langevin Ensembles (MILE)** as
introduced in [*Microcanonical Langevin Ensembles: Advancing the Sampling of Bayesian Neural Networks* (ICLR 2025)](https://arxiv.org/abs/2502.06335).
A conceptual overview of MILE is shown below (in general implementation details of this package are not exactly matching this diagram):

<div style="width: 60%; margin: auto;">
    <img src="doc/_static/img/flowchart.png" alt="MILE Overview" style="width: 100%;">
</div>

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
most common regression and classification workflows. When running outside those
scripts, remember to set the XLA device count so JAX allocates enough host devices (
this needs to be done before importing JAX):

```
export XLA_FLAGS="--xla_force_host_platform_device_count=8"
```

Adjust the value to match the number of CPU (or GPU) devices you plan to use.

### Regression Example

```python
import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

import jax.numpy as jnp
from sklearn.datasets import fetch_openml
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split

from bde import BdeRegressor
from bde.loss import GaussianNLL


data = fetch_openml(name="airfoil_self_noise", as_frame=True)

X = data.data.values
y = data.target.values.reshape(-1, 1)
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
    warmup_steps=5000, # 50k in the original paper
    n_samples=2000, # 10k in the original paper
    n_thinning=2,
    patience=10,
)

regressor.fit(x=Xtr, y=ytr)

means, sigmas = regressor.predict(Xte, mean_and_std=True)

print("RSME: ", root_mean_squared_error(y_true=yte, y_pred=means))

mean, intervals = regressor.predict(Xte, credible_intervals=[0.1, 0.9])

lower = intervals[0]
upper = intervals[1]
coverage = jnp.mean((yte.ravel() >= lower) & (yte.ravel() <= upper))
print(f"Coverage of the 80% credible interval: {coverage * 100:.2f}%")

```

### Classification Example

```python
import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from bde import BdeClassifier
from bde.loss import CategoricalCrossEntropy

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
    warmup_steps=400, # very few steps required for this simple dataset
    n_samples=100,
    n_thinning=1,
    patience=10,
)

classifier.fit(x=X_train, y=y_train)

preds = classifier.predict(X_test)
probs = classifier.predict_proba(X_test)

accuracy = jnp.mean(preds == y_test)
print(f"Test accuracy: {accuracy * 100:.2f}%")
```

Workflow
--------

The high-level estimators follow this flow during `fit` and evaluation:

- `BdeRegressor` / `BdeClassifier` (`bde/bde.py`) delegate to the shared `Bde` base class.
- `Bde.fit` validates data, resolves defaults, and calls `_build_bde()` to instantiate `BdeBuilder`.
- `BdeBuilder.fit_members` (`bde/bde_builder.py`) trains each network, handles device padding, and applies early stopping.
- `_build_log_post` constructs the ensemble log-posterior, then `warmup_bde` (`bde/sampler/warmup.py`) adapts step sizes before sampling.
- Sampler utilities (`bde/sampler/*`) draw posterior samples and cache them for downstream prediction.
- `Bde.evaluate` / predictor utilities (`bde/bde_evaluator.py`) aggregate samples into means, intervals, and probabilities.

```mermaid
flowchart TD
    subgraph User
        FitCall["Call Bde.fit(X, y)"]
    end

    subgraph Bde
        Validate["validate_fit_data / _prepare_targets"]
        Build["_build_bde()"]
        Builder["BdeBuilder"]
        Train["fit_members(X, y, optimizer, loss)"]
        LogPost["_build_log_post(X, y)"]
        WarmSampler["_warmup_sampler(logpost)"]
        Keys["_generate_rng_keys + _normalize_tuned_parameters"]
        Draw["_draw_samples(...)"]
        Cache["positions_eT_ stored in estimator"]
    end

    subgraph Warmup
        Warm["warmup_bde()"]
        Adapter["custom_mclmc_warmup adapter"]
        Adapt["per-member adaptation (pmap/vmap)"]
        Results["AdaptationResults: states_e, tuned params"]
    end

    subgraph Sampling
        Wrapper["MileWrapper"]
        Batch["sample_batched(...)"]
        Posterior["Posterior samples (E x T x ...)"]
    end

    subgraph Evaluation
        EvalCall["Call Bde.evaluate(Xte, ...)"]
        MakePred["_make_predictor(Xte)"]
        Predictor["BdePredictor"]
        Outputs["Predictions (mean, std, intervals, probs, raw)"]
    end

    FitCall --> Validate --> Build --> Builder
    Builder --> Train --> LogPost --> WarmSampler --> Keys --> Draw --> Cache
    WarmSampler --> Warm --> Adapter --> Adapt --> Results
    Draw --> Wrapper --> Batch --> Posterior
    Cache --> EvalCall --> MakePred --> Predictor --> Outputs
    Posterior --> Predictor
```
