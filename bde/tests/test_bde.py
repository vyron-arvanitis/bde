import os
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from bde.bde import BdeClassifier, BdeRegressor
from bde.bde_builder import BdeBuilder
from bde.models import Fnn
from bde.task import TaskType

# ----------------------------
# Fixtures
# ----------------------------


@pytest.fixture(scope="module", autouse=True)
def benchmark_timer():
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    print(f"\n[Benchmark] test_bde_pytest ran in {elapsed:.4f} seconds.")


@pytest.fixture
def reg_data():
    x = jnp.arange(20.0).reshape(-1, 2)
    y = x[..., -1]
    return x, y


@pytest.fixture
def clf_data():
    x = jnp.arange(20.0).reshape(-1, 2)
    y = (x[..., -1] > 5).astype(int)
    return x, y


@pytest.fixture
def reg_model(reg_data):
    x, y = reg_data
    reg = BdeRegressor(hidden_layers=[2], epochs=2, n_members=1)
    reg.fit(x, y)
    return reg


@pytest.fixture
def clf_model(clf_data):
    x, y = clf_data
    clf = BdeClassifier(hidden_layers=[2], epochs=2, n_members=1)
    clf.fit(x, y)
    return clf


@pytest.fixture
def builder():
    def _make_builder(**overrides):
        defaults = dict(
            hidden_sizes=[3],
            n_members=2,
            task=TaskType.REGRESSION,
            seed=0,
            act_fn="relu",
            patience=1,
            val_split=0.1,
            lr=0.01,
            weight_decay=0.01,
        )
        defaults.update(overrides)
        return BdeBuilder(**defaults)

    return _make_builder


# ----------------------------
# Regressor tests
# ----------------------------


def test_reg_predict_before_fit_raises_notfitted(reg_data):
    x, _ = reg_data
    reg = BdeRegressor(hidden_layers=[2], epochs=2, n_members=1)
    with pytest.raises(NotFittedError):
        reg.predict(x)


def test_reg_fit_zero_members_raises(reg_data):
    x, y = reg_data
    reg = BdeRegressor(hidden_layers=[2], epochs=2, n_members=0)
    with pytest.raises(
        ValueError, match="n_members must be at leat 1 to build the ensemble!"
    ):
        reg.fit(x, y)


def test_reg_predict_outputs(reg_model, reg_data):
    x, _ = reg_data

    mean_only = reg_model.predict(x)
    assert mean_only.shape == (x.shape[0],)

    mean_with_std, std = reg_model.predict(x, mean_and_std=True)
    assert jnp.allclose(mean_only, mean_with_std)
    assert std.shape == mean_only.shape

    ci_levels = [0.1, 0.9]
    mean_with_ci, ci = reg_model.predict(x, credible_intervals=ci_levels)
    assert jnp.allclose(mean_only, mean_with_ci)
    assert ci.shape == (len(ci_levels), mean_only.shape[0])

    raw_preds = reg_model.predict(x, raw=True)
    assert raw_preds.shape[-1] == 2
    assert raw_preds.shape[-2] == x.shape[0]


def test_reg_evaluate_raw_shape(reg_model, reg_data):
    x, _ = reg_data
    raw_out = reg_model._evaluate(x, raw=True)
    assert "raw" in raw_out
    raw = raw_out["raw"]
    assert raw.shape[-1] == 2
    assert raw.shape[-2] == x.shape[0]


def test_reg_validate_evaluate_tags(reg_model):
    with pytest.raises(
        ValueError, match="'probabilities' predictions are only supported"
    ):
        reg_model._validate_evaluate_tags(probabilities=True)

    with pytest.raises(
        ValueError,
        match="'mean_and_std' and 'credible_intervals' cannot be requested together",
    ):
        reg_model._validate_evaluate_tags(
            mean_and_std=True, credible_intervals=[0.1, 0.9]
        )

    with pytest.raises(
        ValueError, match="'raw' and 'credible_intervals' cannot be requested together"
    ):
        reg_model._validate_evaluate_tags(raw=True, credible_intervals=[0.1, 0.9])

    with pytest.raises(
        ValueError, match="'raw' and 'mean_and_std' cannot be requested together"
    ):
        reg_model._validate_evaluate_tags(raw=True, mean_and_std=True)


# ----------------------------
# Classifier tests
# ----------------------------


def test_clf_predict_before_fit_raises_notfitted(clf_data):
    x, _ = clf_data
    clf = BdeClassifier(hidden_layers=[2], epochs=2, n_members=1)
    with pytest.raises(NotFittedError):
        clf.predict(x)


def test_clf_fit_zero_members_raises(clf_data):
    x, y = clf_data
    clf = BdeClassifier(hidden_layers=[2], epochs=2, n_members=0)
    with pytest.raises(
        ValueError, match="n_members must be at leat 1 to build the ensemble!"
    ):
        clf.fit(x, y)


def test_clf_evaluate_raw_shape(clf_model, clf_data):
    x, _ = clf_data
    raw_out = clf_model._evaluate(x, raw=True)
    assert "raw" in raw_out
    raw = raw_out["raw"]
    assert raw.shape[-1] == 2
    assert raw.shape[-2] == x.shape[0]


def test_clf_fit_and_predict(clf_model, clf_data):
    x, _ = clf_data
    y_pred = clf_model.predict(x)
    assert y_pred.shape[0] == x.shape[0]


def test_clf_predict_proba(clf_model, clf_data):
    x, _ = clf_data
    probs = clf_model.predict_proba(x)
    assert probs.shape[1] == 2


def test_clf_validate_evaluate_tags(clf_model):
    with pytest.raises(
        ValueError, match="'mean_and_std' predictions are not available"
    ):
        clf_model._validate_evaluate_tags(mean_and_std=True)

    with pytest.raises(
        ValueError, match="'credible_intervals' predictions are not available"
    ):
        clf_model._validate_evaluate_tags(credible_intervals=[0.1, 0.9])


# ----------------------------
# Builder tests
# ----------------------------


def test_build_full_sizes_regression(builder):
    b = builder(task=TaskType.REGRESSION, hidden_sizes=[3, 2])
    x = jnp.ones((5, 4))
    y = jnp.arange(5.0)
    full_sizes = b._build_full_sizes(x, y)
    assert full_sizes == [4, 3, 2, 2]


def test_build_full_sizes_classification(builder):
    b = builder(task=TaskType.CLASSIFICATION, hidden_sizes=[5])
    x = jnp.ones((6, 3))
    y = jnp.array([0, 1, 2, 1, 0, 2])
    full_sizes = b._build_full_sizes(x, y)
    assert full_sizes == [3, 5, 3]


def test_determine_output_dim_invalid_task(builder):
    b = builder()
    b.task = "unsupported"
    y = jnp.zeros((4,))
    with pytest.raises(ValueError, match="Unknown task"):
        b._determine_output_dim(y)


def test_ensure_member_initialization_creates_once(builder):
    b = builder(n_members=2)
    sizes = [4, 3, 2]

    b._ensure_member_initialization(sizes)

    assert len(b.members) == b.n_members
    for m in b.members:
        assert isinstance(m, Fnn)
        assert m.sizes == sizes

    first_members = b.members
    b._ensure_member_initialization([5, 5])
    assert b.members is first_members


def test_create_training_components_defaults(builder):
    b = builder()
    sizes = [3, 3, 2]
    b._ensure_member_initialization(sizes)

    components = b._create_training_components(optimizer=None, loss=None)

    assert callable(components.loss_fn)
    assert callable(components.step_fn)
    assert hasattr(components.optimizer, "init")
    assert components.loss_obj.name == "gaussian_nll"

    member = b.members[0]
    params = member.params
    opt_state = components.optimizer.init(params)

    xb = jnp.ones((2, sizes[0]))
    yb = jnp.ones((2, 1))
    new_params, new_state, loss = components.step_fn(params, opt_state, xb, yb)

    assert loss.shape == ()
    assert jax.tree_util.tree_structure(new_params) == jax.tree_util.tree_structure(
        params
    )
    assert jax.tree_util.tree_structure(new_state) == jax.tree_util.tree_structure(
        opt_state
    )


def test_keys_and_cached_attribute_access(builder):
    b = builder()

    with pytest.raises(ValueError, match="No results saved"):
        b.keys()

    b.results["posterior"] = {"mean": 0.0}
    assert b.keys() == ["posterior"]
    assert b.posterior == {"mean": 0.0}

    with pytest.raises(AttributeError):
        _ = b.not_present


# ----------------------------
# Sanity tests
# ----------------------------


def test_sanity_airfoil():
    reg = BdeRegressor(
        hidden_layers=[4, 4],
        n_members=3,
        epochs=5,
        warmup_steps=100,
        n_samples=10,
        n_thinning=2,
        patience=2,
    )

    data_path = Path(__file__).parent.parent / "data" / "airfoil.csv"
    data = pd.read_csv(data_path)

    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values.reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_train = x_scaler.fit_transform(X_train)
    X_test = x_scaler.transform(X_test)

    y_train = y_scaler.fit_transform(y_train)
    y_test = y_scaler.transform(y_test)

    # ---- BDE ----
    reg.fit(jnp.array(X_train), jnp.array(y_train))
    bde_mean, _ = reg.predict(jnp.array(X_test), mean_and_std=True)
    bde_rmse = root_mean_squared_error(y_test, bde_mean)

    assert bde_rmse < 1.0

    # ---- Linear baseline ----
    lr = MLPRegressor(hidden_layer_sizes=(2, 12))  # very simple MLP
    lr.fit(X_train, y_train.ravel())
    lr_pred = lr.predict(X_test)
    lr_rmse = root_mean_squared_error(y_test, lr_pred)

    # BDE should not be astronomically worse than linear baseline
    assert bde_rmse < lr_rmse


def test_sanity_iris():
    clf = BdeClassifier(
        hidden_layers=[4, 4],
        n_members=3,
        epochs=5,
        warmup_steps=100,
        n_samples=10,
        n_thinning=2,
        patience=2,
    )

    data_path = Path(__file__).parent.parent / "data" / "iris.csv"
    data = pd.read_csv(data_path)

    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values.ravel()  # keep as integers

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale only X
    x_scaler = StandardScaler()
    X_train = x_scaler.fit_transform(X_train)
    X_test = x_scaler.transform(X_test)

    # ---- BDE ----
    clf.fit(jnp.array(X_train), jnp.array(y_train))
    y_pred = clf.predict(jnp.array(X_test))

    bde_acc = (y_pred == y_test).mean()
    # sanity bound — just make sure it's not broken
    assert bde_acc > 0.6

    # ---- Linear baseline (optional diagnostic) ----

    clf = MLPClassifier(hidden_layer_sizes=(2, 12))
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    lr_acc = (pred == y_test).mean()

    # BDE should not be astronomically worse than logistic  baseline
    assert bde_acc > lr_acc * 0.3
