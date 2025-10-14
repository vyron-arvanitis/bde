import unittest
import sys
import os
import time
import jax
import jax.numpy as jnp
from sklearn.exceptions import NotFittedError

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from bde.bde import BdeRegressor, BdeClassifier, Bde
from bde.bde_builder import BdeBuilder

from bde.task import TaskType
from bde.models import Fnn


class TestBdeRegressor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._start_time = time.perf_counter()

    @classmethod
    def tearDownClass(cls):
        elapsed = time.perf_counter() - cls._start_time
        print(f"\n[Benchmark] {cls.__name__} ran in {elapsed:.4f} seconds.")

    def setUp(self):
        x = jnp.arange(20.0).reshape(-1, 2)
        y = x[..., -1]
        self.reg = BdeRegressor(hidden_layers=[2], epochs=2, n_members=1)
        self.x, self.y = x, y
        self.fit = self.reg.fit(self.x, self.y)
        self.y_pred = self.reg.predict(self.x)

    def test_predict_before_fit_raise_NotFittedError(self):
        self.reg_1 = BdeRegressor(hidden_layers=[2], epochs=2, n_members=1)
        self.assertRaises(NotFittedError, self.reg_1.predict, self.x)

    def test_fit_with_zero_members_raises_value_error(self):
        """This tests the _ensure_member_initialization"""
        self.reg_1 = BdeRegressor(hidden_layers=[2], epochs=2, n_members=0)
        with self.assertRaisesRegex(ValueError, "n_members must be at leat 1 to build the ensemble!"):
            self.reg_1.fit(self.x, self.y)

    def test_predict_returns_mean_and_optional_outputs(self):
        # Default call returns only the predictive mean
        mean_only = self.reg.predict(self.x)
        self.assertEqual(mean_only.shape, (self.x.shape[0],))

        # Requesting mean and std returns a tuple of two arrays
        mean_with_std, std = self.reg.predict(self.x, mean_and_std=True)
        self.assertTrue(jnp.allclose(mean_only, mean_with_std))
        self.assertEqual(std.shape, mean_only.shape)

        # Requesting credible intervals returns the mean and quantiles per level
        ci_levels = [0.1, 0.9]
        mean_with_ci, ci = self.reg.predict(self.x, credible_intervals=ci_levels)
        self.assertTrue(jnp.allclose(mean_only, mean_with_ci))
        self.assertEqual(ci.shape, (len(ci_levels), mean_only.shape[0]))

        raw_preds = self.reg.predict(self.x, raw=True)
        self.assertEqual(raw_preds.shape[-1], 2)
        self.assertEqual(raw_preds.shape[-2], self.x.shape[0])

    def test_evaluate_raw_returns_expected_shape(self):
        raw_out = self.reg._evaluate(self.x, raw=True)
        self.assertIn("raw", raw_out)
        raw = raw_out["raw"]
        # Expect shape (E, T, N, 2) with E=ensemble members, T=samples
        self.assertEqual(raw.shape[-1], 2)
        self.assertEqual(raw.shape[-2], self.x.shape[0])

    def test_validate_evaluate_tags_regression(self):
        with self.assertRaisesRegex(
                ValueError,
                "'probabilities' predictions are only supported"):
            self.reg._validate_evaluate_tags(probabilities=True)

        with self.assertRaisesRegex(
                ValueError,
                "'mean_and_std' and 'credible_intervals' cannot be requested together"):
            self.reg._validate_evaluate_tags(mean_and_std=True, credible_intervals=[0.1, 0.9])

        with self.assertRaisesRegex(
                ValueError,
                "'raw' and 'credible_intervals' cannot be requested together"):
            self.reg._validate_evaluate_tags(raw=True, credible_intervals=[0.1, 0.9])

        with self.assertRaisesRegex(
                ValueError,
                "'raw' and 'mean_and_std' cannot be requested together"):
            self.reg._validate_evaluate_tags(raw=True, mean_and_std=True)


class TestBdeClassifier(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._start_time = time.perf_counter()

    @classmethod
    def tearDownClass(cls):
        elapsed = time.perf_counter() - cls._start_time
        print(f"\n[Benchmark] {cls.__name__} ran in {elapsed:.4f} seconds.")

    def setUp(self):
        x = jnp.arange(20.0).reshape(-1, 2)
        y = (x[..., -1] > 5).astype(int)
        self.clf = BdeClassifier(hidden_layers=[2], epochs=2, n_members=1)
        self.x, self.y = x, y
        self.fit = self.clf.fit(self.x, self.y)
        self.y_pred = self.clf.predict(self.x)
        self.probs = self.clf.predict_proba(self.x)

    def test_predict_before_fit_raise_NotFittedError(self):
        self.clf_1 = BdeClassifier(hidden_layers=[2], epochs=2, n_members=1)
        self.assertRaises(NotFittedError, self.clf_1.predict, self.x)

    def test_fit_with_zero_members_raises_value_error(self):
        """This tests the _ensure_member_initialization"""
        self.clf_1 = BdeClassifier(hidden_layers=[2], epochs=2, n_members=0)
        with self.assertRaisesRegex(ValueError, "n_members must be at leat 1 to build the ensemble!"):
            self.clf_1.fit(self.x, self.y)


    def test_evaluate_raw_returns_expected_shape(self):
        raw_out = self.clf._evaluate(self.x, raw=True)
        self.assertIn("raw", raw_out)
        raw = raw_out["raw"]
        # Expect shape (E, T, N, 2) with E=ensemble members, T=samples
        self.assertEqual(raw.shape[-1], 2)
        self.assertEqual(raw.shape[-2], self.x.shape[0])

    def test_fit_and_predict(self):
        self.assertEqual(self.y_pred.shape[0], self.x.shape[0])

    def test_predict_proba(self):
        self.assertAlmostEqual(self.probs.shape[1], 2)  # 2 classes

    def test_validate_evaluate_tags_classification(self):
        with self.assertRaisesRegex(ValueError, "'mean_and_std' predictions are not available"):
            self.clf._validate_evaluate_tags(mean_and_std=True)

        with self.assertRaisesRegex(ValueError, "'credible_intervals' predictions are not available"):
            self.clf._validate_evaluate_tags(credible_intervals=[0.1, 0.9])


class TestBdeBuilderHelpers(unittest.TestCase):
    def _make_builder(self, **overrides):
        defaults = dict(
            hidden_sizes=[3],
            n_members=2,
            task=TaskType.REGRESSION,
            seed=0,
            act_fn="relu",
            patience=1,
        )
        defaults.update(overrides)
        return BdeBuilder(**defaults)

    def test_build_full_sizes_regression(self):
        builder = self._make_builder(task=TaskType.REGRESSION, hidden_sizes=[3, 2])
        x = jnp.ones((5, 4))
        y = jnp.arange(5.0)
        full_sizes = builder._build_full_sizes(x, y)
        self.assertEqual(full_sizes, [4, 3, 2, 2])

    def test_build_full_sizes_classification(self):
        builder = self._make_builder(task=TaskType.CLASSIFICATION, hidden_sizes=[5])
        x = jnp.ones((6, 3))
        y = jnp.array([0, 1, 2, 1, 0, 2])
        full_sizes = builder._build_full_sizes(x, y)
        self.assertEqual(full_sizes, [3, 5, 3])

    def test_determine_output_dim_invalid_task(self):
        builder = self._make_builder()
        builder.task = "unsupported"
        y = jnp.zeros((4,))
        with self.assertRaisesRegex(ValueError, "Unknown task"):
            builder._determine_output_dim(y)

    def test_ensure_member_initialization_creates_members_once(self):
        builder = self._make_builder(n_members=2)
        sizes = [4, 3, 2]
        builder._ensure_member_initialization(sizes)
        self.assertEqual(len(builder.members), builder.n_members)
        for member in builder.members:
            self.assertIsInstance(member, Fnn)
            self.assertListEqual(member.sizes, sizes)

        first_members = builder.members
        builder._ensure_member_initialization([5, 5])
        self.assertIs(builder.members, first_members)

    def test_create_training_components_uses_defaults(self):
        builder = self._make_builder()
        sizes = [3, 3, 2]
        builder._ensure_member_initialization(sizes)

        components = builder._create_training_components(optimizer=None, loss=None)

        self.assertTrue(callable(components.loss_fn))
        self.assertTrue(callable(components.step_fn))
        self.assertTrue(hasattr(components.optimizer, "init"))
        self.assertEqual(components.loss_obj.name, "gaussian_nll")

        member = builder.members[0]
        params = member.params
        opt_state = components.optimizer.init(params)
        xb = jnp.ones((2, sizes[0]))
        yb = jnp.ones((2, 1))
        new_params, new_state, loss = components.step_fn(params, opt_state, xb, yb)

        self.assertEqual(loss.shape, ())
        self.assertEqual(
            jax.tree_util.tree_structure(new_params),
            jax.tree_util.tree_structure(params),
        )
        self.assertEqual(
            jax.tree_util.tree_structure(new_state),
            jax.tree_util.tree_structure(opt_state),
        )

    def test_keys_and_cached_attribute_access(self):
        builder = self._make_builder()

        with self.assertRaisesRegex(ValueError, "No results saved"):
            builder.keys()

        builder.results["posterior"] = {"mean": 0.0}
        self.assertEqual(builder.keys(), ["posterior"])
        self.assertEqual(builder.posterior, {"mean": 0.0})

        with self.assertRaises(AttributeError):
            _ = builder.not_present
