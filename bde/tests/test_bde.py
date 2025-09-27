import unittest
import sys
import os
import time
import jax.numpy as jnp
from sklearn.exceptions import NotFittedError

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from bde.bde import BdeRegressor, BdeClassifier, Bde


class TestBde(unittest.TestCase):
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

    def test_evaluate_raw_returns_expected_shape(self):
        raw_out = self.reg.evaluate(self.x, raw=True)
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

    def test_fit_and_predict(self):
        self.assertEqual(self.y_pred.shape[0], self.x.shape[0])

    def test_predict_proba(self):
        self.assertAlmostEqual(self.probs.shape[1], 2)  # 2 classes

    def test_validate_evaluate_tags_classification(self):
        with self.assertRaisesRegex(ValueError, "'mean_and_std' predictions are not available"):
            self.clf._validate_evaluate_tags(mean_and_std=True)

        with self.assertRaisesRegex(ValueError, "'credible_intervals' predictions are not available"):
            self.clf._validate_evaluate_tags(credible_intervals=[0.1, 0.9])
