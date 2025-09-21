import unittest
import sys
import os
import time
import jax.numpy as jnp

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

    def test__predict(self):
        self.assertEqual(self.y_pred.shape[0], self.x.shape[0])


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
