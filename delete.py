import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

import jax.numpy as jnp
from sklearn.datasets import fetch_openml
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
import bde.BdeRegressor as BdeRegressor
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

# Build the regressor
regressor = BdeRegressor(
    hidden_layers=[16, 16],
    n_members=8,
    seed=0,
    loss=GaussianNLL(),
    epochs=200,
    validation_split=0.15,
    lr=1e-3,
    weight_decay=1e-4,
    warmup_steps=5000,
    n_samples=2000,
    n_thinning=2,
    patience=10,
)

# Fit the regressor
regressor.fit(x=Xtr, y=ytr)

# Get results from regressor
means, sigmas = regressor.predict(Xte, mean_and_std=True)
mean, intervals = regressor.predict(Xte, credible_intervals=[0.1, 0.9])
raw = regressor.predict(
    Xte, raw=True
)  # (ensemble members, n_samples/n_thinning, n_test_data, (mu,sigma))
