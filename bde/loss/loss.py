from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp
import optax
from jax.typing import ArrayLike


class BaseLoss(ABC):
    name: str

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def __call__(self, preds, y_true):
        ...


class Rmse(BaseLoss):
    @property
    def name(self) -> str:
        return "rmse"

    def __call__(self, preds: ArrayLike, y_true: ArrayLike):
        mu = preds[..., 0:1]
        rmse = (mu - y_true) ** 2
        return jnp.mean(rmse)


class GaussianNLL(BaseLoss):
    def __init__(self, min_sigma=1e-6, map_fn=jax.nn.softplus):
        self.min_sigma = min_sigma
        self.map_fn = map_fn

    @property
    def name(self) -> str:
        return "gaussian_nll"

    def __call__(self, preds: ArrayLike, y_true: ArrayLike):
        mu = preds[..., 0:1]
        sigma = self.map_fn(preds[..., 1:2]) + self.min_sigma
        resid = (y_true - mu) / sigma
        nll = 0.5 * (jnp.log(2 * jnp.pi) + 2 * jnp.log(sigma) + resid**2)
        return jnp.mean(nll)


class BinaryCrossEntropy(BaseLoss):
    def __init__(self, min_sigma=1e-6, map_fn=jax.nn.softplus):
        self.min_sigma = min_sigma
        self.map_fn = map_fn

    @property
    def name(self) -> str:
        return "binary_cross_entropy"

    def __call__(self, preds, y_true):
        cross_entropy = optax.sigmoid_binary_cross_entropy(logits=preds, labels=y_true)
        return jnp.mean(cross_entropy)


class CategoricalCrossEntropy(BaseLoss):
    def __init__(self, min_sigma=1e-6, map_fn=jax.nn.softplus):
        self.min_sigma = min_sigma
        self.map_fn = map_fn

    @property
    def name(self) -> str:
        return "categorical_cross_entropy"

    def __call__(self, preds: ArrayLike, y_true: ArrayLike):
        cross_entropy = optax.softmax_cross_entropy(
            logits=preds, labels=jax.nn.one_hot(y_true, preds.shape[-1])
        )
        return jnp.mean(cross_entropy)
