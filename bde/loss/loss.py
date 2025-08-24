from abc import ABC, abstractmethod
import optax
import jax.numpy as jnp
from jax.typing import ArrayLike
from jax import Array


class Loss(ABC):
    """Abstract API for losses."""

    @abstractmethod
    def __call__(self, y_true: ArrayLike, y_pred: ArrayLike, **kwargs) -> Array:
        """

        Parameters
        ----------
        y_true
        y_pred
        kwargs

        Returns
        -------

        """
        ...

    def mean_over_batch(self, y_true: ArrayLike, y_pred: ArrayLike, **kwargs) -> Array:
        """

        Parameters
        ----------
        y_true
        y_pred
        kwargs

        Returns
        -------

        """
        per_example = self(y_true=y_true, y_pred=y_pred, **kwargs)  # (N,)
        return jnp.mean(per_example)


class LossMSE(Loss):
    """Mean Squared Error."""

    def __call__(self, y_true: ArrayLike, y_pred: ArrayLike, **kwargs) -> Array:
        """

        Parameters
        ----------
        y_true
        y_pred
        kwargs

        Returns
        -------

        """

        res = optax.losses.squared_error(y_pred, y_true) # (N, D, ...)
        return res.mean(axis=tuple(range(1, res.ndim))) # (N,)
