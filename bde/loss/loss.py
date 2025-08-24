from abc import ABC, abstractmethod
import optax
import jax.numpy as jnp
from jax.typing import ArrayLike
from jax import Array
from typing import Optional


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
        pass

    @staticmethod
    def _coerce(y_true: ArrayLike, y_pred: ArrayLike) -> tuple[Array, Array]:
        """
        Convert inputs to JAX arrays with a consistent floating dtype.

        - `y_pred` is converted to a JAX array with dtype float32.
        - `y_true` is converted to a JAX array and cast to the same dtype as `y_pred`
          (i.e., float32). This avoids integer math in the loss and prevents dtype
          promotion / needless recompilation.

        Parameters
        ----------
        y_true : ArrayLike
            Ground-truth targets. Shape (N, ...) or (N,).
        y_pred : ArrayLike
            Model outputs (logits, probabilities, or predictions). Shape compatible
            with `y_true`.

        Returns
        -------
        yt : Array
            `y_true` as a JAX array with dtype float32 (matching `y_pred`).
        yp : Array
            `y_pred` as a JAX array with dtype float32.
        """
        yp = jnp.asarray(y_pred, dtype=jnp.float32)
        yt = jnp.asarray(y_true, dtype=yp.dtype)
        return yt, yp

    @staticmethod
    def _reduce_nonbatch(x: Array) -> Array:
        """
        #TODO: documentation
        Parameters
        ----------
        x

        Returns
        -------

        """
        return x if x.ndim == 1 else x.mean(axis=tuple(range(1, x.ndim)))

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
        #TODO: documentation

        Parameters
        ----------
        y_true
        y_pred
        kwargs

        Returns
        -------

        """

        yt, yp = self._coerce(y_true, y_pred)
        per_elem = optax.losses.squared_error(yp, yt)  # (N, D, ...)
        return self._reduce_nonbatch(per_elem)  # (N,)


class LossNLL(Loss):
    """Negative Log likelihood loss"""

    def __call__(self, y_true: ArrayLike, y_pred: ArrayLike, **kwargs):
        """
        #TODO: documentation
        Parameters
        ----------
        y_true
        y_pred
        kwargs

        Returns
        -------

        """
        # TODO: [@SEE comment on issue]
        # jitter = 1e-8
        # res = 0.5 * jnp.log(10)
        # return res.mean(axis=tuple(range(1, res.ndim))) # (N,)
        pass


class LossBinaryCrossEntropy(Loss):
    def __init__(self, eps: float = 1e-7):
        self.eps = eps

    def __call__(self, y_true: ArrayLike, y_pred: ArrayLike, **kwargs) -> Array:
        """
        #TODO: documentation
        Parameters
        ----------
        y_true
        y_pred
        kwargs

        Returns
        -------

        """
        yt, yp = self._coerce(y_true, y_pred)

        p = jnp.clip(yp, self.eps, 1.0 - self.eps) # ensure no log(0)
        per_elem = -(yt * jnp.log(p) + (1.0 - yt) * jnp.log(1.0 - p))
        return self._reduce_nonbatch(per_elem)
