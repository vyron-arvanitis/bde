from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
from jax import tree_util


class NullCallback:
    @staticmethod
    def initialize(params_de):
        return params_de  # or a simple container

    @staticmethod
    def should_evaluate(epoch):
        return False  # skip validation entirely, or True if you still want metrics

    @staticmethod
    def stopped_mask(state):
        first_leaf = tree_util.tree_leaves(state)[0]
        leading_shape = first_leaf.shape[:2]
        return jnp.zeros(leading_shape, dtype=bool)

    @staticmethod
    def update(state, epoch, params_de, val_lvals_de):
        return state

    @staticmethod
    def all_stopped(state):
        return False

    @staticmethod
    def best_params(state, *, ensemble_size):
        return tree_util.tree_map(
            lambda a: a.reshape(-1, *a.shape[2:])[:ensemble_size],
            state,
        )

    @staticmethod
    def stop_epoch_de(state, *, ensemble_size):
        return -jnp.ones((ensemble_size,), dtype=jnp.int32)


@dataclass
class EarlyStoppingState:
    """This class will act as a container for storing values
    of per-member early stopping
    """

    best_params_de: any
    best_metric_de: jnp.ndarray
    epochs_no_improve_de: jnp.ndarray
    stopped_de: jnp.ndarray
    stop_epoch_de: jnp.ndarray


class EarlyStoppingCallback:
    """This class implements the early stopping"""

    def __init__(self, *, patience: int, min_delta: float, eval_every: int = 1):
        self.patience = patience
        self.min_delta = min_delta
        self.eval_every = eval_every
        self._num_devices: int | None = None
        self._members_per_device: int | None = None

    def initialize(self, params_de: Any) -> EarlyStoppingState:
        """Initialise the callback state from the distributed parameter tree."""

        leaf_shape = tree_util.tree_leaves(params_de)[0].shape
        if len(leaf_shape) < 2:
            raise ValueError(
                "Expected distributed params with leading (device, member) axes."
            )

        self._num_devices, self._members_per_device = leaf_shape[:2]

        best_metric = jnp.full((self._num_devices, self._members_per_device), jnp.inf)
        epochs_no_improve = jnp.zeros_like(best_metric, dtype=jnp.int32)
        stopped = jnp.zeros_like(best_metric, dtype=bool)
        stop_epoch = -jnp.ones_like(best_metric, dtype=jnp.int32)

        return EarlyStoppingState(
            best_params_de=params_de,
            best_metric_de=best_metric,
            epochs_no_improve_de=epochs_no_improve,
            stopped_de=stopped,
            stop_epoch_de=stop_epoch,
        )

    def update(
        self,
        state: EarlyStoppingState,
        epoch: int,
        params_de: Any,
        val_lvals_de: jnp.ndarray,
    ) -> EarlyStoppingState:
        """Update the callback state with validation losses for the current epoch."""

        improved = val_lvals_de < (state.best_metric_de - self.min_delta)
        best_metric = jnp.where(improved, val_lvals_de, state.best_metric_de)
        epochs_no_improve = jnp.where(improved, 0, state.epochs_no_improve_de + 1)

        def select_best(new_leaf, old_leaf):
            expand = (None,) * (new_leaf.ndim - improved.ndim)
            mask = improved[(...,) + expand]
            return jnp.where(mask, new_leaf, old_leaf)

        best_params = tree_util.tree_map(select_best, params_de, state.best_params_de)

        newly_stopped = (epochs_no_improve >= self.patience) & (~state.stopped_de)
        stopped = state.stopped_de | newly_stopped
        stop_epoch = jnp.where(newly_stopped, jnp.int32(epoch), state.stop_epoch_de)

        return EarlyStoppingState(
            best_params_de=best_params,
            best_metric_de=best_metric,
            epochs_no_improve_de=epochs_no_improve,
            stopped_de=stopped,
            stop_epoch_de=stop_epoch,
        )

    def should_evaluate(self, epoch: int) -> bool:
        """Return whether evaluation should run on this epoch."""
        return (epoch % self.eval_every) == 0

    def stop_epoch_de(self, state: EarlyStoppingState, *, ensemble_size: int):
        """Return the epoch at which each real member stopped."""
        return state.stop_epoch_de.reshape(-1)[:ensemble_size]

    @staticmethod
    def all_stopped(state: EarlyStoppingState) -> bool:
        """Return True if all members have triggered early stopping."""

        return bool(jnp.all(state.stopped_de))

    @staticmethod
    def active_members(state: EarlyStoppingState) -> int:
        """Number of members that are still training."""

        return int(jnp.sum(~state.stopped_de))

    @staticmethod
    def stopped_mask(state: EarlyStoppingState) -> jnp.ndarray:
        """Return a mask over (device, member) axes for stopped members."""

        return state.stopped_de

    def best_params(self, state: EarlyStoppingState, *, ensemble_size: int) -> Any:
        """Return the best parameters for the real ensemble members."""

        if self._num_devices is None or self._members_per_device is None:
            raise RuntimeError("Callback state has not been initialised.")

        total_members = self._num_devices * self._members_per_device
        return tree_util.tree_map(
            lambda a: a.reshape(total_members, *a.shape[2:])[:ensemble_size],
            state.best_params_de,
        )

    def stop_epochs(
        self, state: EarlyStoppingState, *, ensemble_size: int
    ) -> jnp.ndarray:
        """Return the epoch at which each real member stopped."""

        if self._num_devices is None or self._members_per_device is None:
            raise RuntimeError("Callback state has not been initialised.")

        total_members = self._num_devices * self._members_per_device
        flat_epochs = state.stop_epoch_de.reshape(total_members)
        return flat_epochs[:ensemble_size]
