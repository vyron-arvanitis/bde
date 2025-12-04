"""Utilities for constructing and training Bayesian deep ensembles."""

import logging
from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.tree_util import tree_map
from jax.typing import ArrayLike

from bde.sampler.types import ParamTree
from bde.sampler.utils import pad_axis0
from bde.task import TaskType

from .loss import BaseLoss
from .models import Fnn
from .training.callbacks import EarlyStoppingCallback, NullCallback
from .training.trainer import FnnTrainer

logger = logging.getLogger(__name__)


@dataclass
class TrainingComponents:
    optimizer: optax.GradientTransformation
    loss_obj: BaseLoss
    loss_fn: Callable[[ParamTree, jnp.ndarray, jnp.ndarray], jnp.ndarray]
    step_fn: Callable[
        [ParamTree, optax.OptState, jnp.ndarray, jnp.ndarray],
        tuple[ParamTree, optax.OptState, jnp.ndarray],
    ]


@dataclass
class DistributedTrainingState:
    params_de: ParamTree
    opt_state_de: ParamTree
    pstep: Callable[
        [ParamTree, ParamTree, jnp.ndarray, jnp.ndarray, jnp.ndarray],
        tuple[ParamTree, ParamTree, jnp.ndarray],
    ]
    peval: Callable[[ParamTree, jnp.ndarray, jnp.ndarray], jnp.ndarray]
    ensemble_size: int


@dataclass
class TrainingLoopResult:
    params_de: ParamTree
    opt_state_de: ParamTree
    callback_state: Any


class BdeBuilder(FnnTrainer):
    """Helper that instantiates ensemble members and coordinates their training.

    The builder keeps lightweight references to the underlying `Fnn` instances and
    exposes utilities used by the high-level estimator.
    """

    def __init__(
        self,
        hidden_sizes: list,
        n_members: int,
        task: TaskType,
        seed: int,
        act_fn: str,
        patience: int,
        val_split: float,
        lr: float,
        weight_decay: float,
    ):
        """Configure the builder with architectural and training defaults.

        Parameters
        ----------
        hidden_sizes : list[int]
            Width of each hidden layer shared across ensemble members.
        n_members : int
            Number of FNN members in the ensemble.
        task : TaskType
            Task specification (`REGRESSION` or `CLASSIFICATION`).
        seed : int
            Base random seed used for member initialization.
        act_fn : str
            Activation function name understood by `Fnn`.
        patience : int
            Early stopping patience measured in validation evaluations.
        val_split: float | None, default=0.15
            Proportion to split the data for validation when early stopping is enabled.
            Must lie in (0, 1); when ``None``, all data is used for training and early
            stopping is skipped.
        lr : float
            Learning rate of the optimizer used for member pretraining.
        weight_decay: float | None
            Weight decay parameter for the AdamW optimizer applied to member training.
        """

        FnnTrainer.__init__(self)
        self.hidden_sizes = hidden_sizes
        self.n_members = n_members
        self.seed = seed
        self.task = task
        self.params_e = None  # will be set after training
        self.members = None  #
        self.act_fn = act_fn
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience
        self.val_split = val_split

        self.eval_every = 1  # Check epochs for early stopping
        self.keep_best = True
        self.min_delta = 1e-9
        self.history = {"epoch": [], "train_loss": [], "val_loss": []}
        self.results: dict[str, Any] = {}

    @staticmethod
    def get_model(seed: int, *, act_fn: str, sizes: list[int]) -> Fnn:
        """Instantiate a single fully-connected network.

        Parameters
        ----------
        seed : int
            PRNG seed used for weight initialization.
        act_fn : str
            Activation function name recognised by `Fnn`.
        sizes : list[int]
            Layer widths including input and output dimensions.

        Returns
        -------
        Fnn
            Newly created network with initialized parameters.
        """

        return Fnn(sizes=sizes, init_seed=seed, act_fn=act_fn)

    def _deep_ensemble_creator(
        self, seed: int = 0, *, act_fn, sizes: list[int] = None
    ) -> list[Fnn]:
        """Create an ensemble of ``n_members`` FNN models.

        Each member is initialized with a different random seed to encourage
        diversity within the ensemble. The initialized models are stored in the
        ``members`` attribute and returned.

        Returns
        -------
        list[Fnn]
            List of initialized FNN models comprising the ensemble.
        """

        return [
            self.get_model(seed + i, act_fn=act_fn, sizes=sizes)
            for i in range(self.n_members)
        ]

    def _determine_output_dim(self, y: ArrayLike) -> int:
        """Infer the output dimension for the ensemble from the targets.

        Parameters
        ----------
        y : ArrayLike
            Target values used to determine the output width.

        Returns
        -------
        int
            Number of output units required for the configured task.
        """
        if self.task == TaskType.REGRESSION:
            return 2
        elif self.task == TaskType.CLASSIFICATION:
            return int(y.max()) + 1
        else:
            raise ValueError(f"Unknown task {self.task} !")

    def _build_full_sizes(self, x: ArrayLike, y: ArrayLike) -> list[int]:
        """Compute the layer sizes for each fully-connected ensemble member.

        Parameters
        ----------
        x : ArrayLike
            Feature matrix of shape (n_samples, n_features). Only the feature dimension
            is used here.
        y : ArrayLike
            Target values determining the output dimension via `_determine_output_dim`.

        Returns
        -------
        list[int]
            Full list of layer sizes `[n_features, *hidden_sizes, n_outputs]`.
        """
        n_features = x.shape[1]
        return [n_features] + self.hidden_sizes + [self._determine_output_dim(y)]

    def _ensure_member_initialization(self, full_sizes: list):
        """Lazily create ensemble members when first needed.

        Parameters
        ----------
        full_sizes : list[int]
            Layer specification passed to each `Fnn` member.
        """

        if self.members is None:
            if self.n_members < 1:
                raise ValueError("n_members must be at leat 1 to build the ensemble!")
            self.members = self._deep_ensemble_creator(
                seed=self.seed, act_fn=self.act_fn, sizes=full_sizes
            )

    def _create_training_components(self, optimizer, loss: BaseLoss):
        """Assemble optimizer, loss, and step function used during training.

        Parameters
        ----------
        optimizer : optax.GradientTransformation | None
            Optional externally provided optimizer.
        loss : BaseLoss | None
            Optional custom loss instance.

        Returns
        -------
        TrainingComponents
            Pack containing optimizer, loss object, scalar loss_fn, and step function.
        """

        proto_model = self.members[0]
        opt = optimizer or self.default_optimizer(self.lr, self.weight_decay)
        loss_obj = loss or self.default_loss(self.task)
        loss_fn = FnnTrainer.make_loss_fn(proto_model, loss_obj)
        step_one = FnnTrainer.make_step(loss_fn, opt)
        return TrainingComponents(opt, loss_obj, loss_fn, step_one)

    def _create_callback(self) -> EarlyStoppingCallback | NullCallback:
        if self.patience is None:
            return NullCallback()
        else:
            return EarlyStoppingCallback(
                patience=self.patience,
                min_delta=self.min_delta,
                eval_every=self.eval_every,
            )

    def _reset_history(self):
        """Reset the training history dictionary."""
        self.history = {"epoch": [], "train_loss": [], "val_loss": []}

    def _prepare_distributed_state(
        self, components: TrainingComponents
    ) -> DistributedTrainingState:
        """Pack ensemble parameters and optimizer state for pmap execution.

        Parameters
        ----------
        components : TrainingComponents
            Prepared optimizer, loss, and step functions.

        Returns
        -------
        DistributedTrainingState
            Parameters and optimizer states reshaped for device parallelism along
            with pmap'ed step and eval functions.
        """

        params_e = tree_map(
            lambda *ps: jnp.stack(ps, axis=0), *[m.params for m in self.members]
        )
        ensemble_size = len(self.members)
        device_count = jax.local_device_count()
        logger.info("Kernel devices: %s", device_count)
        pad = (device_count - (ensemble_size % max(device_count, 1))) % max(
            device_count, 1
        )
        ensemble_padded = ensemble_size + pad
        members_per_device = ensemble_padded // max(device_count, 1)
        params_e = tree_map(lambda a: pad_axis0(a, pad), params_e)
        params_de = tree_map(
            lambda a: a.reshape(device_count, members_per_device, *a.shape[1:]),
            params_e,
        )

        def init_chunk(params_chunk: Any) -> jax.vmap:
            return jax.vmap(components.optimizer.init)(params_chunk)

        opt_state_de = jax.pmap(init_chunk, in_axes=0, out_axes=0)(params_de)

        def step_chunk(params_chunk: Any, opt_state_chunk, x_b, y_b, stopped_chunk):
            def step_member(p, s):
                return components.step_fn(p, s, x_b, y_b)

            new_params, new_states, losses = jax.vmap(step_member)(
                params_chunk, opt_state_chunk
            )

            def freeze(new, old):
                expand = (None,) * (new.ndim - stopped_chunk.ndim)
                mask = stopped_chunk[(...,) + expand]
                return jnp.where(mask, old, new)

            new_params = jax.tree_util.tree_map(freeze, new_params, params_chunk)
            new_states = jax.tree_util.tree_map(freeze, new_states, opt_state_chunk)
            return new_params, new_states, losses

        def eval_chunk(params_chunk, x_b, y_b):
            def loss_member(p):
                return components.loss_fn(p, x_b, y_b)

            return jax.vmap(loss_member)(params_chunk)

        pstep = jax.pmap(step_chunk, in_axes=(0, 0, None, None, 0), out_axes=(0, 0, 0))
        peval = jax.pmap(eval_chunk, in_axes=(0, None, None), out_axes=0)

        return DistributedTrainingState(
            params_de, opt_state_de, pstep, peval, ensemble_size
        )

    def _training_loop(
        self,
        state: DistributedTrainingState,
        callback: EarlyStoppingCallback,
        callback_state,
        x_train: ArrayLike,
        y_train: ArrayLike,
        x_val: ArrayLike,
        y_val: ArrayLike,
        epochs: int,
    ) -> TrainingLoopResult:
        """Run the distributed training loop with optional validation.

        Parameters
        ----------
        state : DistributedTrainingState
            Packed parameters, optimizer state, and pmap'ed step/eval functions.
        callback : EarlyStoppingCallback
            Early stopping controller that decides when members should freeze.
        callback_state : Any
            Mutable state tracked by the callback.
        x_train : ArrayLike
            Training features broadcast to every ensemble member.
        y_train : ArrayLike
            Training targets aligned with `x_train`.
        x_val : ArrayLike
            Validation features used for early stopping. May be `None`.
        y_val : ArrayLike
            Validation targets used by the callback. May be `None`.
        epochs : int
            Maximum number of epochs to run.

        Returns
        -------
        TrainingLoopResult
            Final distributed parameters, optimizer state, and callback state.
        """
        params_de = state.params_de
        opt_state_de = state.opt_state_de

        train_loss_e = jnp.full((epochs, state.ensemble_size), jnp.nan)
        val_loss_e = jnp.full((epochs, state.ensemble_size), jnp.nan)

        for epoch in range(epochs):
            stopped_de = callback.stopped_mask(callback_state)
            params_de, opt_state_de, lvals_de = state.pstep(
                params_de, opt_state_de, x_train, y_train, stopped_de
            )

            train_lvals_e = lvals_de.reshape(-1)[: state.ensemble_size]
            self.history["epoch"].append(epoch)
            train_loss_e = train_loss_e.at[epoch].set(train_lvals_e)

            should_eval = (
                (x_val is not None)
                and (y_val is not None)
                and callback.should_evaluate(epoch)
            )
            if should_eval:
                val_lvals_de = state.peval(params_de, x_val, y_val)
                val_lvals_e = val_lvals_de.reshape(-1)[: state.ensemble_size]
                val_loss_e = val_loss_e.at[epoch].set(val_lvals_e)
                callback_state = callback.update(
                    callback_state, epoch, params_de, val_lvals_de
                )
                if epoch % 100 == 0:
                    logger.info(
                        "Epoch %d: %d ensemble members still training",
                        epoch,
                        callback.active_members(callback_state),
                    )

                if callback.all_stopped(callback_state):
                    logger.info("All members stopped by epoch %d.", epoch)
                    break

        epochs_run = len(self.history["epoch"])

        train_hist = train_loss_e[:epochs_run]
        val_hist = val_loss_e[:epochs_run]

        stop_epochs = np.asarray(
            callback.stop_epoch_de(callback_state, ensemble_size=state.ensemble_size)
        )

        self.history = {
            f"Model{i}": {
                "epoch": jnp.arange(epochs_run),
                "trainloss": train_hist[:, i],
                "valloss": val_hist[:, i],
                "stop_epoch": int(stop_epochs[i]),
            }
            for i in range(state.ensemble_size)
        }

        for _, model_hist in self.history.items():
            s = model_hist["stop_epoch"]
            if s < 0:
                continue
            model_hist["trainloss"] = model_hist["trainloss"][: s + 1]
            model_hist["valloss"] = model_hist["valloss"][: s + 1]
            model_hist["epoch"] = model_hist["epoch"][: s + 1]

        return TrainingLoopResult(params_de, opt_state_de, callback_state)

    def fit_members(
        self,
        x: ArrayLike,
        y: ArrayLike,
        epochs: int,
        optimizer=None,
        loss: BaseLoss = None,
    ):
        """Train every ensemble member on the provided dataset.

        Parameters
        ----------
        x : ArrayLike
            Feature matrix of shape (n_samples, n_features).
        y : ArrayLike
            Target array broadcastable to the model output.
        epochs : int
            Maximum number of training epochs.
        optimizer : optax.GradientTransformation | None
            Custom optimizer; defaults to `default_optimizer` when `None`.
        loss : BaseLoss | None
            Loss object; defaults to `default_loss` for the configured task when `None`.

        Returns
        -------
        list[Fnn]
            The trained ensemble members with updated parameters.
        """

        full_sizes = self._build_full_sizes(x, y)
        self._ensure_member_initialization(full_sizes)

        if self.patience is None:
            x_train, y_train = x, y
            x_val = None
            y_val = None
        else:
            x_train, x_val, y_train, y_val = super().split_train_val(
                x,
                y,
                val_size=self.val_split,
            )  # for early stopping

        components = self._create_training_components(optimizer, loss)
        state = self._prepare_distributed_state(components)
        self._reset_history()
        callback = self._create_callback()
        callback_state = callback.initialize(state.params_de)

        loop_result = self._training_loop(
            state,
            callback,
            callback_state,
            x_train,
            y_train,
            x_val,
            y_val,
            epochs,
        )
        callback.stop_epoch_de(
            loop_result.callback_state, ensemble_size=state.ensemble_size
        )

        params_e_final = callback.best_params(
            loop_result.callback_state, ensemble_size=state.ensemble_size
        )
        for i, m in enumerate(self.members):
            m.params = tree_map(lambda a: a[i], params_e_final)
        self.params_e = params_e_final
        return self.members

    def keys(self):
        """Return identifiers of cached results."""

        if not self.results:
            raise ValueError(
                "No results saved. Cache outputs via predict_ensemble(..., cache=True)"
                " first."
            )
        return list(self.results.keys())

    def __getattr__(self, item):
        """Fallback to cached results when available."""

        results = self.__dict__.get("results")
        if results is not None and item in results:
            return results[item]
        raise AttributeError(f"{self.__class__.__name__} has no attribute '{item}' !")
