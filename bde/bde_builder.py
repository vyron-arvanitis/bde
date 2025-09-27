"""this is a bde builder"""
import jax
import jax.numpy as jnp
from jax.tree_util import tree_map

import optax
from typing import Any, Callable
from .models.models import Fnn
from .training.trainer import FnnTrainer
from .training.callbacks import EarlyStoppingCallback

from bde.sampler.my_types import ParamTree
from bde.sampler.utils import _infer_dim_from_position_example, _pad_axis0, _reshape_to_devices
from bde.task import TaskType
from .loss.loss import BaseLoss

from dataclasses import dataclass
from jax.typing import ArrayLike


@dataclass
class TrainingComponents:
    optimizer: optax.GradientTransformation
    loss_obj: BaseLoss
    loss_fn: Callable[[ParamTree, jnp.ndarray, jnp.ndarray], jnp.ndarray]
    step_fn: Callable[
        [ParamTree, optax.OptState, jnp.ndarray, jnp.ndarray], tuple[ParamTree, optax.OptState, jnp.ndarray]]


@dataclass
class DistributedTrainingState:
    params_de: ParamTree
    opt_state_de: ParamTree
    pstep: Callable[
        [ParamTree, ParamTree, jnp.ndarray, jnp.ndarray, jnp.ndarray], tuple[ParamTree, ParamTree, jnp.ndarray]]
    peval: Callable[[ParamTree, jnp.ndarray, jnp.ndarray], jnp.ndarray]
    ensemble_size: int


@dataclass
class TrainingLoopResult:
    params_de: ParamTree
    opt_state_de: ParamTree
    callback_state: Any


class BdeBuilder(FnnTrainer):
    def __init__(self,
                 hidden_sizes: list,
                 n_members: int,
                 task: TaskType,
                 seed: int,
                 act_fn: str,
                 patience: int
                 ):
        FnnTrainer.__init__(self)
        self.hidden_sizes = hidden_sizes
        self.n_members = n_members
        self.seed = seed
        self.task = task
        self.params_e = None  # will be set after training
        self.members = None  #
        self.act_fn = act_fn

        self.patience = patience
        self.eval_every = 1  # Check epochs for early stopping
        self.keep_best = True
        self.min_delta = 1e-9

        self.results = {}

    @staticmethod
    def get_model(seed: int, *, act_fn, sizes) -> Fnn:
        """Create a single Fnn model and initialize its parameters

        Parameters
        ----------
        seed : int
        act_fn:
            #TODO: complete documentation
        sizes:

        Returns
        -------

        """

        return Fnn(sizes=sizes, init_seed=seed, act_fn=act_fn)

    def _deep_ensemble_creator(self, seed: int = 0, *, act_fn, sizes: list[int] = None) -> list[Fnn]:
        """Create an ensemble of ``n_members`` FNN models.

        Each member is initialized with a different random seed to encourage
        diversity within the ensemble. The initialized models are stored in the
        ``members`` attribute and returned.

        Returns
        -------
        list[Fnn]
            List of initialized FNN models comprising the ensemble.
        """

        return [self.get_model(seed + i, act_fn=act_fn, sizes=sizes) for i in range(self.n_members)]

    def _determine_output_dim(self, y: ArrayLike) -> int:
        if self.task == TaskType.REGRESSION:
            return 2
        elif self.task == TaskType.CLASSIFICATION:
            return int(y.max()) + 1
        else:
            raise ValueError(f"Unknown task {self.task} !")

    def _build_full_sizes(self, x: ArrayLike, y: ArrayLike) -> list[int]:
        n_features = x.shape[1]
        return [n_features] + self.hidden_sizes + [self._determine_output_dim(y)]

    def _ensure_member_initialization(self, full_sizes: list):
        if self.members is None:
            if self.n_members < 1:
                raise ValueError("n_members must be at leat 1 to build the ensemble!")
            self.members = self._deep_ensemble_creator(seed=self.seed, act_fn=self.act_fn, sizes=full_sizes)

    def _create_training_components(self, optimizer, loss: BaseLoss):
        proto_model = self.members[0]
        opt = optimizer or self.default_optimizer()
        loss_obj = loss or self.default_loss(self.task)
        loss_fn = FnnTrainer.make_loss_fn(proto_model, loss_obj)
        step_one = FnnTrainer.make_step(loss_fn, opt)
        return TrainingComponents(opt, loss_obj, loss_fn, step_one)

    def _create_callback(self) -> EarlyStoppingCallback:
        return EarlyStoppingCallback(
            patience=self.patience,
            min_delta=self.min_delta,
            eval_every=self.eval_every,
        )

    def _prepare_distributed_state(self, components: TrainingComponents) -> DistributedTrainingState:
        params_e = tree_map(lambda *ps: jnp.stack(ps, axis=0), *[m.params for m in self.members])
        ensemble_size = len(self.members)
        device_count = jax.local_device_count()
        print("Kernel devices:", device_count)
        pad = (device_count - (ensemble_size % max(device_count, 1))) % max(device_count, 1)
        ensemble_padded = ensemble_size + pad
        members_per_device = ensemble_padded // max(device_count, 1)
        params_e = tree_map(lambda a: _pad_axis0(a, pad), params_e)
        params_de = tree_map(lambda a: a.reshape(device_count, members_per_device, *a.shape[1:]), params_e)

        def init_chunk(params_chunk: Any) -> jax.vmap:
            return jax.vmap(components.optimizer.init)(params_chunk)

        opt_state_de = jax.pmap(init_chunk, in_axes=0, out_axes=0)(params_de)

        def step_chunk(params_chunk: Any, opt_state_chunk, x_b, y_b, stopped_chunk):
            def step_member(p, s):
                return components.step_fn(p, s, x_b, y_b)

            new_params, new_states, losses = jax.vmap(step_member)(params_chunk, opt_state_chunk)

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

        return DistributedTrainingState(params_de, opt_state_de, pstep, peval, ensemble_size)

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
        params_de = state.params_de
        opt_state_de = state.opt_state_de

        for epoch in range(epochs):
            stopped_de = callback.stopped_mask(callback_state)
            params_de, opt_state_de, lvals_de = state.pstep(params_de, opt_state_de, x_train, y_train, stopped_de)
            train_mean = float(jnp.mean(jax.device_get(lvals_de)))
            self.history["train_loss"].append(train_mean)
            # if epoch % self.log_every == 0:
            #     print(epoch, train_mean)

            should_eval = (x_val is not None) and (y_val is not None) and callback.should_evaluate(epoch)
            if should_eval:
                val_lvals_de = state.peval(params_de, x_val, y_val)
                callback_state = callback.update(callback_state, epoch, params_de, val_lvals_de)
                # if epoch % 100 == 0:
                #     n_active = callback.active_members(callback_state)
                #     print(f"[epoch {epoch}] active members: {n_active}")

                if callback.all_stopped(callback_state):
                    # print(f"All members stopped by epoch {epoch}.")
                    break
        return TrainingLoopResult(params_de, opt_state_de, callback_state)

    def fit_members(self, x: ArrayLike, y: ArrayLike, epochs: int, optimizer=None, loss: BaseLoss = None):

        full_sizes = self._build_full_sizes(x, y)
        self._ensure_member_initialization(full_sizes)
        x_train, x_val, y_train, y_val = super().split_train_val(x, y)  # for early stopping

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
        stop_epoch_e = callback.stop_epochs(loop_result.callback_state, ensemble_size=state.ensemble_size)
        # for m_id, ep in enumerate(list(map(int, jax.device_get(stop_epoch_e)))):
        #     print(f"member {m_id}: {'stopped at epoch ' + str(ep) if ep >= 0 else 'ran full training'}")

        params_e_final = callback.best_params(loop_result.callback_state, ensemble_size=state.ensemble_size)
        for i, m in enumerate(self.members):
            m.params = tree_map(lambda a: a[i], params_e_final)
        self.params_e = params_e_final
        return self.members

    def keys(self):
        """
        Return the keys currently  in `self.results`.
        """
        if not self.results:
            raise ValueError("No results saved. Call `predict_ensemble(..., cache=True)` first!")
        return list(self.results.keys())

    def __getattr__(self, item):
        """Fallback to cached results when available."""

        results = self.__dict__.get("results")
        if results is not None and item in results:
            return results[item]
        raise AttributeError(f"{self.__class__.__name__} has no attribute '{item}' !")
