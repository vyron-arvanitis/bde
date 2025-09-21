"""this is a bde builder"""
import jax, jaxlib
import jax.numpy as jnp
from jax.tree_util import tree_map, tree_structure

import optax

from .models.models import Fnn
from .training.trainer import FnnTrainer
from .training.callbacks import EarlyStoppingCallback

from bde.sampler.my_types import ParamTree
from bde.sampler.utils import _infer_dim_from_position_example, _pad_axis0, _reshape_to_devices
from bde.task import TaskType


class BdeBuilder(FnnTrainer):
    def __init__(self,
                 hidden_sizes,
                 n_members,
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

    def get_model(self, seed: int, *, act_fn, sizes) -> Fnn:
        """Create a single Fnn model and initialize its parameters

        Parameters
        ----------
        seed : int
        act_fn:
            #TODO: complete documentation

        Returns
        -------

        """

        return Fnn(sizes=sizes, init_seed=seed, act_fn=act_fn)

    def deep_ensemble_creator(self, seed: int = 0, *, act_fn, sizes: list[int] = None) -> list[Fnn]:
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

    def fit_members(self, x, y, epochs, optimizer=None, loss=None):
        # Setup task
        if self.task == TaskType.REGRESSION:
            n_outputs = 2
        elif self.task == TaskType.CLASSIFICATION:
            n_outputs = int(y.max()) + 1
        else:
            raise ValueError(f"Unknown task {self.task} !")

        # Setup FNN sizes
        n_features = x.shape[1]
        full_sizes = [n_features] + self.hidden_sizes + [n_outputs]

        # Setup members
        if self.members is None:
            self.members = self.deep_ensemble_creator(seed=self.seed, act_fn=self.act_fn, sizes=full_sizes)

        # train and validation sets for early stopping
        x_train, x_val, y_train, y_val = super().split_train_val(x, y)

        # Setup training objects
        proto_model = self.members[0]
        opt = optimizer or self.default_optimizer()
        loss_obj = loss or self.default_loss(self.task)
        loss_fn = FnnTrainer.make_loss_fn(proto_model, loss_obj)  # (params, x, y) -> scalar
        step_one = FnnTrainer.make_step(loss_fn, opt)  # (params, opt_state, x, y) -> (params, opt_state, loss)

        # Setup paramtree
        params_e = tree_map(lambda *ps: jnp.stack(ps, axis=0), *[m.params for m in self.members])  # (E, ...)

        # Distribute members to devices via padding
        E = len(self.members)
        D = jax.local_device_count()

        print("Kernel devices:", D)

        pad = (D - (E % max(D, 1))) % max(D, 1)
        E_pad = E + pad
        E_per = E_pad // max(D, 1)

        params_e = tree_map(lambda a: _pad_axis0(a, pad), params_e)
        params_de = tree_map(lambda a: a.reshape(D, E_per, *a.shape[1:]), params_e)

        # Per-device init of optimizer states (vectorized over local chunk)
        def init_chunk(params_chunk):
            return jax.vmap(opt.init)(params_chunk)

        opt_state_de = jax.pmap(init_chunk, in_axes=0, out_axes=0)(params_de)

        # One device does a local vmap over its chunk
        def step_chunk(params_chunk, opt_state_chunk, x_b, y_b, stopped_chunk):
            def step_member(p, s):
                return step_one(p, s, x_b, y_b)

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
                return loss_fn(p, x_b, y_b)

            return jax.vmap(loss_member)(params_chunk)

        pstep = jax.pmap(step_chunk, in_axes=(0, 0, None, None, 0), out_axes=(0, 0, 0))
        peval = jax.pmap(eval_chunk, in_axes=(0, None, None), out_axes=0)

        # Setup training loop
        self._reset_history()
        # best_params_de = params_de
        # best_metric_de = jnp.full((D, E_per), jnp.inf)
        # epochs_no_improve_de = jnp.zeros((D, E_per), dtype=jnp.int32)
        # stopped_de = jnp.zeros((D, E_per), dtype=bool)
        # stop_epoch_de = -jnp.ones((D, E_per), dtype=jnp.int32)
        callback = EarlyStoppingCallback(
            patience=self.patience,
            min_delta=self.min_delta,
            eval_every=self.eval_every,
        )
        callback_state = callback.initialize(params_de)

        # Training loop
        for epoch in range(epochs):
            stopped_de = callback.stopped_mask(callback_state)
            params_de, opt_state_de, lvals_de = pstep(params_de, opt_state_de, x_train, y_train, stopped_de)
            train_mean = float(jnp.mean(jax.device_get(lvals_de)))
            self.history["train_loss"].append(train_mean)
            if epoch % self.log_every == 0:
                print(epoch, train_mean)

            if (x_val is not None) and (y_val is not None) and callback.should_evaluate(epoch):
                val_lvals_de = peval(params_de, x_val, y_val)
                callback_state = callback.update(callback_state, epoch, params_de, val_lvals_de)

                # Note: This output considers the dummy members aswell, which will be later neglected
                if epoch % 100 == 0:
                    n_active = callback.active_members(callback_state)
                    print(f"[epoch {epoch}] active members: {n_active}")

                if callback.all_stopped(callback_state):
                    print(f"All members stopped by epoch {epoch}.")
                    break

        # Output stop epochs
        stop_epoch_e = callback.stop_epochs(callback_state, ensemble_size=E)
        for m_id, ep in enumerate(list(map(int, jax.device_get(stop_epoch_e)))):
            print(f"member {m_id}: {'stopped at epoch ' + str(ep) if ep >= 0 else 'ran full training'}")

        params_e_final = callback.best_params(callback_state, ensemble_size=E)
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
        """
        #TODO: dcoumentation
        Parameters
        ----------
        item

        Returns
        -------

        """
        if item in self.results:
            return self.results[item]
        else:
            raise AttributeError(f"{self.__class__.__name__} has no attribute '{item}' !")
