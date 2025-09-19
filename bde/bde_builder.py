"""this is a bde builder"""
import jax, jaxlib
import jax.numpy as jnp
from jax.tree_util import tree_map, tree_structure

import optax

from .models.models import Fnn
from .training.trainer import FnnTrainer

from bde.sampler.my_types import ParamTree
from bde.sampler.utils import _infer_dim_from_position_example, _pad_axis0, _reshape_to_devices
from bde.task import TaskType


class BdeBuilder(FnnTrainer):
    # TODO: build the BdeBuilderClass
    def __init__(self,
                 hidden_sizes,
                 n_members,
                 task: TaskType,
                 seed: int = 100,
                 act_fn: str = "relu"):

        # Fnn.__init__(self, hidden_layers=hidden_sizes, act_fn=act_fn)
        FnnTrainer.__init__(self)
        self.hidden_sizes = hidden_sizes
        self.n_members = n_members
        self.seed = seed
        self.task = task
        self.params_e = None  # will be set after training
        self.members = None  #
        self.act_fn = act_fn

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
        n_features = x.shape[1]
        if self.task == TaskType.REGRESSION:
            n_outputs = 2
        elif self.task == TaskType.CLASSIFICATION:
            n_outputs = int(y.max()) + 1
        else:
            raise ValueError(f"Unknown task {self.task} !")

        full_sizes = [n_features] + self.hidden_sizes + [n_outputs]

        if self.members is None:
            self.members = self.deep_ensemble_creator(seed=self.seed, act_fn=self.act_fn, sizes=full_sizes)

        E = len(self.members)
        print("backend:", jax.default_backend())
        print("devices:", jax.devices())
        print("local_device_count:", jax.local_device_count())
        D = jax.local_device_count()
        print("Kernel devices:", D)

        opt = optimizer or self.default_optimizer()
        loss_obj = loss or self.default_loss(self.task)

        # Stack params across ensemble axis E
        params_e = tree_map(lambda *ps: jnp.stack(ps, axis=0), *[m.params for m in self.members])  # (E, ...)
        proto_model = self.members[0]
        loss_fn = FnnTrainer.make_loss_fn(proto_model, loss_obj)  # (params, x, y) -> scalar
        step_one = FnnTrainer.make_step(loss_fn, opt)  # (params, opt_state, x, y) -> (params, opt_state, loss)

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
        def step_chunk(params_chunk, opt_state_chunk, x_b, y_b):
            def step_member(p, s):
                return step_one(p, s, x_b, y_b)  # pure, no host I/O

            new_params, new_states, losses = jax.vmap(step_member)(params_chunk, opt_state_chunk)
            return new_params, new_states, losses

        # pmapped step across devices; broadcast data to all devices (in_axes=None)
        pstep = jax.pmap(step_chunk, in_axes=(0, 0, None, None), out_axes=(0, 0, 0))

        self._reset_history()
        for epoch in range(epochs):
            params_de, opt_state_de, lvals_de = pstep(params_de, opt_state_de, x, y)
            # lvals_de shape: (D, E_per). Log mean loss (host side)
            mean_loss = float(jnp.mean(jax.device_get(lvals_de)))
            self.history["train_loss"].append(mean_loss)
            if epoch % self.log_every == 0:
                print(epoch, mean_loss)

        # Stitch back to (E, ...) then write into members
        params_e_final = tree_map(lambda a: a.reshape(E_pad, *a.shape[2:])[:E], params_de)
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
