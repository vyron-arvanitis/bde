import jax
import jax.numpy as jnp

import jax.nn as nn
from abc import ABC, abstractmethod
from bde.data.dataloader import DataLoader
from bde.training.trainer import FnnTrainer
from bde.data.dataloader import DataLoader


class BaseModel(ABC):
    name: str

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def forward(self, params, x): ...


class Fnn(BaseModel):
    """Single FNN that can optionally train itself on init."""

    def __init__(self, sizes, init_seed: int =0, *, act_fn: str):
        """
        #TODO: documentation

        Parameters
        ----------
        sizes
        init_seed
        """
        super().__init__()  # init the trainer side (history, etc.)
        self.sizes = sizes
        self.params = self._init_mlp(seed=init_seed)
        self.activation_name = act_fn
        # self.act_fn = self._get_activation(act_fn) TODO: [@delete] because it is non pickable and raises errors in test

    @property
    def name(self) -> str:
        return "Fnn"

    def _init_mlp(self, seed):
        """
        #TODO:documentation
        Parameters
        ----------
        seed

        Returns
        -------

        """
        key = jax.random.PRNGKey(seed)
        keys = jax.random.split(key, len(self.sizes) - 1)
        params = []
        for k, (m, n) in zip(keys, zip(self.sizes[:-1], self.sizes[1:])):
            W = jax.random.normal(k, (m, n)) / jnp.sqrt(m)
            b = jnp.zeros((n,))
            params.append((W, b))
        self.params = params
        return params

    def forward(self, params, x):
        """Run a forward pass of the network.

        Parameters
        ----------
        params : list[tuple[jax.Array, jax.Array]]
            Sequence of `(weights, bias)` pairs produced by `_init_mlp`.
        x : ArrayLike
            Input batch shaped (n_samples, n_features).

        Returns
        -------
        jax.Array
            Model outputs with shape (n_samples, output_dim).
        """
        # TODO: [@later] have a validation of input layer and number of features
        act_fn = self._get_activation(self.activation_name)
        for (W, b) in params[:-1]:
            x = act_fn(jnp.dot(x, W) + b)
        W, b = params[-1]
        return jnp.dot(x, W) + b

    def apply(self, variables, x, **kwargs): #TODO[@angelos, @vyron]: this is not used somewhere relevant!
        """Mimic Flax API: variables['params'] contains weights."""
        params = variables["params"]
        return self.forward(params, x, **kwargs)

    @staticmethod
    def _get_activation(activation):
        available_activation = {
            "relu": nn.relu,
            "relu6": nn.relu6,
            "sigmoid": nn.sigmoid,
            "softplus": nn.softplus,
            "log_sigmoid": nn.log_sigmoid,
            "soft_sign": nn.soft_sign,
            "silu": nn.silu,  # same as swish
            "swish": nn.swish,
            "leaky_relu": nn.leaky_relu,
            "hard_sigmoid": nn.hard_sigmoid,
            "hard_silu": nn.hard_silu,
            "hard_swish": nn.hard_swish,
            "hard_tanh": nn.hard_tanh,
            "elu": nn.elu,
            "celu": nn.celu,
            "selu": nn.selu,
            "gelu": nn.gelu,
            "glu": nn.glu,
            "squareplus": nn.squareplus,
            "mish": nn.mish,
            "identity": lambda x: x,
        }
        try:
            return available_activation[activation]
        except KeyError:
            raise ValueError(
                f"Unsupported activation function: '{activation}'.\n"
                f"Available options: {', '.join(available_activation.keys())}"
            )
