import jax
import jax.numpy as jnp

import jax.nn as nn
from bde.data.dataloader import DataLoader
from bde.training.trainer import FnnTrainer
from bde.data.dataloader import DataLoader


class Fnn:
    """Single FNN that can optionally train itself on init."""

    def __init__(self, sizes, init_seed=0, *, act_fn):
        """
        #TODO: documentation

        Parameters
        ----------
        sizes
        init_seed
        """
        super().__init__()  # init the trainer side (history, etc.)
        self.sizes = sizes
        self.params = self.init_mlp(seed=init_seed)
        self.act_fn = self._get_activation(act_fn)

    def init_mlp(self, seed):
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
        """
        #TODO: documentation

        Parameters
        ----------
        params
        x

        Returns
        -------

        """
        for (W, b) in params[:-1]:
            x = jnp.dot(x, W) + b
            # x = jax.nn.relu(x)
            x = self.act_fn(x)
        W, b = params[-1]
        return jnp.dot(x, W) + b

    def apply(self, variables, x, **kwargs):
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
        }
        try:
            return available_activation[activation]
        except KeyError:
            raise ValueError(
                f"Unsupported activation function: '{activation}'.\n"
                f"Available options: {', '.join(available_activation.keys())}"
            )
