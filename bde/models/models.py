from abc import ABC, abstractmethod

import jax
import jax.nn as nn
import jax.numpy as jnp
from jax.typing import ArrayLike

from bde.sampler.types import ParamList


class BaseModel(ABC):
    name: str

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def forward(self, params, x):
        ...


class Fnn(BaseModel):
    """Single FNN that can optionally train itself on init."""

    def __init__(self, sizes: list[int], init_seed: int = 0, *, act_fn: str):
        """Initialize a fully-connected feed-forward neural network.

        Parameters
        ----------
        sizes : list[int]
            List of layer sizes. `sizes[0]` is the input dimension, and
            `sizes[-1]` is the output dimension. Each consecutive pair defines
            one linear layer.
        init_seed : int
            Seed used to initialize the random number generator for parameter
            initialization.
        act_fn : str
            Name of the activation function to use for all hidden layers.
            Must be supported by `_get_activation`.
        """

        self.sizes = sizes
        self.params = self._init_mlp(seed=init_seed)
        self.activation_name = act_fn

    @property
    def name(self) -> str:
        return "Fnn"

    def _init_mlp(self, seed: int) -> ParamList:
        """Initialize weights and biases for each layer of the MLP.

        Parameters
        ----------
        seed : int
            Seed used to initialize the random number generator for parameter
            initialization.

        Returns
        -------
        ParamList
            A list of `(W, b)` pairs, one for each linear layer.
            `W` has shape (in_dim, out_dim); `b` has shape (out_dim,).
        """

        key = jax.random.PRNGKey(seed)
        keys = jax.random.split(key, len(self.sizes) - 1)
        params = []
        for k, (m, n) in zip(keys, zip(self.sizes[:-1], self.sizes[1:])):
            W = jax.random.normal(k, (m, n)) / jnp.sqrt(m)
            b = jnp.zeros((n,))
            params.append((W, b))
        return params

    def forward(self, params: ParamList, x: ArrayLike) -> ArrayLike:
        """Run a forward pass of the network.

        Parameters
        ----------
        params : list[tuple[jax.Array, jax.Array]]
            Sequence of `(weights, bias)` pairs produced by `_init_mlp`.
        x : ArrayLike
            Input batch shaped '(n_samples, n_features)'.

        Returns
        -------
        Arraylike
            Model outputs with shape '(n_samples, output_dim)'.
        """

        n_features = x.shape[-1]
        expected_features = self.sizes[0]
        if n_features != expected_features:
            raise ValueError(
                f"Input feature dimension mismatch: expected {expected_features}, got"
                f" {n_features}"
            )
        act_fn = self._get_activation(self.activation_name)
        for W, b in params[:-1]:
            x = act_fn(jnp.dot(x, W) + b)
        W, b = params[-1]
        return jnp.dot(x, W) + b

    @staticmethod
    def _get_activation(activation: str):
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
