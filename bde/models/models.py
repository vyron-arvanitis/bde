import jax
import jax.numpy as jnp

from bde.data.dataloader import DataLoader
from bde.training.trainer import FnnTrainer
from bde.data.dataloader import DataLoader


class Fnn:
    """Single FNN that can optionally train itself on init."""

    def __init__(self, sizes, init_seed=0):
        super().__init__()  # init the trainer side (history, etc.)
        self.sizes = sizes
        self.params = self.init_mlp(seed=init_seed)

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

    @staticmethod
    def forward(params, x):
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
            x = jnp.tanh(x)
        W, b = params[-1]  # Fixed indentation - this should be outside the loop
        return jnp.dot(x, W) + b

    def predict(self, x):
        if self.params is None:
            raise ValueError("Model parameters not initialized!")
        return self.forward(self.params, x)
