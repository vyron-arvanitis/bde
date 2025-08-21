# models.py
import jax
import jax.numpy as jnp


class Fnn:
    """Builds a single FNN"""
    def __init__(self, sizes):
        self.sizes = sizes
        self.params = None  # will hold initialized weights

    def init_mlp(self, seed):
        key = jax.random.PRNGKey(seed)
        keys = jax.random.split(key, len(self.sizes) - 1)
        params = []
        for k, (m, n) in zip(keys, zip(self.sizes[:-1], self.sizes[1:])):
            W = jax.random.normal(k, (m, n)) / jnp.sqrt(m)
            b = jnp.zeros((n,))
            params.append((W, b))
        self.params = params
        return params
