"""Priors. This is used from @MILE."""
from typing import Callable, NamedTuple

import jax
import jax.nn.initializers as jinit
import jax.numpy as jnp
import jax.scipy.stats as stats
from jax.flatten_util import ravel_pytree

from bde.sampler.types import ParamTree
from bde.sampler.types import BaseStrEnum


class PriorDist(BaseStrEnum):
    NORMAL = "Normal"  # allow changing the normal for the user
    STANDARDNORMAL = "StandardNormal"  # maybe remove
    LAPLACE = "Laplace"

    def get_prior(self, **parameters):
        """Return a prior class."""
        return Prior.from_name(self, **parameters)


class Prior(NamedTuple):
    """Base Class for Priors."""

    f_init: Callable
    log_prior: Callable[[ParamTree], jax.Array]
    name: str

    @classmethod
    def from_name(cls, name: PriorDist, **parameters):
        """Initialize the prior class instance."""
        if name == PriorDist.STANDARDNORMAL:
            return cls(
                name=PriorDist.STANDARDNORMAL,
                f_init=f_init_normal(),
                log_prior=log_prior_normal(),
            )
        elif name == PriorDist.NORMAL:
            return cls(
                name=PriorDist.NORMAL,
                f_init=f_init_normal(**parameters),
                log_prior=log_prior_normal(**parameters),
            )
        elif name == PriorDist.LAPLACE:
            return cls(
                name=PriorDist.LAPLACE,
                f_init=f_init_laplace(**parameters),
                log_prior=log_prior_laplace(**parameters),
            )
        else:
            raise NotImplementedError(
                f'Prior Distribution for {name} is not yet implemented.'
            )


def f_init_normal(loc: float = 0.0, scale: float = 1.0) -> Callable:
    """Initialize from Normal distribution."""
    if not loc == 0.0:
        print('Initialization: Ignoring location unequal to zero.')
    return jinit.normal(stddev=scale)


def log_prior_normal(loc: float = 0.0, scale: float = 1.0) -> Callable:
    """Evaluate Normal prior on all weights."""

    def log_prior(params: ParamTree) -> jax.Array:
        scores = stats.norm.logpdf(ravel_pytree(params)[0], loc=loc, scale=scale)
        return jnp.sum(scores)

    return log_prior


def f_init_laplace(loc: float = 0.0, scale: float = 1.0) -> Callable:
    """Initialize from Laplace distribution."""

    def f_init(key, shape, dtype=jnp.float32) -> jax.Array:
        w = jax.random.laplace(key, shape, dtype)
        return w * scale + loc

    return f_init


def log_prior_laplace(loc: float = 0.0, scale: float = 1.0) -> Callable:
    """Evaluate Laplace prior on all weights."""

    def log_prior(params: ParamTree) -> jax.Array:
        scores = stats.laplace.logpdf(ravel_pytree(params)[0], loc=loc, scale=scale)
        return jnp.sum(scores)

    return log_prior
