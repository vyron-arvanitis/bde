"""Convert frequentist Flax modules to Bayesian NNs."""
import logging

import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.scipy.stats as stats

import bde.sampler.utils as train_utils
from bde.sampler.prior import Prior
from bde.sampler.my_types import ParamTree

logger = logging.getLogger(__name__)


class ProbabilisticModel:
    """Convert frequentist Flax modules to Bayesian Numpyro modules."""

    def __init__(
        self,
        module: nn.Module,
        params: ParamTree,
        prior: Prior,
        n_batches: int = 1,
    ):
        """Initialize the ProbabilisticModel class for Bayesian training.

        Parameters:
        -----------
        module: nn.Module
            Flax module to use for Bayesian training
        params: ParamTree
            Parameters of the Flax module.
        prior: Prior
            Prior distribution for the parameters.
        task: Task
            Task type (classification or regression)
        n_batches: int
            Number of batches sampler will see in one epoch.
        """
        self.module = module
        self.n_params = train_utils.count_params(params)
        self.n_batches = n_batches
        self.prior = prior

    def __str__(self):
        """Return informative string representation of the model."""
        return (
            f'{self.__class__.__name__}:\n'  # noqa
            f' | Params: {self.n_params}'
            f' | Batches: {self.n_batches}\n'
            f' | Prior: {self.prior.name}'
        )

    @property
    def minibatch(self):
        return self.n_batches > 1 ### no batches ###

    def log_prior(self, params: ParamTree) -> jax.Array:
        """Compute log prior for given parameters."""
        return self.prior.log_prior(params)

    def log_likelihood(
        self,
        params: ParamTree,
        x: jnp.ndarray,
        y: jnp.ndarray,
        **kwargs,
    ) -> jnp.ndarray:
        """Evaluate Log likelihood for given Parameter Tree and data.

        Parameters:
        -----------
        params: ParamTree
            Parameters of the model.
        x: jnp.ndarray
            Input data of shape (batch_size, ...).
        y: jnp.ndarray
            Target data of shape (batch_size, ...).
        kwargs: dict
            Additional keyword arguments to pass to the model forward pass.

        Raises:
        -------
        NotImplementedError: If computation for given `task` is not implemented.
        """

        lvals = self.module.apply({'params': params}, x, **kwargs)
        
        return jnp.nansum(
            stats.norm.logpdf(
                x=y,
                loc=lvals[..., 0:1],
                scale=jnp.exp(lvals[..., 1:2]).clip(min=1e-6, max=1e6),
            )
        )
        

    def log_unnormalized_posterior(
        self,
        position: ParamTree,
        x: jnp.ndarray,
        y: jnp.ndarray,
        **kwargs,
    ):
        """Log unnormalized posterior (potential) for given parameters and data.

        Parameters:
        -----------
        params: ParamTree
            Parameters of the model.
        x: jnp.ndarray
            Input data of shape (batch_size, ...).
        y: jnp.ndarray
            Target data of shape (batch_size, ...).
        kwargs: dict
            Additional keyword arguments to pass to the model forward pass.
        """
        return (
            self.log_prior(position)
            + self.log_likelihood(position, x, y, **kwargs) * self.n_batches
        )
