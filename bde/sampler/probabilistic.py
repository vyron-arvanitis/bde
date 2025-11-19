"""Convert frequentist Flax modules to Bayesian NNs. This is used from @MILE."""

import logging

import jax
import jax.numpy as jnp
import jax.scipy.stats as stats

import bde.sampler.utils as train_utils
from bde.models import BaseModel
from bde.sampler.prior import Prior
from bde.sampler.types import ParamTree
from bde.task import TaskType

logger = logging.getLogger(__name__)


class ProbabilisticModel:
    """Convert frequentist Flax modules to Bayesian Numpyro modules."""

    def __init__(
        self, model: BaseModel, params: ParamTree, prior: Prior, task: TaskType
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

        self.model = model
        self.n_params = train_utils.count_params(params)
        self.prior = prior
        self.task = task

    def __str__(self):
        """Return informative string representation of the model."""
        return (
            f"{self.__class__.__name__}:\n"  # noqa
            f" | Params: {self.n_params}"
            f" | Prior: {self.prior.name}"
        )

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

        Returns
        -------
        jnp.ndarray
            Scalar log-likelihood of the batch under the current task and parameters.
        """
        lvals = self.model.forward(params, x)

        if self.task == TaskType.REGRESSION:
            return jnp.nansum(
                stats.norm.logpdf(
                    x=y,
                    loc=lvals[..., 0:1],
                    scale=(jax.nn.softplus(lvals[..., 1:2]) + 1e-6).clip(
                        min=1e-6, max=1e6
                    ),
                )
            )
        elif self.task == TaskType.CLASSIFICATION:
            logits = lvals
            log_probs = logits - jax.scipy.special.logsumexp(
                logits, axis=-1, keepdims=True
            )
            return jnp.sum(jnp.take_along_axis(log_probs, y[..., None], axis=-1))
        else:
            raise NotImplementedError(
                f"Task {self.task} not implemented in log_likelihood"
            )

    def log_unnormalized_posterior(
        self,
        positions: ParamTree,
        x: jnp.ndarray,
        y: jnp.ndarray,
        **kwargs,
    ):
        """Log unnormalized posterior (potential) for given parameters and data.

        Parameters:
        -----------
        positions: ParamTree
            Parameters of the model.
        x: jnp.ndarray
            Input data of shape (batch_size, ...).
        y: jnp.ndarray
            Target data of shape (batch_size, ...).
        kwargs: dict
            Additional keyword arguments to pass to the model forward pass.
        """

        return self.log_prior(positions) + self.log_likelihood(
            positions, x, y, **kwargs
        )
