"""Prediction utilities for trained Bayesian deep ensembles."""

from typing import Callable

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from .sampler.types import ParamTree
from .task import TaskType


class BdePredictor:
    """Lightweight wrapper that aggregates ensemble predictions on demand.

    Parameters
    ----------
    forward_fn : Callable
        Function accepting `(params, x)` and returning model outputs.
    positions_eT : ParamTree
        Posterior samples with leading ensemble and sample axes.
    xte : ArrayLike
        Feature matrix to evaluate.
    task : TaskType
        Task associated with the ensemble (regression or classification).
    """

    def __init__(
        self,
        forward_fn: Callable[[ParamTree, ArrayLike], jax.Array],
        positions_eT: ParamTree,
        xte: ArrayLike,
        task: TaskType,
    ):
        self._forward = forward_fn
        self.positions = positions_eT  # (E, T, ...)
        self.xte = xte
        self.task = task
        self._raw_preds: ArrayLike | None = None

    def get_raw_preds(self) -> ArrayLike:
        """Materialise ensemble predictions with leading axes ``(E, T, N, ...)``.

        Returns
        -------
        ArrayLike
            Cached logits or regression outputs for every ensemble member and sample.
        """

        if self._raw_preds is None:
            # pure apply using explicit params
            def apply_with_params(p):
                return self._forward(p, self.xte)

            self._raw_preds = jax.vmap(  # over ensemble members
                jax.vmap(apply_with_params, in_axes=0),  # over samples
                in_axes=0,
            )(self.positions)

        return self._raw_preds

    def _regression_mu_sigma(self) -> tuple[jax.Array, jax.Array]:
        """Split regression outputs into mean and scale tensors.

        Returns
        -------
        tuple[jax.Array, jax.Array]
            Predictive means and softplus-transformed scales with shape (E, T, N).
            The second regression head is treated as an unconstrained scale
            parameter and is transformed with ``jax.nn.softplus`` here and in
            the sampler log-likelihood so that training and sampling share the
            same parameterisation.
        """

        preds = self.get_raw_preds()  # (E, T, N, 2)
        mu = preds[..., 0]
        sigma = jax.nn.softplus(preds[..., 1]) + 1e-6
        return mu, sigma

    def _credible_intervals(
        self,
        mu: ArrayLike,
        sigma: ArrayLike,
        credible_intervals: list[float],
    ) -> ArrayLike:
        r"""Compute predictive credible intervals for a scalar regression model using
        Monte Carlo sampling from each Gaussian posterior component.

        This method implements the *expensive* but statistically correct approach:
        for every posterior parameter pair (mu, sigma), we draw ``n`` samples from
        :math:`\mathcal{N}(\mu, \sigma)` and compute quantiles over the entire set
        of synthetic predictions. This corresponds to estimating credible intervals
        from the full mixture distribution produced by the ensemble and posterior
        samples.

        Parameters
        ----------
        mu : jax.Array
            Predictive means with shape ``(E, T, N)`` where ``E`` is the number of
            ensemble members, ``T`` the number of posterior samples per member, and
            ``N`` the number of datapoints.

        sigma : jax.Array
            Predictive standard deviations with shape ``(E, T, N)``, obtained after
            applying the softplus transformation to the second regression head.

        credible_intervals : list[float]
            A list of quantile levels in ``[0, 1]`` (e.g. ``[0.1, 0.9]``), defining
            the desired credible interval bounds.

        Returns
        -------
        jax.Array
            Array of shape ``(Q, N)`` where ``Q = len(credible_intervals)``.
            Each row corresponds to a quantile evaluated across all samples drawn
            from the mixture of Gaussians defined by the ensemble and its posterior
            samples.
        """

        E, T, N, _ = self.get_raw_preds().shape

        n = 1 if T > 1000 else 10
        key = jax.random.PRNGKey(0)

        # Expand for broadcasting
        mu_exp = mu[:, :, :, None]  # (E,T,N,1)
        sigma_exp = sigma[:, :, :, None]  # (E,T,N,1)

        noise = jax.random.normal(key, shape=(E, T, N, n))  # (E,T,N,n)

        # Samples from the mixture of Gaussians
        samples = mu_exp + sigma_exp * noise  # (E,T,N,n)

        return jnp.quantile(
            samples, q=jnp.array(credible_intervals), axis=(0, 1, 3)
        )  # (Q,N)

    def get_preds_per_member(self) -> tuple[jax.Array, jax.Array]:
        """Summarise each ensemble member by mean prediction and total uncertainty.

        Returns
        -------
        tuple[jax.Array, jax.Array]
            Tuple of mean predictions and standard deviations shaped ``(E, N)``.
        """

        mu, sigma = self._regression_mu_sigma()
        mu_mean_e = jnp.mean(mu, axis=1)
        var_ale_e = jnp.mean(sigma**2, axis=1)
        var_epi_e = jnp.var(mu, axis=1)
        std_total_e = jnp.sqrt(var_ale_e + var_epi_e)
        return mu_mean_e, std_total_e

    def _predict_regression(
        self, *, mean_and_std: bool, credible_intervals: list[float] | None, raw: bool
    ) -> dict[str, jax.Array]:
        """Aggregate regression predictions under different output modalities.

        Parameters
        ----------
        mean_and_std : bool
            Whether to include total predictive standard deviation.
        credible_intervals : list[float] | None
            Optional quantile levels evaluated over the predictive distribution.
        raw : bool
            When `True`, include the cached raw ensemble outputs.

        Returns
        -------
        dict[str, jax.Array]
            Mapping of requested statistics (mean, std, credible_intervals, raw).
        """

        mu, sigma = self._regression_mu_sigma()
        mu_mean = jnp.mean(mu, axis=(0, 1))
        var_ale = jnp.mean(sigma**2, axis=(0, 1))  # aleatoric variance
        var_epi = jnp.var(mu, axis=(0, 1))  # epistemic variance
        std_total = jnp.sqrt(var_ale + var_epi)
        out = {"mean": mu_mean}
        if mean_and_std:
            out["std"] = std_total
        if credible_intervals:
            out["credible_intervals"] = self._credible_intervals(
                mu, sigma, credible_intervals
            )
        if raw:
            out["raw"] = self.get_raw_preds()
        return out

    def _predict_classification(
        self, *, probabilities: bool, raw: bool
    ) -> dict[str, jax.Array]:
        """Aggregate classification logits into labels and probabilities.

        Parameters
        ----------
        probabilities : bool
            Whether to include mean class probabilities.
        raw : bool
            When `True`, include raw logits for every sample.

        Returns
        -------
        dict[str, jax.Array]
            Mapping containing `labels`, optionally `probs` and `raw`.
        """

        logits = self.get_raw_preds()
        probs = jax.nn.softmax(logits, axis=-1)
        mean_probs = jnp.mean(probs, axis=(0, 1))
        preds_cls = jnp.argmax(mean_probs, axis=-1)

        out = {"labels": preds_cls}
        if probabilities:
            out["probs"] = mean_probs
        if raw:
            out["raw"] = logits
        return out

    def predict(
        self,
        *,
        mean_and_std=False,
        credible_intervals=None,
        raw=False,
        probabilities=False,
    ):
        """Return prediction artefacts appropriate for the configured task.

        Parameters
        ----------
        mean_and_std : bool
            Regression-only flag to include predictive standard deviation.
        credible_intervals : list[float] | None
            Regression-only quantile levels.
        raw : bool
            Include raw ensemble outputs.
        probabilities : bool
            Classification-only flag to return class probabilities.
        """

        if self.task == TaskType.REGRESSION:
            return self._predict_regression(
                mean_and_std=mean_and_std,
                credible_intervals=credible_intervals,
                raw=raw,
            )
        if self.task == TaskType.CLASSIFICATION:
            return self._predict_classification(probabilities=probabilities, raw=raw)
        raise ValueError(f"Unknown task {self.task}")
