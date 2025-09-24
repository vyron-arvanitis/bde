import jax
import jax.numpy as jnp
from jax.tree_util import tree_map
from bde.task import TaskType
from .models.models import BaseModel
from jax.typing import ArrayLike


class BdePredictor:  # TODO: [@question] Maybe merge with BDE Builder class
    def __init__(self, forward_fn, positions_eT, xte, task: TaskType):
        self._forward = forward_fn
        self.positions = positions_eT  # (E, T, ...)
        self.xte = xte
        self.task = task
        self._raw_preds: jax.vmap = None

    def get_raw_preds(self):
        """

        Returns
        -------

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

    def _regression_mu_sigma(self):
        """

        Returns
        -------
        mu, sigma:
        """
        preds = self.get_raw_preds()  # (E, T, N, 2)
        mu = preds[..., 0]
        sigma = jax.nn.softplus(preds[..., 1]) + 1e-6
        return mu, sigma

    def get_preds_per_member(self):
        """

        Returns
        -------

        """
        mu, sigma = self._regression_mu_sigma()
        mu_mean_e = jnp.mean(mu, axis=1)
        var_ale_e = jnp.mean(sigma ** 2, axis=1)
        var_epi_e = jnp.var(mu, axis=1)
        std_total_e = jnp.sqrt(var_ale_e + var_epi_e)
        return mu_mean_e, std_total_e

    def _predict_regression(self, *, mean_and_std: bool, credible_intervals: list[float], raw: bool) -> dict:
        """

        Parameters
        ----------
        mean_and_std: bool
        credible_intervals : list[float]
        raw: bool

        Returns
        -------
        out: dict

        """

        mu, sigma = self._regression_mu_sigma()
        mu_mean = jnp.mean(mu, axis=(0, 1))
        var_ale = jnp.mean(sigma ** 2, axis=(0, 1))
        var_epi = jnp.var(mu, axis=(0, 1))
        std_total = jnp.sqrt(var_ale + var_epi)
        out = {"mean": mu_mean}
        if mean_and_std:
            out["std"] = std_total
        if credible_intervals:
            qs = jnp.quantile(mu, q=jnp.array(credible_intervals), axis=(0, 1))
            out["credible_intervals"] = qs
        if raw:
            out["raw"] = self.get_raw_preds()
        return out

    def _predict_classification(self, *, probabilities: bool, raw: bool) -> dict:
        """

        Parameters
        ----------
        probabilities: bool
        raw: bool

        Returns
        -------
        out : dict

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

    def predict(self, *, mean_and_std=False, credible_intervals=None, raw=False, probabilities=False):
        if self.task == TaskType.REGRESSION:
            return self._predict_regression(
                mean_and_std=mean_and_std,
                credible_intervals=credible_intervals,
                raw=raw,
            )
        if self.task == TaskType.CLASSIFICATION:
            return self._predict_classification(probabilities=probabilities, raw=raw)
        raise ValueError(f"Unknown task {self.task}")

    @staticmethod
    def predictive_accuracy(y, mu, sigma):
        # Pulls
        pulls = (y - mu) / sigma
        pull_mean = jnp.mean(pulls)
        pull_std = jnp.std(pulls)

        # Coverage
        within_1sigma = jnp.mean(jnp.abs(y - mu) <= 1 * sigma)
        within_2sigma = jnp.mean(jnp.abs(y - mu) <= 2 * sigma)
        within_3sigma = jnp.mean(jnp.abs(y - mu) <= 3 * sigma)

        return {
            "pull_mean": float(pull_mean),
            "pull_std": float(pull_std),
            "coverage_1σ": float(within_1sigma),
            "coverage_2σ": float(within_2sigma),
            "coverage_3σ": float(within_3sigma),
        }

    # credible intervals arg "q" for quantile
