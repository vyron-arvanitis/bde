import jax
import jax.numpy as jnp
from jax.tree_util import tree_map
from bde.task import TaskType


class BDEPredictor:  # TODO: [@question] Maybe merge with BDE Builder class
    def __init__(self, model, positions_eT, Xte, task: TaskType):
        self.model = model
        self.positions = positions_eT  # (E, T, ...)
        self.Xte = Xte
        self.task = task

    def get_raw_preds(self):
        # pure apply using explicit params (no mutation of model.params)
        def apply_with_params(p):
            return self.model.forward(p, self.Xte)  # (N, 2)

        return jax.vmap(  # over ensemble members
            jax.vmap(apply_with_params, in_axes=0),  # over samples
            in_axes=0
        )(self.positions)

        # if self.task == TaskType.REGRESSION:
        #     mu = preds[..., 0]
        #     sigma = jax.nn.softplus(preds[..., 1]) + 1e-6
        #     mu_mean = jnp.mean(mu, axis=(0, 1))
        #     var_ale = jnp.mean(sigma ** 2, axis=(0, 1))
        #     var_epi = jnp.var(mu, axis=(0, 1))
        #     std_total = jnp.sqrt(var_ale + var_epi)
        #
        #     return mu_mean, std_total
        #
        # elif self.task == TaskType.CLASSIFICATION:
        #     logits = preds  # (E, T, N, C)
        #     probs = jax.nn.softmax(logits, axis=-1)
        #     mean_probs = jnp.mean(probs, axis=(0, 1))  # average over ensemble and samples
        #     preds_cls = jnp.argmax(mean_probs, axis=-1)
        #     return mean_probs, preds_cls
        # else:
        #     raise ValueError(f"Unknown task {self.task}")

    def get_preds_per_member(self):
        def apply_with_params(p):
            return self.model.forward(p, self.Xte)  # (N, 2)

        preds = jax.vmap(jax.vmap(apply_with_params, in_axes=0), in_axes=0)(self.positions)  # (E, T, N, 2)
        mu = preds[..., 0]  # (E, T, N)
        sigma = jax.nn.softplus(preds[..., 1]) + 1e-6  # (E, T, N)

        mu_mean_e = jnp.mean(mu, axis=1)  # (E, N)
        var_ale_e = jnp.mean(sigma ** 2, axis=1)  # (E, N)
        var_epi_e = jnp.var(mu, axis=1)  # (E, N)
        std_total_e = jnp.sqrt(var_ale_e + var_epi_e)  # (E, N)
        return mu_mean_e, std_total_e

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
