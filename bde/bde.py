from abc import abstractclassmethod

from pandas.core.window.doc import kwargs_scipy

from bde.bde_builder import BdeBuilder
from bde.bde_evaluator import BDEPredictor
from bde.loss.loss import BaseLoss
from bde.sampler.probabilistic import ProbabilisticModel
from bde.sampler.prior import PriorDist
from bde.sampler.warmup import warmup_bde
from bde.sampler.mile_wrapper import MileWrapper
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from jax.tree_util import tree_map, tree_leaves
from bde.task import TaskType
from functools import partial
import optax
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin


class Bde(BaseEstimator):
    def __init__(self,
                 n_members: int = 5,
                 hidden_layers: list = None,
                 seed: int = 0,
                 task: TaskType = None,
                 loss: BaseLoss = None,
                 activation: str = "relu",
                 epochs: int = 100,
                 patience: int = 25,
                 n_samples: int = 100,
                 warmup_steps: int = 50,
                 lr: float = 1e-3,
                 n_thinning: int = 10,
                 desired_energy_var_start: float = 0.5,
                 desired_energy_var_end: float = 0.1,
                 step_size_init: int = None
                 ):

        self.n_members = n_members
        self.hidden_layers = hidden_layers
        self.seed = seed
        self.task = task
        self.loss = loss
        self.task.validate_loss(self.loss)  # validate loss function
        self.activation = activation
        self.epochs = epochs
        self.patience = patience
        self.n_samples = n_samples
        self.warmup_steps = warmup_steps
        self.lr = lr
        self.n_thinning = n_thinning
        self.step_size_init = step_size_init if step_size_init is not None else lr
        self.desired_energy_var_start = desired_energy_var_start
        self.desired_energy_var_end = desired_energy_var_end

        if self.hidden_layers is not None:
            self._build_bde()

        self.positions_eT = None  # will be set after training + sampling

    def _build_bde(self):
        self.bde = BdeBuilder(
            hidden_sizes=self.hidden_layers,
            patience=self.patience,
            n_members=self.n_members,
            task=self.task,
            seed=self.seed,
            act_fn=self.activation
        )

        self.members = self.bde.members

    def _build_log_post(self, x: ArrayLike, y: ArrayLike):
        prior = PriorDist.STANDARDNORMAL.get_prior()
        proto = self.bde.members[0]
        pm = ProbabilisticModel(module=proto, params=proto.params, prior=prior, task=self.task)
        return partial(pm.log_unnormalized_posterior, x=x, y=y)

    def _warmup_sampler(self, logpost):
        warm = warmup_bde(
            self.bde,
            logpost,
            step_size_init=self.step_size_init,
            warmup_steps=self.warmup_steps,
            desired_energy_var_start=self.desired_energy_var_start,
            desired_energy_var_end=self.desired_energy_var_end,
        )

        return warm.state.position, warm.parameters  # (pytree with leading E,  MCLMCAdaptationState)

    def _generate_rng_keys(self, num_chains: int):
        rng = jax.random.PRNGKey(int(self.seed))
        return jax.vmap(lambda i: jax.random.fold_in(rng, i))(jnp.arange(num_chains))

    @staticmethod
    def _normalize_tuned_parameters(tuned, num_chains: int):
        L_e = tuned.L if jnp.ndim(tuned.L) == 1 else jnp.full((num_chains,), tuned.L)
        step_e = tuned.step_size if jnp.ndim(tuned.step_size) == 1 else jnp.full((num_chains,), tuned.step_size)
        sqrt_diag_e = tuned.sqrt_diag_cov
        return L_e, step_e, sqrt_diag_e

    def _draw_samples(self,
                      logpost,
                      rng_keys_e,
                      init_positions_e,
                      L_e,
                      step_e,
                      sqrt_diag_e,
                      ):
        sampler = MileWrapper(logpost)
        positions_eT, _, _ = sampler.sample_batched(
            rng_keys_e=rng_keys_e,
            init_positions_e=init_positions_e,
            num_samples=self.n_samples,
            thinning=self.n_thinning,
            L_e=L_e,
            step_e=step_e,
            sqrt_diag_e=sqrt_diag_e,
            store_states=True,
        )
        return positions_eT

    def fit(self, x: ArrayLike, y: ArrayLike):

        self.bde.fit_members(x=x, y=y, optimizer=optax.adam(self.lr), epochs=self.epochs, loss=self.loss)

        logpost_one = self._build_log_post(x, y)
        init_positions_e, tuned = self._warmup_sampler(logpost_one)

        num_chains = tree_leaves(init_positions_e)[0].shape[0]
        rng_keys_e = self._generate_rng_keys(num_chains)
        L_e, step_e, sqrt_diag_e = self._normalize_tuned_parameters(tuned, num_chains)

        self.positions_eT = self._draw_samples(
            logpost_one,
            rng_keys_e,
            init_positions_e,
            L_e,
            step_e,
            sqrt_diag_e,
        )

        return self

    def evaluate(self,
                 xte: ArrayLike,
                 mean_and_std: bool = False,
                 credible_intervals: list[float] | None = None,
                 raw: bool = False,
                 probabilities: bool = False):
        predictor = BDEPredictor(self.bde.members[0], self.positions_eT, Xte=xte,
                                 task=self.task)  # TODO: [@angelos] think of somehting better I dont think that works fine
        raw_preds = predictor.get_raw_preds()
        if self.task == TaskType.REGRESSION:
            mu = raw_preds[..., 0]
            sigma = jax.nn.softplus(raw_preds[..., 1]) + 1e-6
            mu_mean = jnp.mean(mu, axis=(0, 1))
            var_ale = jnp.mean(sigma ** 2, axis=(0, 1))
            var_epi = jnp.var(mu, axis=(0, 1))
            std_total = jnp.sqrt(var_ale + var_epi)

            out = {"mean": mu_mean}
            if mean_and_std:
                out["mean"] = mu_mean
                out["std"] = std_total
            if credible_intervals:
                qs = jnp.quantile(mu, q=jnp.array(credible_intervals), axis=(0, 1))
                out["credible_intervals"] = qs
            if raw:
                out["raw"] = raw_preds
            return out


        elif self.task == TaskType.CLASSIFICATION:
            logits = raw_preds  # (E, T, N, C)
            probs = jax.nn.softmax(logits, axis=-1)
            mean_probs = jnp.mean(probs, axis=(0, 1))  # (N, C)
            preds_cls = jnp.argmax(mean_probs, axis=-1)

            out = {}
            if probabilities:
                out["probs"] = mean_probs
            out["labels"] = preds_cls
            if raw:
                out["raw"] = raw_preds
            return out

        else:
            raise ValueError(f"Unknown task {self.task}")


# TODO: [@angelos] maybe put them in another file?
class BdeRegressor(Bde, RegressorMixin):
    def __init__(self, **kwargs):
        super().__init__(task=TaskType.REGRESSION, **kwargs)

    def predict(self,
                x: ArrayLike, mean_and_std: bool = False,
                credible_intervals: list[float] = None,
                raw: bool = False):
        out = self.evaluate(
            x,
            mean_and_std=mean_and_std,
            credible_intervals=credible_intervals,
            raw=raw,
        )
        if mean_and_std:
            return out["mean"], out["std"]
        elif credible_intervals:
            return out["mean"], out["credible_intervals"]
        return out["mean"]

    def get_raw_predictions(self, x: ArrayLike):
        """Return raw ensemble predictions.

        Shape: (E, T, N, 2), where:
          - E = ensemble members
          - T = posterior samples per member
          - N = number of test points
          - 2 = (mu, sigma_param)
        """
        return self.evaluate(x, raw=True)["raw"]


class BdeClassifier(Bde, ClassifierMixin):
    def __init__(self, **kwargs):
        super().__init__(task=TaskType.CLASSIFICATION, **kwargs)

    def predict(self, x: ArrayLike):
        out = self.evaluate(x)
        return out["labels"]

    def predict_proba(self, x: ArrayLike):
        out = self.evaluate(x, probabilities=True)
        return out["probs"]

    def get_raw_predictions(self, x: ArrayLike):
        """Return raw ensemble predictions.

        Shape: (E, T, N, C), where:
          - E = ensemble members
          - T = posterior samples per member
          - N = number of test points
          - C = number of classes (logits before softmax)
        """
        return self.evaluate(x, raw=True)["raw"]
