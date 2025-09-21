from bde.bde_builder import BdeBuilder
from bde.bde_evaluator import BdePredictor
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
from typing import Any, Protocol, cast

class _WarmupState(Protocol):
    """Protocol for warmup states exposing the current position."""

    position: Any


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
        if self.task is not None and self.loss is not None:
            self.task.validate_loss(self.loss)
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
        warm_state = cast(_WarmupState, warm.state)
        return warm_state.position, warm.parameters  # (pytree with leading E,  MCLMCAdaptationState)

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

    def _make_predictor(self, x: ArrayLike) -> BdePredictor:
        if self.positions_eT is None:
            raise RuntimeError("Call 'fit' before requesting predictions.")
        if not getattr(self.bde, "members", None):
            raise RuntimeError("BDE members are not initialized; ensure 'fit' has been executed successfully.")

        return BdePredictor(
            forward_fn=self.bde.members[0].forward,
            positions_eT=self.positions_eT,
            xte=x,
            task=self.task,
        )

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
        """

        Parameters
        ----------
        xte
        mean_and_std
        credible_intervals
        raw
        probabilities

        Returns
        -------

        """
        predictor = self._make_predictor(xte)
        return predictor.predict(
            mean_and_std=mean_and_std,
            credible_intervals=credible_intervals,
            raw=raw,
            probabilities=probabilities,
        )

    def get_raw_predictions(self, x: ArrayLike):
        """Return raw ensemble predictions for a given input batch."""
        return self.evaluate(x, raw=True)["raw"]


# TODO: [@angelos] maybe put them in another file?
class BdeRegressor(Bde, RegressorMixin):
    def __init__(self, **kwargs):
        if "task" in kwargs:
            raise TypeError("'task' cannot be overridden for BdeRegressor; it is fixed to regression.")
        super().__init__(task=TaskType.REGRESSION, **kwargs)

    def predict(self,
                x: ArrayLike,
                mean_and_std: bool = False,
                credible_intervals: list[float] | None = None,
                raw: bool = False):
        out = self.evaluate(
            x,
            mean_and_std=mean_and_std,
            credible_intervals=credible_intervals,
            raw=raw,
        )
        if mean_and_std:
            return out["mean"], out["std"]
        if credible_intervals:
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

        return super().get_raw_predictions(x)


class BdeClassifier(Bde, ClassifierMixin):
    def __init__(self, **kwargs):
        if "task" in kwargs:
            raise TypeError("'task' cannot be overridden for BdeClassifier; it is fixed to classification.")
        super().__init__(task=TaskType.CLASSIFICATION, **kwargs)

    def predict(self, x: ArrayLike):
        return self.evaluate(x)["labels"]

    def predict_proba(self, x: ArrayLike):
        return self.evaluate(x, probabilities=True)["probs"]

    def get_raw_predictions(self, x: ArrayLike):
        """Return raw ensemble predictions.

        Shape: (E, T, N, C), where:
          - E = ensemble members
          - T = posterior samples per member
          - N = number of test points
          - C = number of classes (logits before softmax)
        """
        return super().get_raw_predictions(x)
