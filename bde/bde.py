from bde.bde_builder import BdeBuilder
from bde.bde_evaluator import BdePredictor
from bde.loss.loss import BaseLoss
from bde.models.models import BaseModel
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
from sklearn.utils.validation import check_is_fitted
from typing import Any, Protocol, TYPE_CHECKING, cast
import numpy as np

if TYPE_CHECKING:
    from sklearn.preprocessing import LabelEncoder

from bde.data.utils import (
    validate_fit_data,
    validate_predict_data,
)
from bde.sampler.my_types import ParamTree


class _WarmupState(Protocol):
    """Protocol for warmup states exposing the current position."""

    position: Any


class Bde:
    positions_eT_: ParamTree
    is_fitted_: bool
    members_: list[BaseModel]
    hidden_layers: list[int] | None
    step_size_init: float | None

    def __init__(self,
                 n_members: int = 2,
                 hidden_layers: list[int] | None = None,
                 seed: int = 0,
                 task: TaskType = None,
                 loss: BaseLoss = None,
                 activation: str = "relu",
                 epochs: int = 20,
                 patience: int = 25,
                 n_samples: int = 10,
                 warmup_steps: int = 50,
                 lr: float = 1e-3,
                 n_thinning: int = 2,
                 desired_energy_var_start: float = 0.5,
                 desired_energy_var_end: float = 0.1,
                 step_size_init: float = None
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
        self.desired_energy_var_start = desired_energy_var_start
        self.desired_energy_var_end = desired_energy_var_end
        self.step_size_init = step_size_init

        # Internal caches resolved during fit
        self._resolved_hidden_layers = None
        self._resolved_step_size_init = None
        self._bde: BdeBuilder | None = None

    def _build_bde(self):
        """

        Returns
        -------

        """
        hidden_layers = self.hidden_layers if self.hidden_layers is not None else [4, 4]
        self._resolved_hidden_layers = list(hidden_layers)

        self._bde = BdeBuilder(
            hidden_sizes=self._resolved_hidden_layers,
            patience=self.patience,
            n_members=self.n_members,
            task=self.task,
            seed=self.seed,
            act_fn=self.activation
        )

        self.members_ = self._bde.members

    def _build_log_post(self, x: ArrayLike, y: ArrayLike):
        """

        Parameters
        ----------
        x
        y

        Returns
        -------

        """
        prior = PriorDist.STANDARDNORMAL.get_prior()
        proto = self._bde.members[0]
        pm = ProbabilisticModel(module=proto, params=proto.params, prior=prior, task=self.task)
        return partial(pm.log_unnormalized_posterior, x=x, y=y)

    def _warmup_sampler(self, logpost):
        """

        Parameters
        ----------
        logpost

        Returns
        -------

        """
        warm = warmup_bde(
            self._bde,
            logpost,
            step_size_init=self._resolved_step_size_init,
            warmup_steps=self.warmup_steps,
            desired_energy_var_start=self.desired_energy_var_start,
            desired_energy_var_end=self.desired_energy_var_end,
        )
        warm_state = cast(_WarmupState, warm.state)
        return warm_state.position, warm.parameters  # (pytree with leading E,  MCLMCAdaptationState)

    def _generate_rng_keys(self, num_chains: int):
        """

        Parameters
        ----------
        num_chains

        Returns
        -------

        """
        rng = jax.random.PRNGKey(int(self.seed))
        return jax.vmap(lambda i: jax.random.fold_in(rng, i))(jnp.arange(num_chains))

    @staticmethod
    def _normalize_tuned_parameters(tuned, num_chains: int):
        """

        Parameters
        ----------
        tuned
        num_chains

        Returns
        -------

        """
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
        """

        Parameters
        ----------
        logpost
        rng_keys_e
        init_positions_e
        L_e
        step_e
        sqrt_diag_e

        Returns
        -------

        """
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
        """

        Parameters
        ----------
        x

        Returns
        -------

        """

        check_is_fitted(self)
        if not getattr(self._bde, "members", None):
            raise RuntimeError("BDE members are not initialized; ensure 'fit' has been executed successfully.")

        return BdePredictor(
            forward_fn=self._bde.members[0].forward,
            positions_eT=self.positions_eT_,
            xte=x,
            task=self.task,
        )

    @staticmethod
    def _prepare_targets(y_checked: ArrayLike) -> ArrayLike:
        """This method is to be overwritten in the BdeClassifier class in order to prepare the labels for classification

        Parameters
        ----------
        y_checked

        Returns
        -------

        """

        return y_checked

    def fit(self, x: ArrayLike, y: ArrayLike):
        """

        Parameters
        ----------
        x
        y

        Returns
        -------

        """

        x_np, y_np = validate_fit_data(self, x, y)
        y_prepared = self._prepare_targets(y_np)

        x_checked = jnp.asarray(x_np)
        y_checked = jnp.asarray(y_prepared)

        self._resolved_step_size_init = (
            self.step_size_init if self.step_size_init is not None else self.lr
        )

        self._build_bde()
        self._bde.fit_members(
            x=x_checked,
            y=y_checked,
            optimizer=optax.adam(self.lr),
            epochs=self.epochs,
            loss=self.loss,
        )

        logpost_one = self._build_log_post(x_checked, y_checked)
        init_positions_e, tuned = self._warmup_sampler(logpost_one)

        num_chains = tree_leaves(init_positions_e)[0].shape[0]
        rng_keys_e = self._generate_rng_keys(num_chains)
        L_e, step_e, sqrt_diag_e = self._normalize_tuned_parameters(tuned, num_chains)

        samples = self._draw_samples(
            logpost_one,
            rng_keys_e,
            init_positions_e,
            L_e,
            step_e,
            sqrt_diag_e,
        )
        self.positions_eT_ = tree_map(lambda arr: jnp.asarray(arr), samples)

        self.is_fitted_ = True

        return self

    # scikit-learn compatibility tags
    def _more_tags(self):
        return {
            "poor_score": True,  # training can be stochastic and heavy
            "multioutput": False,
        }

    def _validate_evaluate_tags(
            self,
            *,
            mean_and_std: bool = False,
            credible_intervals: list[float] | None = None,
            raw: bool = False,
            probabilities: bool = False,
    ):
        if self.task == TaskType.REGRESSION:
            if probabilities:
                raise ValueError("'probabilities' predictions are only supported for classification tasks.")
            if mean_and_std and credible_intervals:
                raise ValueError("'mean_and_std' and 'credible_intervals' cannot be requested together.")
            if raw and credible_intervals:
                raise ValueError("'raw' and 'credible_intervals' cannot be requested together.")
            if raw and mean_and_std:
                raise ValueError("'raw' and 'mean_and_std' cannot be requested together.")
            return

        if self.task == TaskType.CLASSIFICATION:
            if mean_and_std:
                raise ValueError("'mean_and_std' predictions are not available for classification tasks.")
            if credible_intervals:
                raise ValueError("'credible_intervals' predictions are not available for classification tasks.")
            return

        raise ValueError(f"Unsupported task type {self.task}")

    def evaluate(
            self,
            xte: ArrayLike,
            mean_and_std: bool = False,
            credible_intervals: list[float] | None = None,
            raw: bool = False,
            probabilities: bool = False,
    ):
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
        xte_np = validate_predict_data(self, xte)
        xte_jnp = jnp.asarray(xte_np)
        predictor = self._make_predictor(xte_jnp)
        self._validate_evaluate_tags(
            mean_and_std=mean_and_std,
            credible_intervals=credible_intervals,
            raw=raw,
            probabilities=probabilities,
        )
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
class BdeRegressor(Bde, BaseEstimator, RegressorMixin):
    def __init__(
            self,
            n_members: int = 2,
            hidden_layers: list[int] | None = None,
            seed: int = 0,
            loss: BaseLoss | None = None,
            activation: str = "relu",
            epochs: int = 20,
            patience: int = 25,
            n_samples: int = 10,
            warmup_steps: int = 50,
            lr: float = 1e-3,
            n_thinning: int = 2,
            desired_energy_var_start: float = 0.5,
            desired_energy_var_end: float = 0.1,
            step_size_init: float | None = None,
    ):
        super().__init__(
            n_members=n_members,
            hidden_layers=hidden_layers,
            seed=seed,
            task=TaskType.REGRESSION,
            loss=loss,
            activation=activation,
            epochs=epochs,
            patience=patience,
            n_samples=n_samples,
            warmup_steps=warmup_steps,
            lr=lr,
            n_thinning=n_thinning,
            desired_energy_var_start=desired_energy_var_start,
            desired_energy_var_end=desired_energy_var_end,
            step_size_init=step_size_init,
        )

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


class BdeClassifier(Bde, BaseEstimator, ClassifierMixin):
    label_encoder_: "LabelEncoder"

    def __init__(
            self,
            n_members: int = 2,
            hidden_layers: list[int] | None = None,
            seed: int = 0,
            loss: BaseLoss | None = None,
            activation: str = "relu",
            epochs: int = 20,
            patience: int = 25,
            n_samples: int = 10,
            warmup_steps: int = 50,
            lr: float = 1e-3,
            n_thinning: int = 2,
            desired_energy_var_start: float = 0.5,
            desired_energy_var_end: float = 0.1,
            step_size_init: float | None = None,
    ):
        super().__init__(
            n_members=n_members,
            hidden_layers=hidden_layers,
            seed=seed,
            task=TaskType.CLASSIFICATION,
            loss=loss,
            activation=activation,
            epochs=epochs,
            patience=patience,
            n_samples=n_samples,
            warmup_steps=warmup_steps,
            lr=lr,
            n_thinning=n_thinning,
            desired_energy_var_start=desired_energy_var_start,
            desired_energy_var_end=desired_energy_var_end,
            step_size_init=step_size_init,
        )

    def _prepare_targets(self, y_checked):
        from sklearn.preprocessing import LabelEncoder

        encoder = LabelEncoder()
        encoded = encoder.fit_transform(np.asarray(y_checked))
        self.label_encoder_ = encoder
        self.classes_ = np.asarray(encoder.classes_)
        return encoded.astype(np.int32)

    def predict(self, x: ArrayLike):
        labels = np.asarray(self.evaluate(x)["labels"])
        if not hasattr(self, "label_encoder_"):
            return labels
        return self.label_encoder_.inverse_transform(labels.astype(int))

    def predict_proba(self, x: ArrayLike):
        probs = np.asarray(self.evaluate(x, probabilities=True)["probs"])
        return probs

    def get_raw_predictions(self, x: ArrayLike):
        """Return raw ensemble predictions.

        Shape: (E, T, N, C), where:
          - E = ensemble members
          - T = posterior samples per member
          - N = number of test points
          - C = number of classes (logits before softmax)
        """
        return super().get_raw_predictions(x)
