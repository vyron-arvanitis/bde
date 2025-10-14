"""High-level scikit-learn style estimators for Bayesian deep ensembles."""

from functools import partial
from typing import TYPE_CHECKING, Any, Protocol, cast, Callable

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.tree_util import tree_leaves, tree_map
from jax.typing import ArrayLike
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_is_fitted

from .bde_builder import BdeBuilder
from .bde_evaluator import BdePredictor
from .data.utils import validate_fit_data, validate_predict_data
from .loss import BaseLoss
from .models import BaseModel
from .sampler.mile_wrapper import MileWrapper
from .sampler.probabilistic import ProbabilisticModel
from .sampler.prior import PriorDist
from .sampler.types import ParamTree
from .sampler.warmup import warmup_bde
from .task import TaskType

if TYPE_CHECKING:
    from sklearn.preprocessing import LabelEncoder


class _WarmupState(Protocol):
    """Protocol for warmup states exposing the current position."""

    position: Any


class Bde:
    """Base estimator that orchestrates training and sampling of deep ensembles.

    The class follows the scikit-learn API and is specialised by regression and
    classification mixins.
    """
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
                 patience: int | None = None,
                 n_samples: int = 10,
                 warmup_steps: int = 50,
                 lr: float = 1e-3,
                 n_thinning: int = 2,
                 desired_energy_var_start: float = 0.5,
                 desired_energy_var_end: float = 0.1,
                 step_size_init: float = None
                 ):
        """Initialise the estimator with architectural and sampling settings.

        Parameters
        ----------
        n_members : int
            Number of networks in the deep ensemble.
        hidden_layers : list[int] | None
            Hidden-layer widths; defaults to `[4, 4]` when `None`.
        seed : int
            Random seed shared across ensemble and sampling routines.
        task : TaskType | None
            Task identifier; subclasses set this automatically.
        loss : BaseLoss | None
            Optional user-provided loss function.
        activation : str
            Activation function passed to each network.
        epochs : int
            Maximum training epochs for each ensemble member.
        patience : int
            Early-stopping patience forwarded to the builder.
        n_samples : int
            Number of posterior samples per ensemble member.
        warmup_steps : int
            Warmup iterations for the sampler.
        lr : float
            Learning rate for the default Adam optimizer.
        n_thinning : int
            Thinning factor applied to MCMC samples.
        desired_energy_var_start : float
            Initial target energy variance for warmup.
        desired_energy_var_end : float
            Final target energy variance for warmup.
        step_size_init : float | None
            Optional initial step size for the sampler.
        """

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
        """Instantiate the builder and ensemble members based on current settings."""

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
        """Construct the log-posterior callable for the ensemble.

        Parameters
        ----------
        x : ArrayLike
            Feature matrix that the posterior conditions on.
        y : ArrayLike
            Targets associated with `x`.

        Returns
        -------
        Callable
            Function mapping parameter states to their unnormalized log posterior.
        """
        prior = PriorDist.STANDARDNORMAL.get_prior()
        proto = self._bde.members[0]
        pm = ProbabilisticModel(model=proto, params=proto.params, prior=prior, task=self.task)
        return partial(pm.log_unnormalized_posterior, x=x, y=y)

    def _warmup_sampler(self, logpost):
        """Run the adaptive warmup phase for the MCMC sampler.

        Parameters
        ----------
        logpost : Callable
            Unnormalised log posterior accepting parameter trees.

        Returns
        -------
        tuple[ParamTree, Any]
            Warmed-up starting positions and adaptation metadata.
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
        """Construct distinct PRNG keys for each MCMC chain.

        Parameters
        ----------
        num_chains : int
            Number of independent chains required.

        Returns
        -------
        jax.Array
            Array of shape (num_chains, 2) containing PRNG keys.
        """

        rng = jax.random.PRNGKey(int(self.seed))
        return jax.vmap(lambda i: jax.random.fold_in(rng, i))(jnp.arange(num_chains))

    @staticmethod
    def _normalize_tuned_parameters(tuned, num_chains: int):
        """Broadcast tuned sampler parameters to the number of chains.

        Parameters
        ----------
        tuned : Any
            Object returned by warmup containing `L`, `step_size`, `sqrt_diag_cov`.
        num_chains : int
            Number of chains requested for sampling.

        Returns
        -------
        tuple[jax.Array, jax.Array, jax.Array]
            Tuple of inverse mass matrix diagonals, step sizes, and covariance square roots.
        """

        L_e = tuned.L if jnp.ndim(tuned.L) == 1 else jnp.full((num_chains,), tuned.L)
        step_e = tuned.step_size if jnp.ndim(tuned.step_size) == 1 else jnp.full((num_chains,), tuned.step_size)
        sqrt_diag_e = tuned.sqrt_diag_cov
        return L_e, step_e, sqrt_diag_e

    def _draw_samples(self,
                      logpost: Callable,
                      rng_keys_e: ArrayLike,
                      init_positions_e,
                      L_e,
                      step_e,
                      sqrt_diag_e,
                      ):
        """Generate posterior samples for each ensemble member.

        Parameters
        ----------
        logpost : Callable
            Log posterior callable produced by `_build_log_post`.
        rng_keys_e : jax.Array
            PRNG keys for each chain.
        init_positions_e : ParamTree
            Warmed-up starting positions with leading ensemble axis.
        L_e : jax.Array
            Inverse mass matrix factors per chain.
        step_e : jax.Array
            Step sizes per chain.
        sqrt_diag_e : jax.Array
            Square root of diagonal covariance estimates.

        Returns
        -------
        ParamTree
            Posterior samples with leading axes (ensemble, samples, ...).
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
        """Create a predictor helper configured for the provided features.

        Parameters
        ----------
        x : ArrayLike
            Feature matrix of shape (n_samples, n_features) to evaluate.

        Returns
        -------
        BdePredictor
            Lightweight wrapper exposing ensemble prediction utilities.

        Raises
        ------
        NotFittedError
            If `fit` has not been called yet.
        RuntimeError
            If the underlying builder did not initialize ensemble members.
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
        """Prepare target values before training.

        Parameters
        ----------
        y_checked : ArrayLike
            Target array validated by `validate_fit_data`.

        Returns
        -------
        ArrayLike
            Possibly transformed targets. The base implementation is identity and
            subclasses can override it (e.g. to one-hot encode classification labels).
        """

        return y_checked

    def _fit(self, x: ArrayLike, y: ArrayLike):
        """Fit the Bayesian Deep Ensemble on the provided dataset.

        Parameters
        ----------
        x : ArrayLike
            Feature matrix of shape (n_samples, n_features).
        y : ArrayLike
            Target array with shape compatible to the configured task.

        Returns
        -------
        Bde
            The fitted estimator.
        """

        x_np, y_np = validate_fit_data(self, x, y)  # x_np: (N, D), y_np: (N, 1) for regression
        y_prepared = self._prepare_targets(y_np)  # preserve (N, 1) for regression targets

        x_checked = jnp.asarray(x_np)  # (N, D)
        y_checked = jnp.asarray(y_prepared)  # regression: (N, 1); classification: (N,)

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
        """This method validates the tags parsed into the evaluate method, makes sure no two competing tags are given.

        Parameters
        ----------
        mean_and_std : bool
            When `True`, return both the predictive mean and standard deviation.
        credible_intervals : list[float] | None
            Credible interval levels in (0, 1). In regression mode mutually exclusive
            with `mean_and_std`.
        raw : bool
            Return the raw ensemble outputs without aggregation.
        probabilities : bool
            Return class probabilities (classification only).
        """

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

    def _evaluate(
            self,
            xte: ArrayLike,
            mean_and_std: bool = False,
            credible_intervals: list[float] | None = None,
            raw: bool = False,
            probabilities: bool = False,
    ):
        """Evaluate the fitted ensemble under different output modes.

        Parameters
        ----------
        xte : ArrayLike
            Feature matrix for which predictions are requested.
        mean_and_std : bool
            When `True`, return both the predictive mean and standard deviation.
        credible_intervals : list[float] | None
            Credible interval levels in (0, 1). In regression mode mutually exclusive
            with `mean_and_std`.
        raw : bool
            Return the raw ensemble outputs without aggregation.
        probabilities : bool
            Return class probabilities (classification only).

        Returns
        -------
        dict[str, ArrayLike]
            Mapping containing the requested prediction artifacts.
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


class BdeRegressor(Bde, BaseEstimator, RegressorMixin):
    """Regression-friendly wrapper exposing scikit-learn style API.
    """

    def __init__(
            self,
            n_members: int = 2,
            hidden_layers: list[int] | None = None,
            seed: int = 0,
            loss: BaseLoss | None = None,
            activation: str = "relu",
            epochs: int = 20,
            patience: int | None = None,
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

    def fit(self, x: ArrayLike, y: ArrayLike):
        return super()._fit(x, y)

    def predict(self,
                x: ArrayLike,
                mean_and_std: bool = False,
                credible_intervals: list[float] | None = None,
                raw: bool = False):
        out = self._evaluate(
            x,
            mean_and_std=mean_and_std,
            credible_intervals=credible_intervals,
            raw=raw,
        )
        if raw:
            return out["raw"]
        if mean_and_std:
            return out["mean"], out["std"]  # both (N,) for regression
        if credible_intervals:
            return out["mean"], out["credible_intervals"]  # mean (N,), intervals (Q, N)
        return out["mean"]  # (N,) regression predictive mean


class BdeClassifier(Bde, BaseEstimator, ClassifierMixin):
    """Classification wrapper with label encoding helpers.
    """

    label_encoder_: "LabelEncoder"

    def __init__(
            self,
            n_members: int = 2,
            hidden_layers: list[int] | None = None,
            seed: int = 0,
            loss: BaseLoss | None = None,
            activation: str = "relu",
            epochs: int = 20,
            patience: int | None = None,
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

    def fit(self, x: ArrayLike, y: ArrayLike):
        return super()._fit(x, y)

    def predict(self, x: ArrayLike, raw: bool = False):
        if raw:
            return self._evaluate(x, raw=True)["raw"]
        labels = np.asarray(self._evaluate(x)["labels"])
        if not hasattr(self, "label_encoder_"):
            return labels
        return self.label_encoder_.inverse_transform(labels.astype(int))

    def predict_proba(self, x: ArrayLike):
        probs = np.asarray(self._evaluate(x, probabilities=True)["probs"])
        return probs
