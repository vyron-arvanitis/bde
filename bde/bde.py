"""High-level scikit-learn style estimators for Bayesian deep ensembles."""

from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Protocol, cast

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.tree_util import tree_leaves, tree_map
from jax.typing import ArrayLike
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils._tags import (
    ClassifierTags,
    InputTags,
    RegressorTags,
    Tags,
    TargetTags,
)
from sklearn.utils.validation import check_is_fitted

from .bde_builder import BdeBuilder
from .bde_evaluator import BdePredictor
from .data.utils import validate_fit_data, validate_predict_data
from .loss import BaseLoss
from .models import BaseModel
from .sampler.mile_wrapper import MileWrapper
from .sampler.prior import Prior, PriorDist
from .sampler.probabilistic import ProbabilisticModel
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

    def __init__(
        self,
        n_members: int = 2,
        hidden_layers: list[int] | None = None,
        seed: int = 0,
        task: TaskType = None,
        loss: BaseLoss = None,
        activation: str = "relu",
        epochs: int = 20,
        patience: int | None = None,
        validation_split: float = 0.15,
        n_samples: int = 10,
        warmup_steps: int = 50,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        n_thinning: int = 2,
        desired_energy_var_start: float = 0.5,
        desired_energy_var_end: float = 0.1,
        step_size_init: float = None,
        prior_family: str | PriorDist = "standardnormal",
        prior_kwargs: dict[str, Any] | None = None,
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
            Warmup iterations for the sampler. When ``0`` or ``None``, warmup is skipped
            and sampling starts from the trained parameters.
        lr : float
            Learning rate for the default Adam optimizer.
        validation_split : float | None
            Fraction of data reserved for validation when early stopping is used. Must
            lie in (0, 1) if provided; when ``None``, all data is used for training and
            early stopping is disabled.
        weight_decay: float
            Weight decay parameter for the AdamW optimizer applied during member
            pretraining.
        n_thinning : int
            Thinning factor applied to MCMC samples.
        desired_energy_var_start : float
            Initial target energy variance for warmup.
        desired_energy_var_end : float
            Final target energy variance for warmup.
        step_size_init : float | None
            Optional initial step size for the sampler.
        prior_family : str or PriorDist
            Prior distribution for network weights; accepts a ``PriorDist`` enum
            or a string key (case-insensitive). Defaults to ``\"standardnormal\"``.
        prior_kwargs : dict[str, Any] | None
            Optional keyword arguments forwarded to the chosen `prior_family`
            (e.g. ``{"scale": 0.1}`` for a wider or narrower Normal or Laplace).
        """

        self.n_members = n_members
        self.hidden_layers = hidden_layers
        self.seed = seed
        self.task = task
        self.loss = loss
        self.activation = activation
        self.epochs = epochs
        self.patience = patience
        self.validation_split = validation_split
        self.n_samples = n_samples
        self.warmup_steps = warmup_steps
        self.lr = lr
        self.weight_decay = weight_decay
        self.n_thinning = n_thinning
        self.desired_energy_var_start = desired_energy_var_start
        self.desired_energy_var_end = desired_energy_var_end
        self.step_size_init = step_size_init
        self.prior_family = prior_family
        self.prior_kwargs = prior_kwargs

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
            val_split=self.validation_split,
            n_members=self.n_members,
            lr=self.lr,
            weight_decay=self.weight_decay,
            task=self.task,
            seed=self.seed,
            act_fn=self.activation,
        )

        self.members_ = self._bde.members

    def _resolve_prior_family(self) -> Prior | PriorDist:
        """Normalise the prior specification to a PriorDist or Prior instance."""

        pf = self.prior_family
        if isinstance(pf, Prior):
            return pf
        if isinstance(pf, PriorDist):
            return pf
        if isinstance(pf, str):
            try:
                return PriorDist[pf.upper()]
            except KeyError as exc:
                valid = ", ".join([p.name.lower() for p in PriorDist])
                raise ValueError(
                    f"Unknown prior_family '{pf}'. Valid options: {valid}."
                ) from exc
        raise ValueError(
            f"Unsupported prior_family type {type(pf).__name__}; expected str,"
            " PriorDist or Prior."
        )

    def _build_log_post(
        self,
        x: ArrayLike,
        y: ArrayLike,
        prior_family: Prior | PriorDist,
        prior_kwargs=None,
    ) -> Callable[[ParamTree], ArrayLike]:
        """Construct the log-posterior callable for the ensemble.

        Parameters
        ----------
        x : ArrayLike
            Feature matrix that the posterior conditions on.
        y : ArrayLike
            Targets associated with `x`.

        Returns
        -------
        Callable[[ParamTree], ArrayLike]
            Function mapping parameter states to their unnormalized log posterior.
        """

        prior_obj = (
            prior_family
            if isinstance(prior_family, Prior)
            else prior_family.get_prior(**(prior_kwargs or {}))
        )

        proto = self._bde.members[0]
        pm = ProbabilisticModel(
            model=proto, params=proto.params, prior=prior_obj, task=self.task
        )
        return partial(pm.log_unnormalized_posterior, x=x, y=y)

    def _warmup_sampler(self, logpost: Callable[[ParamTree], ArrayLike]):
        """Run the adaptive warmup phase for the MCMC sampler.

        Parameters
        ----------
        logpost : Callable[[ParamTree], ArrayLike]
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
        return (
            warm_state.position,
            warm.parameters,
        )  # (pytree with leading E,  MCLMCAdaptationState)

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
            Tuple of inverse mass matrix diagonals,
            step sizes and covariance square roots.
        """

        L_e = tuned.L if jnp.ndim(tuned.L) == 1 else jnp.full((num_chains,), tuned.L)
        step_e = (
            tuned.step_size
            if jnp.ndim(tuned.step_size) == 1
            else jnp.full((num_chains,), tuned.step_size)
        )
        sqrt_diag_e = tuned.sqrt_diag_cov
        return L_e, step_e, sqrt_diag_e

    def _draw_samples(
        self,
        logpost: Callable[[ParamTree], ArrayLike],
        rng_keys_e: ArrayLike,
        init_positions_e: ArrayLike,
        L_e: ArrayLike,
        step_e: ArrayLike,
        sqrt_diag_e: ArrayLike,
    ):
        """Generate posterior samples for each ensemble member.

        Parameters
        ----------
        logpost : Callable[[ParamTree], ArrayLike]
            Log posterior callable produced by `_build_log_post`.
        rng_keys_e : jax.Array
            PRNG keys for each chain.
        init_positions_e : ParamTree
            Warmed-up starting positions with leading ensemble axis.
        L_e : ArrayLike
            Inverse mass matrix factors per chain.
        step_e : ArrayLike
            Step sizes per chain.
        sqrt_diag_e : ArrayLike
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
            raise RuntimeError(
                "BDE members are not initialized; ensure 'fit' has been executed"
                " successfully."
            )

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

        x_np, y_np = validate_fit_data(
            self, x, y
        )  # x_np: (N, D), y_np: (N, 1) for regression
        self.n_features_in_ = x_np.shape[1]
        y_prepared = self._prepare_targets(
            y_np
        )  # preserve (N, 1) for regression targets
        loss_obj = self.loss if isinstance(self.loss, BaseLoss) else None
        if self.task is not None and loss_obj is not None:
            self.task.validate_loss(loss_obj)
        resolved_prior_family = self._resolve_prior_family()

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
            loss=loss_obj,
        )

        logpost_one = self._build_log_post(
            x_checked, y_checked, resolved_prior_family, self.prior_kwargs
        )
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
    def __sklearn_tags__(self) -> Tags:
        tags = super().__sklearn_tags__()
        tags._skip_test = False
        tags.no_validation = False
        tags.requires_fit = True
        tags.input_tags = InputTags(
            one_d_array=False,
            two_d_array=True,
            three_d_array=False,
            sparse=False,
            categorical=False,
            string=False,
            dict=False,
            positive_only=False,
            allow_nan=False,
            pairwise=False,
        )
        tags.target_tags = TargetTags(
            required=getattr(tags.target_tags, "required", False),
            one_d_labels=False,
            two_d_labels=False,
            positive_only=False,
            multi_output=False,
            single_output=True,
        )
        return tags

    def _validate_evaluate_tags(
        self,
        *,
        mean_and_std: bool = False,
        credible_intervals: list[float] | None = None,
        raw: bool = False,
        probabilities: bool = False,
    ):
        """This method validates the tags parsed into the evaluate method,
        makes sure no two competing tags are given.

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
                raise ValueError(
                    "'probabilities' predictions are only supported for classification"
                    " tasks."
                )
            if mean_and_std and credible_intervals:
                raise ValueError(
                    "'mean_and_std' and 'credible_intervals' cannot be requested"
                    " together."
                )
            if raw and credible_intervals:
                raise ValueError(
                    "'raw' and 'credible_intervals' cannot be requested together."
                )
            if raw and mean_and_std:
                raise ValueError(
                    "'raw' and 'mean_and_std' cannot be requested together."
                )
            return

        if self.task == TaskType.CLASSIFICATION:
            if mean_and_std:
                raise ValueError(
                    "'mean_and_std' predictions are not available for classification"
                    " tasks."
                )
            if credible_intervals:
                raise ValueError(
                    "'credible_intervals' predictions are not available for"
                    " classification tasks."
                )
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

    def history(self):
        """Return training history from the builder.

        Returns
        -------
        dict[str, dict[str, list[float]]]
            Dictionary mapping member names to their training and validation loss
            histories.
        """

        if self._bde is None:
            raise RuntimeError(
                "The BDE builder is not initialized; ensure 'fit' has been executed"
                " successfully."
            )
        return self._bde.history


class BdeRegressor(Bde, RegressorMixin, BaseEstimator):
    """Regression-friendly wrapper exposing scikit-learn style API."""

    def __init__(
        self,
        n_members: int = 2,
        hidden_layers: list[int] | None = None,
        seed: int = 0,
        loss: BaseLoss | None = None,
        activation: str = "relu",
        epochs: int = 20,
        patience: int | None = None,
        validation_split: float = 0.15,
        n_samples: int = 10,
        warmup_steps: int = 50,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        n_thinning: int = 2,
        desired_energy_var_start: float = 0.5,
        desired_energy_var_end: float = 0.1,
        step_size_init: float | None = None,
        prior_family: str | PriorDist = "standardnormal",
        prior_kwargs: dict[str, Any] | None = None,
    ):
        """Initialise the regressor with architecture,
        optimisation, and sampling settings.

        Parameters
        ----------
        n_members : int, default=2
            Number of deterministic networks in the ensemble.
        hidden_layers : list[int] | None, default= None
            Hidden layer widths; defaults to ``[4, 4]`` internally when ``None``.
        seed : int, default=0
            Shared PRNG seed for member initialisation and sampling.
        loss : BaseLoss | None
            Custom training loss; defaults to :class:`bde.loss.GaussianNLL`.
        activation : str, default='relu'
            Activation function applied to each hidden layer.
        epochs : int, default=20
            Maximum training epochs during the deterministic phase.
        patience : int | None, optional
            Early-stopping patience measured in epochs; ``None`` disables it.
        validation_split : float, default=0.15
            Fraction of data reserved for validation when early stopping is enabled.
            Must lie in (0, 1); when ``None``, all data is used and early stopping is
            skipped.
        n_samples : int, default=10
            Posterior samples retained for each ensemble member.
        warmup_steps : int, default=50
            Number of warm-up iterations for the MCMC sampler.
        lr : float, default=1e-3
            Learning rate for the Adam optimiser used in pre-sampling training.
        weight_decay: float
            Weight decay parameter for the AdamW optimiser applied during member
            training.
        n_thinning : int, default=2
            Thinning interval applied to posterior samples.
        desired_energy_var_start : float, default=0.5
            Target energy variance at the start of warm-up.
        desired_energy_var_end : float, default=0.1
            Target energy variance at the end of warm-up.
        step_size_init : float | None, optional
            Override for the sampler's initial step size; falls back to ``lr``.
        prior_family : str or PriorDist
            Prior distribution for network weights; accepts a ``PriorDist`` enum
            or string key. Defaults to ``\"standardnormal\"``.
        prior_kwargs : dict[str, Any] | None
            Optional keyword arguments forwarded to the chosen `prior_family`
            (e.g. ``{"scale": 0.1}`` for a wider or narrower Normal or Laplace).
        """

        super().__init__(
            n_members=n_members,
            hidden_layers=hidden_layers,
            seed=seed,
            task=TaskType.REGRESSION,
            loss=loss,
            activation=activation,
            epochs=epochs,
            patience=patience,
            validation_split=validation_split,
            n_samples=n_samples,
            warmup_steps=warmup_steps,
            lr=lr,
            weight_decay=weight_decay,
            n_thinning=n_thinning,
            desired_energy_var_start=desired_energy_var_start,
            desired_energy_var_end=desired_energy_var_end,
            step_size_init=step_size_init,
            prior_family=prior_family,
            prior_kwargs=prior_kwargs,
        )
        self._estimator_type = "regressor"

    def __sklearn_tags__(self) -> Tags:
        base = super().__sklearn_tags__()
        base.estimator_type = "regressor"
        base.target_tags = TargetTags(
            required=True,
            one_d_labels=False,
            two_d_labels=False,
            positive_only=False,
            multi_output=False,
            single_output=True,
        )
        base.regressor_tags = RegressorTags(poor_score=True)
        return base

    def fit(self, x: ArrayLike, y: ArrayLike):
        """Fit the regression ensemble on the provided dataset.

        Parameters
        ----------
        x : ArrayLike
            Feature matrix of shape (n_samples, n_features).
        y : ArrayLike
            Continuous targets shaped (n_samples,) or (n_samples, 1).

        Returns
        -------
        BdeRegressor
            The fitted estimator instance.
        """

        return super()._fit(x, y)

    def predict(
        self,
        x: ArrayLike,
        mean_and_std: bool = False,
        credible_intervals: list[float] | None = None,
        # Docstring necessary to explain this parameter which
        # actually lists quantiles not the intervals
        raw: bool = False,
    ):
        """Predict regression targets with optional uncertainty summaries.

        Parameters
        ----------
        x : ArrayLike
            Feature matrix of shape (n_samples, n_features).
        mean_and_std : bool, default=False
            When ``True``, return a tuple ``(mean, std)`` where both arrays
            have shape ``(n_samples,)`` and ``std`` combines aleatoric and
            epistemic uncertainty.
        credible_intervals : list[float] | None, default=None
            Quantile levels in (0, 1) used to summarise the predictive
            distribution. When not ``None`` and ``mean_and_std`` is ``False``, the
            method returns ``(mean, quantiles)`` where ``mean`` has shape
            ``(n_samples,)`` and ``quantiles`` has shape ``(Q, n_samples)`` with
            one row per requested level. Note that the values are quantiles,
            not closed intervals; for an 80% interval you would typically pass
            ``[0.1, 0.9]`` and interpret the two returned quantile curves as
            lower and upper bounds.
        raw : bool, default=False
            When ``True``, ignore other flags and return the raw ensemble
            outputs with shape ``(n_members, n_samples_draws, n_samples, 2)``,
            corresponding to per-member, per-draw mean and scale parameters.

        Note
        ----
            The regression likelihood is Gaussian with mean given by the first
            head and scale given by the second head passed through ``softplus``
            (plus a small epsilon) during training, sampling, and evaluation;


        Returns
        -------
        ArrayLike | tuple[jax.Array, jax.Array]
            - If ``raw`` is ``True``, the raw tensor described above.
            - If ``mean_and_std`` is ``True``, a tuple ``(mean, std)``.
            - If ``credible_intervals`` is provided, a tuple
              ``(mean, quantiles)``.
            - Otherwise, the predictive mean ``(n_samples,)``.
        """

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


class BdeClassifier(Bde, ClassifierMixin, BaseEstimator):
    """Classification wrapper with label encoding helpers."""

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
        validation_split: float = 0.15,
        n_samples: int = 10,
        warmup_steps: int = 50,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        n_thinning: int = 2,
        desired_energy_var_start: float = 0.5,
        desired_energy_var_end: float = 0.1,
        step_size_init: float | None = None,
        prior_family: str | PriorDist = "standardnormal",
        prior_kwargs: dict[str, Any] | None = None,
    ):
        """Initialise the classifier with architecture,
        optimisation, and sampling settings.

        Parameters
        ----------
        n_members : int, default=2
            Number of deterministic networks in the ensemble.
        hidden_layers : list[int] | None, default= None
            Hidden layer widths; defaults to ``[4, 4]`` internally when ``None``.
        seed : int, default=0
            Shared PRNG seed for member initialisation and sampling.
        loss : BaseLoss | None
            Custom training loss; defaults to :class:`bde.loss.GaussianNLL`.
        activation : str, default='relu'
            Activation function applied to each hidden layer.
        epochs : int, default=20
            Maximum training epochs during the deterministic phase.
        patience : int | None, optional
            Early-stopping patience measured in epochs; ``None`` disables it.
        validation_split : float, default=0.15
            Fraction of data reserved for validation when early stopping is enabled.
            Must lie in (0, 1); when ``None``, all data is used and early stopping is
            skipped.
        n_samples : int, default=10
            Posterior samples retained for each ensemble member.
        warmup_steps : int, default=50
            Number of warm-up iterations for the MCMC sampler.
        lr : float, default=1e-3
            Learning rate for the Adam optimiser used in pre-sampling training.
        weight_decay: float
            Weight decay parameter for the AdamW optimiser applied during member
            training.
        n_thinning : int, default=2
            Thinning interval applied to posterior samples.
        desired_energy_var_start : float, default=0.5
            Target energy variance at the start of warm-up.
        desired_energy_var_end : float, default=0.1
            Target energy variance at the end of warm-up.
        step_size_init : float | None, optional
            Override for the sampler's initial step size; falls back to ``lr``.
        prior_family : str or PriorDist
            Prior distribution for network weights; accepts a ``PriorDist`` enum
            or string key. Defaults to ``\"standardnormal\"``.
        prior_kwargs : dict[str, Any] | None
            Optional keyword arguments forwarded to the chosen `prior_family`
            (e.g. ``{"scale": 0.1}`` for a wider or narrower Normal or Laplace).
        """

        super().__init__(
            n_members=n_members,
            hidden_layers=hidden_layers,
            seed=seed,
            task=TaskType.CLASSIFICATION,
            loss=loss,
            activation=activation,
            epochs=epochs,
            patience=patience,
            validation_split=validation_split,
            n_samples=n_samples,
            warmup_steps=warmup_steps,
            lr=lr,
            weight_decay=weight_decay,
            n_thinning=n_thinning,
            desired_energy_var_start=desired_energy_var_start,
            desired_energy_var_end=desired_energy_var_end,
            step_size_init=step_size_init,
            prior_family=prior_family,
            prior_kwargs=prior_kwargs,
        )
        self._estimator_type = "classifier"

    def _prepare_targets(self, y_checked):
        """Encode class labels and cache the label encoder.

        Parameters
        ----------
        y_checked : ArrayLike
            Target labels validated by ``validate_fit_data``.

        Returns
        -------
        numpy.ndarray
            Integer-encoded labels suitable for training, with the fitted
            encoder stored on the estimator as ``label_encoder_`` and the
            decoded class labels exposed via ``classes_``.
        """

        from sklearn.preprocessing import LabelEncoder

        encoder = LabelEncoder()
        encoded = encoder.fit_transform(np.asarray(y_checked))
        self.label_encoder_ = encoder
        self.classes_ = np.asarray(encoder.classes_)
        return encoded.astype(np.int32)

    def fit(self, x: ArrayLike, y: ArrayLike):
        """Fit the classification ensemble on the provided dataset.

        Parameters
        ----------
        x : ArrayLike
            Feature matrix of shape (n_samples, n_features).
        y : ArrayLike
            Class labels as a one-dimensional array.

        Returns
        -------
        BdeClassifier
            The fitted estimator instance.
        """

        return super()._fit(x, y)

    def predict(self, x: ArrayLike, raw: bool = False):
        """Predict class labels or return raw logits.

        Parameters
        ----------
        x : ArrayLike
            Feature matrix of shape (n_samples, n_features).
        raw : bool, default=False
            When ``True``, return the raw ensemble logits with shape
            ``(n_members, n_samples_draws, n_samples, n_classes)``. When
            ``False``, return hard labels shaped ``(n_samples,)`` in the
            original label encoding (if available).

        Returns
        -------
        numpy.ndarray | ArrayLike
            Hard labels when ``raw=False``, or raw logits when ``raw=True``.
        """

        if raw:
            return self._evaluate(x, raw=True)["raw"]
        labels = np.asarray(self._evaluate(x)["labels"])
        if not hasattr(self, "label_encoder_"):
            return labels
        return self.label_encoder_.inverse_transform(labels.astype(int))

    def predict_proba(self, x: ArrayLike):
        """Predict class probabilities for each sample.

        Parameters
        ----------
        x : ArrayLike
            Feature matrix of shape (n_samples, n_features).

        Returns
        -------
        numpy.ndarray
            Array of shape ``(n_samples, n_classes)`` with mean class
            probabilities across ensemble members and posterior samples.
        """

        probs = np.asarray(self._evaluate(x, probabilities=True)["probs"])
        return probs

    def __sklearn_tags__(self) -> Tags:
        base = super().__sklearn_tags__()
        base.estimator_type = "classifier"
        base.target_tags = TargetTags(
            required=True,
            one_d_labels=False,
            two_d_labels=False,
            positive_only=False,
            multi_output=False,
            single_output=True,
        )
        base.classifier_tags = ClassifierTags(
            poor_score=True,
            multi_class=True,
            multi_label=False,
        )
        return base
