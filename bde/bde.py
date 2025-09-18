from abc import abstractclassmethod

from bde.bde_builder import BdeBuilder
from bde.bde_evaluator import BDEPredictor
from bde.loss.loss import BaseLoss
from bde.sampler.probabilistic import ProbabilisticModel
from bde.sampler.prior import PriorDist
from bde.sampler.warmup import warmup_bde
from bde.sampler.mile_wrapper import MileWrapper

import jax
import jax.numpy as jnp
from jax.tree_util import tree_map, tree_leaves
from bde.task import TaskType
from functools import partial
import optax
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin


class Bde(BaseEstimator):
    def __init__(self, n_members=5,
                 sizes=None,
                 seed=0,
                 task: TaskType = None,
                 loss: BaseLoss = None,
                 activation="relu",
                 epochs=100,
                 n_samples=100,
                 warmup_steps=50,
                 lr=1e-3,
                 n_thinning=10
                 ):

        self.n_members = n_members
        self.sizes = sizes
        self.seed = seed
        self.task = task
        self.loss = loss
        self.task.validate_loss(self.loss)  # validate loss function
        self.activation = activation
        self.epochs = epochs
        self.n_samples = n_samples
        self.warmup_steps = warmup_steps
        self.lr = lr
        self.n_thinning = n_thinning

        if self.sizes is not None:
            self._build_bde()

        self.positions_eT = None  # will be set after training + sampling

    def _build_bde(self):
        self.bde = BdeBuilder(self.sizes, self.n_members, self.task, self.seed, act_fn=self.activation)
        self.members = self.bde.members

    def fit(self, X, y, ):
        self.bde.fit_members(x=X, y=y, optimizer=optax.adam(self.lr), epochs=self.epochs, loss=self.loss)

        prior = PriorDist.STANDARDNORMAL.get_prior()
        proto = self.bde.members[0]
        pm = ProbabilisticModel(module=proto, params=proto.params, prior=prior, task=self.task)

        logpost_one = partial(pm.log_unnormalized_posterior, x=X, y=y)

        warm = warmup_bde(self.bde, logpost_one, step_size_init=self.lr, warmup_steps=self.warmup_steps)

        init_positions_e = warm.state.position  # pytree with leading E
        tuned = warm.parameters  # MCLMCAdaptationState (vmapped)

        E = tree_leaves(init_positions_e)[0].shape[0]
        rng = jax.random.PRNGKey(int(self.seed))
        rng_keys_e = jax.vmap(lambda i: jax.random.fold_in(rng, i))(jnp.arange(E))

        # Normalize tuned hyperparam shapes
        L_e = tuned.L if jnp.ndim(tuned.L) == 1 else jnp.full((E,), tuned.L)
        step_e = tuned.step_size if jnp.ndim(tuned.step_size) == 1 else jnp.full((E,), tuned.step_size)
        sqrt_diag_e = tuned.sqrt_diag_cov

        sampler = MileWrapper(logpost_one)
        positions_eT, infos_eT, states_e = sampler.sample_batched(
            rng_keys_e=rng_keys_e,
            init_positions_e=init_positions_e,
            num_samples=self.n_samples,
            thinning=self.n_thinning,
            L_e=L_e,
            step_e=step_e,
            sqrt_diag_e=sqrt_diag_e,
            store_states=True,
        )

        self.positions_eT = positions_eT  # TODO: [@suggestion] maybe we should create this attribute in the __init__
        return self

    def evaluate(self,
                 Xte,
                 mean_and_std: bool = False,
                 credible_intervals: list[float] | None = None,
                 raw: bool = False,
                 probabilities: bool = False):
        predictor = BDEPredictor(self.bde, self.positions_eT, Xte=Xte, task=self.task)
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
    def __init__(self,
                 n_members=5,
                 sizes=None,
                 seed=0,
                 loss: BaseLoss = None,
                 activation="relu",
                 epochs=100,
                 n_samples=100,
                 warmup_steps=50,
                 lr=1e-3,
                 n_thinning=10):
        super().__init__(
            n_members=n_members,
            sizes=sizes,
            seed=seed,
            task=TaskType.REGRESSION,  # fixed
            loss=loss,
            activation=activation,
            epochs=epochs,
            n_samples=n_samples,
            warmup_steps=warmup_steps,
            lr=lr,
            n_thinning=n_thinning,
        )

    def predict(self, X, mean_and_std=False, credible_intervals=None, raw=False):
        out = self.evaluate(
            X,
            mean_and_std=mean_and_std,
            credible_intervals=credible_intervals,
            raw=raw,
        )
        if mean_and_std:
            return out["mean"], out["std"]
        elif credible_intervals:
            return out["mean"], out["credible_intervals"]
        return out["mean"]

    def get_raw_predictions(self, X):
        """Return raw ensemble predictions.

        Shape: (E, T, N, 2), where:
          - E = ensemble members
          - T = posterior samples per member
          - N = number of test points
          - 2 = (mu, sigma_param)
        """
        return self.evaluate(X, raw=True)["raw"]


class BdeClassifier(Bde, ClassifierMixin):
    def __init__(self,
                 n_members=5,
                 sizes=None,
                 seed=0,
                 loss: BaseLoss = None,
                 activation="relu",
                 epochs=100,
                 n_samples=100,
                 warmup_steps=50,
                 lr=1e-3,
                 n_thinning=10):
        super().__init__(
            n_members=n_members,
            sizes=sizes,
            seed=seed,
            task=TaskType.CLASSIFICATION,  # fixed
            loss=loss,
            activation=activation,
            epochs=epochs,
            n_samples=n_samples,
            warmup_steps=warmup_steps,
            lr=lr,
            n_thinning=n_thinning,
        )

    def predict(self, X):
        out = self.evaluate(X)
        return out["labels"]

    def predict_proba(self, X):
        out = self.evaluate(X, probabilities=True)
        return out["probs"]

    def get_raw_predictions(self, X):
        """Return raw ensemble predictions.

        Shape: (E, T, N, C), where:
          - E = ensemble members
          - T = posterior samples per member
          - N = number of test points
          - C = number of classes (logits before softmax)
        """
        return self.evaluate(X, raw=True)["raw"]
