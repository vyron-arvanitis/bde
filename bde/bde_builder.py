"""this is a bde builder"""
import jax
import jax.numpy as jnp
from jax.tree_util import tree_map, tree_structure

import optax

from .models.models import Fnn
from .training.trainer import FnnTrainer

from bde.sampler.my_types import ParamTree


class BdeBuilder(Fnn, FnnTrainer):
    # TODO: build the BdeBuilderClass
    def __init__(self, sizes, n_members, seed: int = 100):
        Fnn.__init__(self, sizes)
        FnnTrainer.__init__(self)
        self.sizes = sizes
        self.n_members = n_members
        self.seed = seed

        self.members = self.deep_ensemble_creator(seed=self.seed)

    def get_model(self, seed: int) -> Fnn:
        """Create a single Fnn model and initialize its parameters

        Parameters
        ----------
        seed : int
            #TODO: complete documentation

        Returns
        -------

        """

        return Fnn(self.sizes, init_seed=seed)

    def deep_ensemble_creator(self, seed: int = 0) -> list[Fnn]:
        """Create an ensemble of ``n_members`` FNN models.

        Each member is initialized with a different random seed to encourage
        diversity within the ensemble. The initialized models are stored in the
        ``members`` attribute and returned.

        Returns
        -------
        list[Fnn]
            List of initialized FNN models comprising the ensemble.
        """

        return [self.get_model(seed + i) for i in range(self.n_members)]

    def fit_members(self, x, y, epochs, optimizer=None, loss=None):
        members = self.members

        opt = optimizer or self.default_optimizer()
        loss_obj = loss or self.default_loss()

        # Stack params across ensemble axis E
        params_e = tree_map(lambda *ps: jnp.stack(ps, axis=0),
                            *[m.params for m in members])  # (E, ...)

        # All members share the same architecture; use one model for forward()
        proto_model = members[0]

        loss_fn = FnnTrainer.make_loss_fn(proto_model, loss_obj)
        step_one = FnnTrainer.make_step(loss_fn, opt)
        vstep = FnnTrainer.make_vstep(step_one)

        opt_state_e = jax.vmap(opt.init)(params_e)

        self._reset_history()
        for epoch in range(epochs):
            params_e, opt_state_e, lvals_e = vstep(params_e, opt_state_e, x, y)
            mean_loss = float(jnp.mean(lvals_e))
            self.history["train_loss"].append(mean_loss)
            if epoch % self.log_every == 0:
                print(epoch, mean_loss)

        for i, m in enumerate(members):
            m.params = tree_map(lambda a: a[i], params_e)

        self.params_e = params_e  # TODO: [@suggestion] add this in the __init__ we never create this attribute apart from here

        return members

    def predict_ensemble(self, x, include_members: bool = False):
        """
        Ensemble prediction.

        Parameters
        ----------
        x : jnp.ndarray
            Input data of shape (n_samples, n_features).
        include_members : bool, optional
            If True, also return the stacked member predictions with shape
            (n_members, n_samples, output_dim).

        Returns
        -------
        tuple
            If return_members is False:
                (mean_pred, var_pred)
            Shapes:
                mean_pred: (n_samples, output_dim)
                var_pred: (n_samples, output_dim)
            If return_members is True:
                (mean_pred, var_pred, member_preds)
            where member_preds has shape (n_members, n_samples, output_dim).
        """
        # TODO: sklearn predict functions ... ??
        # TODO: we do not only need point predictions but interval prediction
        # TODO: predict(type_of_pred : interval, point ...
        # TODO: we can also use https://docs.jax.dev/en/latest/_autosummary/jax.random.normal.html
        if not self.members:
            raise ValueError("Ensemble has no members. Call `fit` or "
                             "`deep_ensemble_creator` first.")

        # single-model forward from the trainer; avoids name collision with this method
        member_means = jnp.stack(
            [model.predict(x) for model in self.members],
            axis=0
        )  # (n_members, n_samples, output_dim)

        ensemble_mean = jnp.mean(member_means, axis=0)  # (N, D) that is the point prediction
        ensemble_var = jnp.var(member_means, axis=0)  # (N, D) epistemic

        out = {
            "ensemble_mean": ensemble_mean,
            "ensemble_var": ensemble_var,
        }
        if include_members:
            out["member_means"] = member_means
        self.results = out
        return out

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

    def keys(self):
        """
        Return the keys currently  in `self.results`.
        """
        if not self.results:
            raise ValueError("No results saved. Call `predict_ensemble(..., cache=True)` first!")
        return list(self.results.keys())

    def __getattr__(self, item):
        """
        #TODO: dcoumentation
        Parameters
        ----------
        item

        Returns
        -------

        """
        if item in self.results:
            return self.results[item]
        else:
            raise AttributeError(f"{self.__class__.__name__} has no attribute '{item}' !")

    #
    # def store_ensemble_results(self, x, y=None, include_members: bool = True):
    #     """
    #     Cache ensemble predictions and, optionally, compute MSEs.
    #
    #     Returns
    #     -------
    #     dict
    #         Keys: "ensemble_mean", "ensemble_var", optional "member_means",
    #               optional "y", "ensemble_mse", "member_mse".
    #     """
    #     res = self.predict_ensemble(x, include_members=include_members)
    #
    #     if y is not None:
    #         res["y"] = y
    #         # Ensemble MSE (no params object for ensemble)
    #         res["ensemble_mse"] = jnp.mean((res["ensemble_mean"] - y) ** 2)
    #         # Per-member MSEs
    #         member_mse = [
    #             super().mse_loss(self, m.params, x, y) for m in self.members
    #         ]
    #         # stack to (n_members,) or (n_members, out_dim) depending on y shape
    #         res["member_mse"] = jnp.stack(member_mse, axis=0)
    #
    #     self.results = res
    #     return res
