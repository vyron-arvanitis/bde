"""this is a bde builder"""

from .models.models import Fnn
from .training.trainer import FnnTrainer
import optax
import jax.numpy as jnp

class BdeBuilder(Fnn, FnnTrainer):
    # TODO: build the BdeBuilderClass
    def __init__(self, sizes, n_members, epochs, optimizer, base_seed: int = 100):
        Fnn.__init__(self, sizes)
        FnnTrainer.__init__(self)
        self.sizes = sizes
        self.n_members = n_members
        self.epochs = epochs
        self.base_seed = base_seed

        self.members = self.deep_ensemble_creator(base_seed=self.base_seed)
        self.optimizer = optimizer or optax.adam(learning_rate=0.01)
        self.results = {}

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

    def deep_ensemble_creator(self, base_seed: int = 0) -> list[Fnn]:
        """Create an ensemble of ``n_members`` FNN models.

        Each member is initialized with a different random seed to encourage
        diversity within the ensemble. The initialized models are stored in the
        ``members`` attribute and returned.

        Returns
        -------
        list[Fnn]
            List of initialized FNN models comprising the ensemble.
        """

        return [self.get_model(base_seed + i) for i in range(self.n_members)]

    def fit(self, x, y, epochs=None) -> "BdeBuilder":
        """Train each member of the ensemble

        Parameters
        ---------
        #TODO: documentation
        """
        #TODO: this should get the dataloader and the datapreprocessor
        #TODO: here comes the SAMPLING AS WELL
        for member in self.members:
            super().train(model=member, x=x, y=y, optimizer=self.optimizer, epochs=epochs,loss=None)
        return self

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
        #TODO: sklearn predict functions ... ??
        #TODO: we do not only need point predictions but interval prediction
        #TODO: predict(type_of_pred : interval, point ...
        #TODO: we can also use https://docs.jax.dev/en/latest/_autosummary/jax.random.normal.html
        if not self.members:
            raise ValueError("Ensemble has no members. Call `fit` or "
                             "`deep_ensemble_creator` first.")

        # single-model forward from the trainer; avoids name collision with this method
        member_preds = jnp.stack(
            [model.predict(x) for model in self.members],
            axis=0
        )  # (n_members, n_samples, output_dim)

        ensemble_mean = jnp.mean(member_preds, axis=0)  # (N, D) that is the point prediction
        ensemble_var = jnp.var(member_preds, axis=0)  # (N, D) epistemic

        out = {
            "ensemble_mean": ensemble_mean,
            "ensemble_var": ensemble_var,
        }
        if include_members:
            out["member_means"] = member_preds
        self.results = out
        return out

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
