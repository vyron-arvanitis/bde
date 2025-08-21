"""this is a bde builder"""

from models.models import Fnn
from training.trainer import FnnTrainer
import optax
import jax.numpy as jnp


class BdeBuilder(Fnn, FnnTrainer):
    # TODO: build the BdeBuilderClass
    def __init__(self, sizes, n_members, epochs, optimizer):
        Fnn.__init__(self, sizes)
        FnnTrainer.__init__(self)
        self.sizes = sizes
        self.n_members = n_members
        self.epochs = epochs

        self.members = []
        self.optimizer = optimizer or optax.adam(learning_rate=0.01)

    def get_model(self, seed: int) -> Fnn:
        """Create a single Fnn model and initialize its parameters

        Parameters
        ----------
        seed : int
            #TODO: complete documentation

        Returns
        -------

        """
        m = Fnn(self.sizes)
        m.init_mlp(seed=seed)
        return m

    def deep_ensemble_creator(self):
        """Create an ensemble of ``n_members`` FNN models.

        Each member is initialized with a different random seed to encourage
        diversity within the ensemble. The initialized models are stored in the
        ``members`` attribute and returned.

        Returns
        -------
        list[Fnn]
            List of initialized FNN models comprising the ensemble.
        """

        self.members = [self.get_model(seed) for seed in range(self.n_members)]
        return self.members

    def fit(self, model, x, y, optimizer, epochs=100):
        """Train each member of the ensemble

        Parameters
        ---------
        #TODO: documentation
        """
        if not self.members:
            self.deep_ensemble_creator()

        for member in self.members:
            super().fit(model=member, x=x, y=y, optimizer=self.optimizer, epochs=epochs or self.epochs)
        return self.members

    def predict(self, x, return_members: bool = False):
        """
        Ensemble prediction.

        Parameters
        ----------
        x : jnp.ndarray
            Input data of shape (n_samples, n_features).
        return_members : bool, optional
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
        if not self.members:
            raise ValueError("Ensemble has no members. Call `fit` or "
                             "`deep_ensemble_creator` first.")

        # single-model forward from the trainer; avoids name collision with this method
        member_preds = jnp.stack(
            [FnnTrainer.predict(self, m.params, x) for m in self.members],
            axis=0
        )  # (n_members, n_samples, output_dim)

        mean_pred = jnp.mean(member_preds, axis=0)  # (n_samples, output_dim)
        var_pred = jnp.var(member_preds, axis=0)
        if return_members:
            return mean_pred, var_pred, member_preds
        else:
            return mean_pred, var_pred

    def store_ensemble_results(self):
        pass
