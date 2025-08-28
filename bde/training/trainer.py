import jax
import jax.numpy as jnp
import optax
from bde.loss.loss import LossMSE

from typing import (
    Any,
    Optional,
    Sequence,
    Tuple,
)


# TODO: major restructure is needed!
class FnnTrainer:

    def __init__(self):
        """
        #TODO: documentation

        """
        self.history = {}
        self.log_every = 100
        self.keep_best = False
        self.default_optimizer = self.default_optimizer

    def _reset_history(self):
        """
        #TODO: documentation

        Returns
        -------

        """
        self.history = {"train_loss": []}

    @staticmethod
    def create_train_step(model, optimizer, loss_obj):
        """
        #TODO:documentation

        Parameters
        ----------
        model
        optimizer
        loss_obj

        Returns
        -------

        """

        def loss_fn(params, x, y):
            preds = model.forward(params, x)  # (N,D)
            return loss_obj.mean_over_batch(y_true=y, y_pred=preds)  # scalar

        value_and_grad = jax.value_and_grad(loss_fn) # TODO: maybe move this inside the train_step?

        @jax.jit
        def train_step(params, opt_state, x, y):
            """
            #TODO: documentation
            Parameters
            ----------
            params
            opt_state
            x
            y

            Returns
            -------

            """
            loss_val, grads = value_and_grad(params, x, y)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss_val

        return train_step

    def train(
            self,
            model,
            x,
            y,
            optimizer: Optional[optax.GradientTransformation] = None,
            epochs: int = 100,
            loss=None,
            ):
        """
        Generic training loop.
        - model.forward(params, x) must exist
        - loss must implement Loss API (apply_reduced)
        """
        # lazy defaults TODO this is also not needed
        if loss is None:
            loss = self.default_loss()

        # init params #TODO: i think this is not needed
        # if getattr(model, "params", None) is None:
        #     model.init_mlp(seed=0)

        self._reset_history()

        params = model.params
        opt_state = optimizer.init(params)
        train_step = self.create_train_step(model, optimizer, loss)

        for step in range(epochs):
            params, opt_state, loss_val = train_step(params, opt_state, x, y)
            self.history["train_loss"].append(float(loss_val))
            if step % self.log_every == 0:
                print(step, float(loss_val))


        model.params = params
        return model

    @staticmethod
    def default_optimizer():
        return optax.adam(learning_rate=0.01)

    @staticmethod
    def default_loss():
        return LossMSE()
