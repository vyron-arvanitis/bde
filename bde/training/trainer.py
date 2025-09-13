import jax
import jax.numpy as jnp
import jax.scipy.stats as stats

import optax
from bde.loss.loss import GaussianNLL

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
    def make_loss_fn(model, loss_obj):
        # returns (params, x, y) -> scalar
        def loss_fn(p, xb, yb):
            return loss_obj(p, model, xb, yb)  # loss must call model.forward(p, xb)

        return loss_fn

    @staticmethod
    def make_step(loss_fn, optimizer):
        @jax.jit
        def step(p, opt_state, xb, yb):
            lval, grads = jax.value_and_grad(loss_fn)(p, xb, yb)
            updates, opt_state = optimizer.update(grads, opt_state, p)  # optimizer is GradientTransformation
            p = optax.apply_updates(p, updates)
            return p, opt_state, lval

        return step

    @staticmethod
    def make_vstep(step_one):
        # vmaps the single-member step across the leading ensemble axis
        @jax.jit
        def vstep(p_e, os_e, xb, yb):
            return jax.vmap(step_one, in_axes=(0, 0, None, None),
                            out_axes=(0, 0, 0))(p_e, os_e, xb, yb)

        return vstep

    def fit_model(self, model, x, y, optimizer=None, epochs=100, loss=None):
        opt = optimizer or self.default_optimizer()
        loss_obj = loss or self.default_loss()

        self._reset_history()
        params = model.params
        opt_state = opt.init(params)

        loss_fn = self.make_loss_fn(model, loss_obj)
        step_one = self.make_step(loss_fn, opt)

        for epoch in range(epochs):
            params, opt_state, lval = step_one(params, opt_state, x, y)
            self.history["train_loss"].append(float(lval))
            if epoch % self.log_every == 0:
                print(epoch, float(lval))

        model.params = params
        return model

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

        if loss is None:
            loss = self.default_loss()

        self._reset_history()

        params = model.params
        opt_state = optimizer.init(params)

        def loss_fn(p, x, y):
            return loss(p, model, x, y)

        @jax.jit
        def step(params, opt_state, x, y):
            loss_val, grads = jax.value_and_grad(loss_fn)(params, x, y)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss_val

        for epoch in range(epochs):
            params, opt_state, loss_val = step(params, opt_state, x, y)
            self.history["train_loss"].append(float(loss_val))
            if epoch % self.log_every == 0:
                print(epoch, float(loss_val))

        model.params = params
        return model

    @staticmethod
    def default_optimizer():
        return optax.adam(learning_rate=0.01)

    @staticmethod
    def default_loss():
        return GaussianNLL()
