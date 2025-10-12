import jax
import jax.numpy as jnp

import optax
from bde.task import TaskType
from bde.loss.loss import *

from sklearn.model_selection import train_test_split

from typing import (
    Any,
    Optional,
    Sequence,
    Tuple,
)

class FnnTrainer:

    def __init__(self):
        """
        #TODO: documentation

        """
        self.history = {}
        self.log_every = 100
        self.keep_best = False

    def _reset_history(self):
        """
        #TODO: documentation

        Returns
        -------

        """
        self.history = {"train_loss": []}

    @staticmethod
    def make_loss_fn(model, loss_obj : BaseLoss):
        # returns (params, x, y) -> scalar
        def loss_fn(p, x, y):
            preds = model.forward(p, x)
            return loss_obj(preds, y) # loss must call model.forward(p, xb)
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
    def split_train_val(X, y, *, train_size=0.85, val_size=0.15, random_state=42, stratify=False, shuffle=True):
        """Split features and targets into training and validation subsets.

        Parameters
        ----------
        X : ArrayLike
            Feature matrix shaped (n_samples, n_features).
        y : ArrayLike
            Target array aligned with `X`.
        train_size : float
            Legacy alias for the training fraction. Ignored when `val_size` is given.
        val_size : float
            Fraction of samples reserved for validation. Must lie in (0, 1).
        random_state : int
            Seed forwarded to `train_test_split`.
        stratify : bool
            Whether to stratify by the label distribution.
        shuffle : bool
            Whether to shuffle before splitting.

        Returns
        -------
        tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]
            Training features, validation features, training targets, validation targets.
        """
        # prefer val_size; keep train_size for backward-compat
        if val_size is None and train_size is not None:
            val_size = 1.0 - train_size
        elif val_size is None:
            val_size = 0.15

        if not (0.0 < val_size < 1.0):
            raise ValueError("val_size must be in (0, 1).")

        strat = y if stratify else None

        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=val_size,
            random_state=random_state,
            stratify=strat,
            shuffle=shuffle,
        )
        return X_train, X_val, y_train, y_val

    @staticmethod
    def default_optimizer():
        return optax.adam(learning_rate=0.01)

    @staticmethod
    def default_loss(task : TaskType):
        if task == TaskType.REGRESSION:
            return GaussianNLL()
        elif task == TaskType.CLASSIFICATION:
            return CategoricalCrossEntropy()
        else:
            raise ValueError(f"Not an available task type was given!")
