import jax
import jax.numpy as jnp
import optax

from bde.models.models import Fnn
from bde.training.trainer import FnnTrainer
from bde.bde_builder import BdeBuilder
from bde.viz.plotting import plot_pred_vs_true

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

def main():
    # generate True data for test purposes
    main_key = jax.random.PRNGKey(0)
    k_X, k_W, k_eps = jax.random.split(main_key, 3)
    X_true = jax.random.normal(k_X, (1024, 10))
    true_W = jax.random.normal(k_W, (10, 1))
    y_true = X_true @ true_W + 0.1 * jax.random.normal(k_eps, (1024, 1))

    sizes = [10, 64, 64, 1]

    model = Fnn(sizes)
    trainer = FnnTrainer()

    # Create optimizer
    optimizer = optax.adam(learning_rate=0.01)

    trainer.fit(
        model=model,
        x=X_true,
        y=y_true,
        optimizer=optimizer,
        epochs=1000
    )
    print(trainer.history["train_loss"][:10])  # first 10 losses


    y_pred = trainer.predict(model.params, X_true)
    print("the first predictions are ", y_pred)

    print("-----------------------------------------------------------")
    bde = BdeBuilder(sizes, n_members=3, epochs=500, optimizer=optax.adam(1e-2))
    # fit + predict
    bde.fit(x=X_true, y=y_true, optimizer=bde.optimizer, epochs=500, model=None)
    out = bde.predict_ensemble(X_true, include_members=True)
    print(out["ensemble_mean"])
    print(out["ensemble_var"])

    print("keys:", list(out.keys()))            # ['ensemble_mean', 'ensemble_var']
    print("mean shape:", out["ensemble_mean"].shape)
    plot_pred_vs_true(out["ensemble_mean"], y_true, "trial", savepath="to_be_deleted" )

if __name__ == "__main__":
    main()