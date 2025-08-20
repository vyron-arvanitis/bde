import jax
import jax.numpy as jnp
from bde.models.FNN_Builder import FNN
from bde.training.FNN_Trainer import FNN_Trainer
import optax

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

    model = FNN(sizes)
    trainer = FNN_Trainer()

    # Create optimizer
    optimizer = optax.adam(learning_rate=0.01)

    trainer.fit(
        model=model,
        X=X_true,
        y=y_true,
        optimizer=optimizer,
        epochs=1000
    )

    y_pred = trainer.predict(model.params, X_true)
    print("the first predictions are ", y_pred)


if __name__ == "__main__":
    main()