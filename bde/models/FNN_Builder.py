# FNN_Builder.py
import jax
import jax.numpy as jnp
import optax

from bde.training.FNN_Trainer import FNN_Trainer

class FNN():
    'Builds a single FNN'
    def __init__(self, sizes):
        self.sizes = sizes
        self.params = None  # will hold initialized weights

    def init_mlp(self):
        sizes = self.sizes
        key = jax.random.PRNGKey(0)
        keys = jax.random.split(key, len(sizes) - 1)
        params = []
        for k, (m, n) in zip(keys, zip(sizes[:-1], sizes[1:])):
            W = jax.random.normal(k, (m, n)) / jnp.sqrt(m)
            b = jnp.zeros((n,))
            params.append((W, b))
        self.params = params
        return params



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

if __name__ == "__main__":
    main()