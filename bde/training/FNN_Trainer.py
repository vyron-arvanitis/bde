import jax
import jax.numpy as jnp
import optax


class FNN_Trainer():
    @staticmethod
    def mlp_forward(params, X):
        for (W, b) in params[:-1]:
            X = jnp.dot(X, W) + b
            X = jnp.tanh(X)
        W, b = params[-1]  # Fixed indentation - this should be outside the loop
        return jnp.dot(X, W) + b

    @staticmethod
    def mse_loss(params, X, y):
        pred = FNN_Trainer.mlp_forward(params, X)
        return jnp.mean((pred - y) ** 2)

    def create_train_step(self, optimizer):
        """Factory function to create a jitted train_step with the optimizer"""

        @jax.jit
        def train_step(params, opt_state, X, y):
            grads = jax.grad(self.mse_loss)(params, X, y)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state

        return train_step

    def fit(self, model, X, y, optimizer, epochs=100):
        if model.params is None:
            model.init_mlp()
        opt_state = optimizer.init(model.params)
        params = model.params

        # Create the jitted training step function
        train_step_fn = self.create_train_step(optimizer)

        for step in range(epochs):
            params, opt_state = train_step_fn(params, opt_state, X, y)
            if step % 200 == 0:
                loss = float(self.mse_loss(params, X, y))
                print(step, loss)
        model.params = params

    def predict(self):
        # TODO: Create an issue for predict method
        pass