import jax
import jax.numpy as jnp
import optax


# TODO: major restructure is needed!
class FnnTrainer:

    def __init__(self):
        self.history = {}
        self.log_every = 100
        self.keep_best = False
        self.default_optimizer = self.default_optimizer

    def _reset_history(self):
        self.history = {"train_loss": []}

    @staticmethod
    def mse_loss(model, params, x, y):
        """
        #TODO:documentation
        Parameters
        ----------
        model
        params
        x
        y

        Returns
        -------

        """
        prediction = model.forward(params, x)
        return jnp.mean((prediction - y) ** 2)

    def create_train_step(self, model, optimizer):
        """
        #TODO:documentation

        Parameters
        ----------
        model
        optimizer

        Returns
        -------

        """

        @jax.jit
        def train_step(params, opt_state, x, y):
            """
            #TODO:documentation

            Parameters
            ----------
            params
            opt_state
            x
            y

            Returns
            -------

            """
            grads = jax.grad(lambda p, x, y: self.mse_loss(model, p, x, y))(params, x, y)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state

        return train_step

    def fit(self, model, x, y, optimizer, epochs=100):
        """
        #TODO: documentation

        Parameters
        ----------
        model
        x
        y
        optimizer
        epochs

        Returns
        -------

        """
        if model.params is None:
            model.init_mlp(seed=0)

        self._reset_history()  # clear history at the start of each training

        opt_state = optimizer.init(model.params)
        params = model.params

        # Create the jitted training step function
        train_step_fn = self.create_train_step(model, optimizer)

        for step in range(epochs):
            params, opt_state = train_step_fn(params, opt_state, x, y)
            loss = float(self.mse_loss(model, params, x, y))
            self.history["train_loss"].append(loss)

            if step % self.log_every == 0:
                print(step, loss)
        model.params = params

        return model

    @staticmethod
    def default_optimizer():
        return optax.adam(learning_rate=0.01)
