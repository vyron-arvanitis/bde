import jax
import jax.numpy as jnp
from jax.tree_util import tree_map

class BDEPredictor:
    def __init__(self, model, positions_eT, Xte):
        """
        model: must have a pure forward(params, x) -> preds with shape (N, 2)
        positions_eT: params pytree with leading axes (E, T, ...)
        Xte: test inputs, shape (N, D)
        """
        self.model = model
        self.positions = positions_eT  # (E, T, ...)
        self.Xte = Xte

    def get_preds(self):
        # pure apply using explicit params (no mutation of model.params)
        def apply_with_params(p):
            return self.model.forward(p, self.Xte)  # (N, 2)

        # preds shape: (E, T, N, 2)
        preds = jax.vmap(                     # over ensemble members
                    jax.vmap(apply_with_params, in_axes=0),  # over samples
                    in_axes=0
                )(self.positions)

        mu    = preds[..., 0]                          # (E, T, N)
        sigma = jax.nn.softplus(preds[..., 1]) + 1e-6  # (E, T, N)

        # combine uncertainty across both axes: Var = E[sigma^2] + Var[mu]
        mu_mean   = jnp.mean(mu, axis=(0, 1))                 # (N,)
        var_ale   = jnp.mean(sigma**2, axis=(0, 1))           # (N,)
        var_epi   = jnp.var(mu, axis=(0, 1))                  # (N,)
        std_total = jnp.sqrt(var_ale + var_epi)               # (N,)

        return mu_mean, std_total
