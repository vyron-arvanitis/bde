import jax
import jax.numpy as jnp
from jax.tree_util import tree_map

class BDEPredictor:
    def __init__(self, model, positions_eT, Xte):

        self.model = model
        self.positions = positions_eT  # (E, T, ...)
        self.Xte = Xte

    def get_preds(self):
        # pure apply using explicit params (no mutation of model.params)
        def apply_with_params(p):
            return self.model.forward(p, self.Xte)  # (N, 2)

        preds = jax.vmap(                     # over ensemble members
                    jax.vmap(apply_with_params, in_axes=0),  # over samples
                    in_axes=0
                )(self.positions)

        mu    = preds[..., 0]                         
        sigma = jax.nn.softplus(preds[..., 1]) + 1e-6 

        mu_mean   = jnp.mean(mu, axis=(0, 1))
        var_ale   = jnp.mean(sigma**2, axis=(0, 1))
        var_epi   = jnp.var(mu, axis=(0, 1))
        std_total = jnp.sqrt(var_ale + var_epi)

        return mu_mean, std_total
    
    def get_preds_per_member(self):
        def apply_with_params(p):
            return self.model.forward(p, self.Xte)  # (N, 2)

        preds = jax.vmap(jax.vmap(apply_with_params, in_axes=0), in_axes=0)(self.positions)  # (E, T, N, 2)
        mu    = preds[..., 0]                          # (E, T, N)
        sigma = jax.nn.softplus(preds[..., 1]) + 1e-6  # (E, T, N)

        mu_mean_e = jnp.mean(mu, axis=1)                     # (E, N)
        var_ale_e = jnp.mean(sigma**2, axis=1)               # (E, N)
        var_epi_e = jnp.var(mu, axis=1)                      # (E, N)
        std_total_e = jnp.sqrt(var_ale_e + var_epi_e)        # (E, N)
        return mu_mean_e, std_total_e
