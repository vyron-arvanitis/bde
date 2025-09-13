import jax
import jax.numpy as jnp
import blackjax
from jax.tree_util import tree_map
from jax.flatten_util import ravel_pytree
from bde.sampler.callbacks import progress_bar_scan

class MileWrapper:
    def __init__(self, logdensity_fn):
        self.logdensity_fn = logdensity_fn
        self._kernel_builder = lambda sqrt_diag_cov: blackjax.mcmc.mclmc.build_kernel(
            logdensity_fn=self.logdensity_fn,
            integrator=blackjax.mcmc.integrators.isokinetic_mclachlan,
            sqrt_diag_cov=sqrt_diag_cov,
        )

    def init(self, position, rng_key):
        return blackjax.mcmc.mclmc.init(position=position,
                                        logdensity_fn=self.logdensity_fn,
                                        rng_key=rng_key)

    def step(self, rng_key, state, L, step_size, sqrt_diag_cov=None):
        if sqrt_diag_cov is None:
            dim = ravel_pytree(state.position)[0].shape[0]
            sqrt_diag_cov = jnp.ones((dim,))
        kernel = self._kernel_builder(sqrt_diag_cov)
        next_state, info = kernel(rng_key=rng_key, state=state, L=L, step_size=step_size)
        return next_state, info

    def init_batched(self, positions_e, keys_e):
        """positions_e: pytree with leading axis E; keys_e: (E,) PRNGKey"""
        init_one = lambda pos, key: blackjax.mcmc.mclmc.init(position=pos,
                                                             logdensity_fn=self.logdensity_fn,
                                                             rng_key=key)
        return jax.vmap(init_one)(positions_e, keys_e)

    def _step_batched(self, keys_e, states_e, L_e, step_e, sqrt_diag_e=None):
        """Vectorized one-step over ensemble axis E."""
        if sqrt_diag_e is None:
            # make ones per member with correct dim
            dim = ravel_pytree(states_e.position)[0].shape[1]  # (E, dim)
            sqrt_diag_e = jnp.ones((states_e.position[0].shape[0], dim))  # fallback; adjust if needed

        def step_one(key, state, L_i, step_i, sdc_i):
            kernel = self._kernel_builder(sdc_i)
            next_state, info = kernel(rng_key=key, state=state, L=L_i, step_size=step_i)
            return next_state, info

        return jax.vmap(step_one, in_axes=(0, 0, 0, 0, 0))(keys_e, states_e, L_e, step_e, sqrt_diag_e)

    def sample_batched(self, rng_keys_e, init_positions_e, num_samples, thinning=1, 
                       L_e=None, step_e=None, sqrt_diag_e=None, store_states=True):
        E = jax.tree_util.tree_leaves(init_positions_e)[0].shape[0]
        if L_e is None or step_e is None:
            raise ValueError("Pass per-member L_e and step_e from warmup.")
        if sqrt_diag_e is None:
            dim = jax.flatten_util.ravel_pytree(init_positions_e)[0].shape[-1]
            sqrt_diag_e = jnp.ones((E, dim))

        def sample_one(key, init_pos, L_i, step_i, sdc_i):
            keys = jax.random.split(key, num_samples + 1)
            state = self.init(init_pos, keys[0])

            def one_step(st, k):
                st, info = self.step(k, st, L=L_i, step_size=step_i, sqrt_diag_cov=sdc_i)
                return st, (st.position, info)

            state, (positions, infos) = jax.lax.scan(one_step, state, keys[1:])
            if thinning > 1:
                positions = jax.tree_util.tree_map(lambda x: x[::thinning], positions)
                infos     = jax.tree_util.tree_map(lambda x: x[::thinning], infos)
            return positions, infos, state

        positions_eT, infos_eT, states_e = jax.vmap(
            sample_one,
            in_axes=(0, 0, 0, 0, 0),
            out_axes=(0, 0, 0),
        )(rng_keys_e, init_positions_e, L_e, step_e, sqrt_diag_e)

        return (positions_eT, infos_eT, states_e) if store_states else (states_e.position, infos_eT, states_e)

