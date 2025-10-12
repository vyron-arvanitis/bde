"""Thin wrapper around BlackJAX MCLMC for ensemble sampling."""

from typing import Callable

import blackjax
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from jax.tree_util import tree_leaves, tree_map

from .types import ParamTree
from .utils import infer_dim_from_position_example, pad_axis0, _reshape_to_devices


class MileWrapper:
    """Convenience wrapper that hides boilerplate for MCLMC sampling."""

    def __init__(self, logdensity_fn: Callable[..., jax.Array]):
        self.logdensity_fn = logdensity_fn

    def _build_kernel(self, sqrt_diag_cov: jax.Array):
        """Construct an MCLMC kernel with the provided diagonal preconditioner."""
        return blackjax.mcmc.mclmc.build_kernel(
            logdensity_fn=self.logdensity_fn,
            integrator=blackjax.mcmc.integrators.isokinetic_mclachlan,
            sqrt_diag_cov=sqrt_diag_cov,
        )

    def init(self, position: ParamTree, rng_key: jax.Array):
        """Initialise a single-chain state."""
        return blackjax.mcmc.mclmc.init(
            position=position,
            logdensity_fn=self.logdensity_fn,
            rng_key=rng_key,
        )

    def step(
        self,
        rng_key: jax.Array,
        state,
        L: jax.Array,
        step_size: jax.Array,
        sqrt_diag_cov: jax.Array | None = None,
    ):
        """Advance one chain by a single MCLMC step."""
        if sqrt_diag_cov is None:
            dim = ravel_pytree(state.position)[0].shape[0]
            sqrt_diag_cov = jnp.ones((dim,))
        kernel = self._build_kernel(sqrt_diag_cov)
        next_state, info = kernel(rng_key=rng_key, state=state, L=L, step_size=step_size)
        return next_state, info

    def init_batched(self, positions_e: ParamTree, keys_e: jax.Array):
        """Initialise an ensemble of chains (vectorised over members)."""
        init_one = lambda pos, key: self.init(pos, key)
        return jax.vmap(init_one)(positions_e, keys_e)

    def _step_batched(
        self,
        keys_e: jax.Array,
        states_e,
        L_e: jax.Array,
        step_e: jax.Array,
        sqrt_diag_e: jax.Array | None = None,
    ):
        """Advance all ensemble members by one step (vectorised over members)."""
        if sqrt_diag_e is None:
            sample_leaf = tree_leaves(states_e.position)[0]
            sqrt_diag_e = jnp.ones((sample_leaf.shape[0], sample_leaf.shape[1]))

        def step_one(key, state, L_i, step_i, sdc_i):
            kernel = self._build_kernel(sdc_i)
            next_state, info = kernel(rng_key=key, state=state, L=L_i, step_size=step_i)
            return next_state, info

        return jax.vmap(step_one, in_axes=(0, 0, 0, 0, 0))(keys_e, states_e, L_e, step_e, sqrt_diag_e)

    def sample_batched(
        self,
        rng_keys_e: jax.Array,
        init_positions_e: ParamTree,
        num_samples: int,
        thinning: int = 1,
        L_e: jax.Array | None = None,
        step_e: jax.Array | None = None,
        sqrt_diag_e: jax.Array | None = None,
        store_states: bool = True,
    ) -> tuple:
        """Draw samples for every ensemble member in parallel.

        Parameters
        ----------
        rng_keys_e : jax.Array
            PRNG keys for each ensemble member.
        init_positions_e : ParamTree
            Warm-start positions produced by the warmup routine.
        num_samples : int
            Number of samples to draw per member (before thinning).
        thinning : int, default=1
            Factor by which to subsample the chain after drawing.
        L_e : jax.Array
            Per-member integrator lengths from warmup.
        step_e : jax.Array
            Per-member step sizes from warmup.
        sqrt_diag_e : jax.Array | None
            Optional per-member diagonal covariance factors.
        store_states : bool, default=True
            When ``True`` returns the final states alongside samples.

        Returns
        -------
        tuple
            Tuple containing sampled positions, kernel infos, and (optionally)
            final states.
        """

        if L_e is None or step_e is None:
            raise ValueError("Pass per-member L_e and step_e from warmup.")

        ensemble_leaves = tree_leaves(init_positions_e)
        E = ensemble_leaves[0].shape[0]
        device_count = jax.local_device_count()
        if device_count == 0:
            raise RuntimeError("No devices available for sampling.")

        if sqrt_diag_e is None:
            dim = infer_dim_from_position_example(init_positions_e)
            sqrt_diag_e = jnp.ones((E, dim))

        pad = (device_count - (E % max(device_count, 1))) % max(device_count, 1)
        E_pad = E + pad
        members_per_device = E_pad // max(device_count, 1)

        def pad_array(arr):
            return pad_axis0(arr, pad)

        rng_keys_e = pad_array(rng_keys_e)
        L_e = pad_array(L_e)
        step_e = pad_array(step_e)
        sqrt_diag_e = pad_array(sqrt_diag_e)
        init_positions_e = tree_map(pad_array, init_positions_e)

        keys_de = rng_keys_e.reshape(device_count, members_per_device, *rng_keys_e.shape[1:])
        L_de = _reshape_to_devices(L_e, device_count, members_per_device)
        step_de = _reshape_to_devices(step_e, device_count, members_per_device)
        sdc_de = _reshape_to_devices(sqrt_diag_e, device_count, members_per_device)
        pos_de = tree_map(lambda a: _reshape_to_devices(a, device_count, members_per_device), init_positions_e)

        def sample_chunk(keys_chunk, init_pos_chunk, L_chunk, step_chunk, sdc_chunk):
            def sample_one_member(key, init_pos, L_i, step_i, sdc_i):
                keys = jax.random.split(key, num_samples + 1)
                state = self.init(init_pos, keys[0])
                kernel = self._build_kernel(sdc_i)

                def body(st, k):
                    st, info = kernel(rng_key=k, state=st, L=L_i, step_size=step_i)
                    return st, (st.position, info)

                st, (pos_T, info_T) = jax.lax.scan(body, state, keys[1:])
                return pos_T, info_T, st

            return jax.vmap(sample_one_member, in_axes=(0, 0, 0, 0, 0), out_axes=(0, 0, 0))(
                keys_chunk, init_pos_chunk, L_chunk, step_chunk, sdc_chunk
            )

        positions_dET, infos_dET, states_dE = jax.pmap(
            sample_chunk, in_axes=(0, 0, 0, 0, 0), out_axes=(0, 0, 0)
        )(keys_de, pos_de, L_de, step_de, sdc_de)

        positions_eT = tree_map(lambda a: a.reshape(E_pad, *a.shape[2:]), positions_dET)
        infos_eT = tree_map(lambda a: a.reshape(E_pad, *a.shape[2:]), infos_dET)
        states_e = tree_map(lambda a: a.reshape(E_pad, *a.shape[2:]), states_dE)

        if pad:
            positions_eT = tree_map(lambda a: a[:E], positions_eT)
            infos_eT = tree_map(lambda a: a[:E], infos_eT)
            states_e = tree_map(lambda a: a[:E], states_e)

        if thinning > 1:
            positions_eT = tree_map(lambda x: x[:, ::thinning, ...], positions_eT)
            infos_eT = tree_map(lambda x: x[:, ::thinning, ...], infos_eT)

        if store_states:
            return positions_eT, infos_eT, states_e
        return states_e.position, infos_eT, states_e
