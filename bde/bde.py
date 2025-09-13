from bde.bde_builder import BdeBuilder
from bde.bde_evaluator import BDEPredictor
from bde.sampler.probabilistic import ProbabilisticModel
from bde.sampler.prior import PriorDist
from bde.sampler.warmup import warmup_bde
from bde.sampler.mile_wrapper import MileWrapper

import jax
import jax.numpy as jnp
from jax.tree_util import tree_map, tree_leaves

from functools import partial
import optax


class BDE:
    def __init__(self,
                 n_members,
                 sizes,
                 seed
                 ):
        self.sizes = sizes
        self.n_members = n_members
        self.seed = seed
        self.bde = BdeBuilder(sizes, n_members, seed)
        self.members = self.bde.members

    def train(self,
              X,
              y,
              epochs,
              n_samples,
              warmup_steps,
              lr=1e-3,
              n_thinning=10
              ):
        self.bde.fit_members(x=X, y=y, optimizer=optax.adam(lr), epochs=epochs)

        prior = PriorDist.STANDARDNORMAL.get_prior()
        proto = self.bde.members[0]
        pm = ProbabilisticModel(module=proto, params=proto.params, prior=prior)

        logpost_one = partial(pm.log_unnormalized_posterior, x=X, y=y)

        warm = warmup_bde(self.bde, logpost_one, step_size_init=lr, warmup_steps=warmup_steps)

        init_positions_e = warm.state.position  # pytree with leading E
        tuned = warm.parameters  # MCLMCAdaptationState (vmapped)

        E = tree_leaves(init_positions_e)[0].shape[0]
        rng = jax.random.PRNGKey(int(self.seed))
        rng_keys_e = jax.vmap(lambda i: jax.random.fold_in(rng, i))(jnp.arange(E))

        # Normalize tuned hyperparam shapes
        L_e = tuned.L if jnp.ndim(tuned.L) == 1 else jnp.full((E,), tuned.L)
        step_e = tuned.step_size if jnp.ndim(tuned.step_size) == 1 else jnp.full((E,), tuned.step_size)
        sqrt_diag_e = tuned.sqrt_diag_cov

        sampler = MileWrapper(logpost_one)
        positions_eT, infos_eT, states_e = sampler.sample_batched(
            rng_keys_e=rng_keys_e,
            init_positions_e=init_positions_e,
            num_samples=n_samples,
            thinning=n_thinning,
            L_e=L_e,
            step_e=step_e,
            sqrt_diag_e=sqrt_diag_e,
            store_states=True,
        )

        self.positions_eT = positions_eT  # TODO: [@suggestion] maybe we should create this attribute in the __init__

    def evaluate(self, Xte):
        predictor = BDEPredictor(self.bde, self.positions_eT, Xte=Xte)
        return predictor.get_preds()
