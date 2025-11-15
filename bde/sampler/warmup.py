"""Multiple warmup implementations for different samplers.
Code snippets have been taken from
https://github.com/EmanuelSommer/MILE."""

import math
from threading import Lock
from typing import Callable, Optional

import blackjax
import blackjax.mcmc as mcmc
import jax
import jax.experimental
import jax.numpy as jnp
from blackjax.adaptation.base import AdaptationResults
from blackjax.adaptation.mclmc_adaptation import MCLMCAdaptationState
from blackjax.base import (
    AdaptationAlgorithm,
    ArrayLikeTree,
    PRNGKey,
)
from blackjax.diagnostics import effective_sample_size
from blackjax.util import pytree_size, streaming_average_update
from jax.flatten_util import ravel_pytree
from jax.tree_util import tree_map
from tqdm import tqdm

from bde.bde_builder import BdeBuilder


class WarmupProgress:
    """Thread-safe wrapper around tqdm progress bar for warmup."""

    def __init__(
        self,
        total: float,
        *,
        desc: str = "MCLMC warmup",
        scale: float = 1.0,
    ) -> None:
        self._bar = tqdm(
            total=total,
            desc=desc,
            position=0,
            dynamic_ncols=True,
            leave=True,
        )
        self._lock = Lock()
        self._scale = scale
        self._ticks = 0.0

    def update(self, n: int) -> None:
        if n <= 0:
            return
        with self._lock:
            self._ticks += float(n)
            target = self._ticks * self._scale
            desired = math.floor(target + 1e-9)
            delta = desired - self._bar.n
            if delta > 0:
                self._bar.update(delta)
                self._bar.refresh()

    def close(self) -> None:
        with self._lock:
            target = self._ticks * self._scale
            desired = math.ceil(target - 1e-9)
            delta = desired - self._bar.n
            if delta > 0:
                self._bar.update(delta)
            self._bar.close()


def mclmc_find_L_and_step_size(
    mclmc_kernel,
    state: MCLMCAdaptationState,
    rng_key: jax.random.PRNGKey,
    tune1_steps: int = 100,
    tune2_steps: int = 100,
    tune3_steps: int = 100,
    step_size_init: float = 0.005,
    desired_energy_var_start: float = 5e-4,
    desired_energy_var_end: float = 5e-4,
    trust_in_estimate: float = 1.5,
    num_effective_samples: int = 150,
    tick=None,
):
    """
    Find the optimal value of the parameters for the MCLMC algorithm.

    Parameters
    ----------
    mclmc_kernel
        The kernel function used for the MCMC algorithm.
    state
        The initial state of the MCMC algorithm.
    rng_key
        The random number generator key.
    tune1_steps
        The number of steps for the first step of the adaptation.
    tune2_steps
        The number of steps for the second step of the adaptation.
    tune3_steps
        The number of steps for the third step of the adaptation.
    step_size_init
        The initial step size for the MCMC algorithm.
    desired_energy_var_start
        The desired energy variance for the MCMC algorithm.
    desired_energy_var_end
        The desired energy variance for the MCMC algorithm at the end of the
        linear decay schedule. If the value is the same as the start value, no
        decay is applied.
    trust_in_estimate
        The trust in the estimate of optimal stepsize.
    num_effective_samples
        The number of effective samples for the MCMC algorithm.

    Returns
    -------
    A tuple containing the final state of the MCMC algorithm and the final
    hyperparameters.
    """

    part1_key, part2_key = jax.random.split(rng_key, 2)
    dim = pytree_size(state.position)
    params = MCLMCAdaptationState(
        L=jnp.maximum(jnp.sqrt(dim), 15.0),
        step_size=step_size_init,
        sqrt_diag_cov=jnp.ones((dim,)),
    )

    part1_key, part2_key = jax.random.split(rng_key, 2)
    state, params = make_L_step_size_adaptation(
        kernel=mclmc_kernel,
        dim=dim,
        tune1_steps=tune1_steps,
        tune2_steps=tune2_steps,
        desired_energy_var_start=desired_energy_var_start,
        desired_energy_var_end=desired_energy_var_end,
        trust_in_estimate=trust_in_estimate,
        num_effective_samples=num_effective_samples,
        tick=tick,
    )(state, params, part1_key)

    if tune3_steps != 0:
        state, params = make_adaptation_L(
            mclmc_kernel(params.sqrt_diag_cov),
            Lfactor=0.4,
            tick=tick,
        )(state, params, tune3_steps, part2_key)

    return state, params


def make_L_step_size_adaptation(
    kernel,
    dim: int,
    tune1_steps: int,
    tune2_steps: int,
    desired_energy_var_start: float,
    desired_energy_var_end: float,
    trust_in_estimate: float,
    num_effective_samples: int,
    tick=None,
):
    """
    Adapt the stepsize and L of the MCLMC kernel.

    Designed for the unadjusted MCLMC
    """
    decay_rate = (num_effective_samples - 1.0) / (num_effective_samples + 1.0)

    def get_desired_energy_var_linear(step: int) -> float:
        """Linearly decrease desired_energy_var from start to end."""
        total_steps = tune1_steps + tune2_steps + 1
        progress = jax.numpy.minimum(step / total_steps, 1.0)
        return (
            desired_energy_var_start
            - (desired_energy_var_start - desired_energy_var_end) * progress
        )

    def predictor(
        previous_state,
        params,
        adaptive_state: MCLMCAdaptationState,
        rng_key: jax.random.PRNGKey,
        step_number: int,
    ):
        """
        Do one step with the dynamics and updates the prediction for the opt. stepsize.

        Designed for the unadjusted MCHMC
        """
        time, x_average, step_size_max = adaptive_state

        # dynamics
        next_state, info = kernel(params.sqrt_diag_cov)(
            rng_key=rng_key,
            state=previous_state,
            L=params.L,
            step_size=params.step_size,
        )
        # step updating
        success, state, step_size_max, energy_change = handle_nans(
            previous_state,
            next_state,
            params.step_size,
            step_size_max,
            info.energy_change,
        )

        # Warning: var = 0 if there were nans, but we will give it a very small weight
        desired_energy_var = get_desired_energy_var_linear(step_number)
        xi = (
            jnp.square(energy_change) / (dim * desired_energy_var)
        ) + 1e-8  # 1e-8 is added to avoid divergences in log xi
        weight = jnp.exp(
            -0.5 * jnp.square(jnp.log(xi) / (6.0 * trust_in_estimate))
        )  # the weight reduces the impact of stepsizes which are much larger
        # or much smaller than the desired one.

        x_average = decay_rate * x_average + weight * (
            xi / jnp.power(params.step_size, 6.0)
        )
        time = decay_rate * time + weight
        step_size = jnp.power(
            x_average / time, -1.0 / 6.0
        )  # We use the Var[E] = O(eps^6) relation here.
        step_size = (step_size < step_size_max) * step_size + (
            step_size > step_size_max
        ) * step_size_max  # if the proposed stepsize is above the stepsize
        # where we have seen divergences
        # step_size = jnp.maximum(step_size, 1e-7) # Additional stepsize cap?
        params_new = params._replace(step_size=step_size)

        adaptive_state = (time, x_average, step_size_max)
        #
        # jax.debug.print("step {i} | ok={ok} | step_size={eps} | dE={dE}",
        #                 i=step_number,
        #                 ok=success,
        #                 eps=params.step_size,
        #                 dE=energy_change)

        return state, params_new, adaptive_state, success

    def step(iteration_state, weight_key_step):
        """Do one step and update the estimates of the post. & step size."""
        mask, rng_key, step_number = weight_key_step
        state, params, adaptive_state, streaming_avg = iteration_state

        state, params, adaptive_state, success = predictor(
            state,
            params,
            adaptive_state,
            rng_key,
            step_number,
        )

        if tick is not None:
            jax.debug.callback(tick, jnp.array(1, dtype=jnp.int32))

        x = ravel_pytree(state.position)[0]
        # update the running average of x, x^2
        streaming_avg = streaming_average_update(
            current_value=jnp.array([x, jnp.square(x)]),
            previous_weight_and_average=streaming_avg,
            weight=(1 - mask) * success * params.step_size,
            zero_prevention=mask,
        )

        return (state, params, adaptive_state, streaming_avg), None

    def run_steps(xs, state, params):
        step_number = jnp.arange(len(xs[0]))

        return jax.lax.scan(
            step,
            init=(
                state,
                params,
                (0.0, 0.0, jnp.inf),
                (0.0, jnp.array([jnp.zeros(dim), jnp.zeros(dim)])),
            ),
            xs=(xs[0], xs[1], step_number),
        )[0]

    def L_step_size_adaptation(state, params, rng_key):
        """Initialize adaptation of the step size and L."""
        L_step_size_adaptation_keys = jax.random.split(
            rng_key, tune1_steps + tune2_steps + 1
        )
        L_step_size_adaptation_keys, _final_key = (
            L_step_size_adaptation_keys[:-1],
            L_step_size_adaptation_keys[-1],
        )

        # we use the last num_steps2 to compute the diagonal preconditioner
        mask = 1 - jnp.concatenate((jnp.zeros(tune1_steps), jnp.ones(tune2_steps)))
        # run the steps
        state, params, _, (_, average) = run_steps(
            xs=(mask, L_step_size_adaptation_keys), state=state, params=params
        )

        L = params.L
        # determine L
        sqrt_diag_cov = params.sqrt_diag_cov
        if tune2_steps != 0.0:
            x_average, x_squared_average = average[0], average[1]
            variances = x_squared_average - jnp.square(x_average)
            # jax.debug.print('Average variances: {x}', x=jnp.mean(variances))
            L = jnp.sqrt(jnp.sum(variances))

        return state, MCLMCAdaptationState(L, params.step_size, sqrt_diag_cov)

    return L_step_size_adaptation


def make_adaptation_L(kernel, Lfactor, tick=None):
    """
    Determine L by the autocorrelations .

    (around 10 effective samples are needed for this to be accurate)
    """
    Lfactor = Lfactor

    def adaptation_L(
        state,
        params,
        num_steps: int,
        key: jax.random.PRNGKey,
        fft_params_limit: int = 2000,
        fft_samples_limit: int = 10000,
    ):
        adaptation_L_keys = jax.random.split(key, num_steps)

        def step(state, key):
            next_state, _ = kernel(
                rng_key=key,
                state=state,
                L=params.L,
                step_size=params.step_size,
            )

            if tick is not None:
                jax.debug.callback(tick, jnp.array(1, dtype=jnp.int32))

            return next_state, next_state.position

        state, samples = jax.lax.scan(
            f=step,
            init=state,
            xs=adaptation_L_keys,
        )

        flat_samples = jax.vmap(lambda x: ravel_pytree(x)[0])(samples)
        # Limit the number of samples and parameters to compute the FFT
        if flat_samples.shape[1] > fft_params_limit:
            flat_samples = flat_samples[
                :,
                jax.random.permutation(key, jnp.arange(flat_samples.shape[1]))[
                    :fft_params_limit
                ],
            ]
        if flat_samples.shape[0] > fft_samples_limit:
            # here not random but equally spaced (thinning)
            flat_samples = flat_samples[
                jnp.linspace(0, flat_samples.shape[0] - 1, fft_samples_limit).astype(
                    jnp.int32
                )
            ]
        ess = effective_sample_size(flat_samples[None, ...])
        # jax.debug.print("ESS: {x}", x=ess)

        return state, params._replace(
            L=Lfactor * params.step_size * jnp.mean(num_steps / ess)
        )

    return adaptation_L


def handle_nans(previous_state, next_state, step_size, step_size_max, kinetic_change):
    """
    Handle nans in the state.

    if there are nans, let's reduce the stepsize, and not update the state. The
    function returns the old state in this case.
    """
    reduced_step_size = 1
    p, _ = ravel_pytree(next_state.position)
    nonans = jnp.all(jnp.isfinite(p))
    state, step_size, kinetic_change = jax.tree_util.tree_map(
        lambda new, old: jax.lax.select(nonans, jnp.nan_to_num(new), old),
        (next_state, step_size_max, kinetic_change),
        (previous_state, step_size * reduced_step_size, 0.0),
    )
    return nonans, state, step_size, kinetic_change


def custom_mclmc_warmup(
    logdensity_fn: Callable,
    desired_energy_var_start: float = 5e-4,
    desired_energy_var_end: float = 1e-4,
    trust_in_estimate: float = 1.5,
    num_effective_samples: int = 100,
    step_size_init: float = 0.005,
    # TODO add saving_path: Path | None = None,
    progress: Optional[WarmupProgress] = None,
) -> AdaptationAlgorithm:
    """Warmup the initial state using MCLMC.

    Parameters
    ----------
    logdensity_fn
        The log density probability density function from which we wish to
        sample.
    desired_energy_var_start
        The desired energy variance at the start of the linear decay schedule.
    desired_energy_var_end
        The desired energy variance at the end of the linear decay schedule. If the
        value is the same as the start value, no decay is applied.
    trust_in_estimate
        The trust in the estimate.
    num_effective_samples
        The number of effective samples.

    Returns
    -------
    AdaptationResults
        The current state of the chain and the parameters for the sampler.
    """

    def kernel(sqrt_diag_cov):
        """Build the MCLMC kernel."""
        return mcmc.mclmc.build_kernel(
            logdensity_fn=logdensity_fn,
            integrator=mcmc.integrators.isokinetic_mclachlan,
            sqrt_diag_cov=sqrt_diag_cov,
        )

    def run(
        rng_key: PRNGKey,
        position: ArrayLikeTree,
        num_steps: int = 100,
    ) -> AdaptationResults:
        """Run the MCLMC warmup."""
        local_bar: Optional[WarmupProgress] = None
        if progress is None:
            local_bar = WarmupProgress(total=num_steps)

            def _tick(n):
                increment = int(n)
                if increment <= 0:
                    return
                local_bar.update(increment)

        else:

            def _tick(n):
                progress.update(int(n))

        initial_state = blackjax.mcmc.mclmc.init(
            position=position, logdensity_fn=logdensity_fn, rng_key=rng_key
        )

        phase_ratio = (0.8, 0.1, 0.1)  # might let the user change this

        # find values for L and step size
        (
            blackjax_state_after_tuning,
            blackjax_mclmc_sampler_params,
        ) = mclmc_find_L_and_step_size(
            mclmc_kernel=kernel,
            state=initial_state,
            rng_key=rng_key,
            step_size_init=step_size_init,
            tune1_steps=int(num_steps * phase_ratio[0]),
            tune2_steps=int(num_steps * phase_ratio[1]),
            tune3_steps=int(num_steps * phase_ratio[2]),
            desired_energy_var_start=desired_energy_var_start,
            desired_energy_var_end=desired_energy_var_end,
            trust_in_estimate=trust_in_estimate,
            num_effective_samples=num_effective_samples,
            tick=_tick,
        )
        if local_bar is not None:
            local_bar.close()
        return AdaptationResults(
            blackjax_state_after_tuning,
            blackjax_mclmc_sampler_params,
        )

    return AdaptationAlgorithm(run)


def warmup_bde(
    bde: BdeBuilder,
    logpost_one,
    step_size_init: float,
    desired_energy_var_start: float,
    desired_energy_var_end: float,
    warmup_steps: int,
) -> AdaptationResults:
    n_members = bde.n_members
    n_devices = jax.local_device_count()

    pad = (n_devices - (n_members % n_devices)) % n_devices if n_devices else 0
    n_members_pad = n_members + pad if n_devices else n_members
    device_weight = max(n_devices, 1)
    scale = 1.0 / device_weight
    progress = (
        WarmupProgress(total=warmup_steps, scale=scale) if warmup_steps > 0 else None
    )

    # Build the warmup adapter (same as your current code)
    adapt = custom_mclmc_warmup(
        logdensity_fn=logpost_one,
        desired_energy_var_start=desired_energy_var_start,
        desired_energy_var_end=desired_energy_var_end,
        trust_in_estimate=1.5,
        num_effective_samples=100,
        step_size_init=step_size_init,
        progress=progress,
    )

    def run_member(key, position):
        ar = adapt.run(key, position, warmup_steps)
        return ar.state, ar.parameters  # return plain pytrees (pmap-friendly)

    # Stack member params if needed (expect (n_members, ...) leaves)
    params_e = getattr(bde, "params_e", None)
    if params_e is None:
        params_e = tree_map(
            lambda *ps: jnp.stack(ps, axis=0), *[m.params for m in bde.members]
        )

    # Pad to multiple of n_devices (such that reshape (n_devices, n_members_per, ...))
    if pad:
        params_e = tree_map(
            lambda a: jnp.concatenate([a, jnp.repeat(a[:1], pad, axis=0)], axis=0),
            params_e,
        )
    n_members_pad = n_members + pad
    n_members_per = n_members_pad // max(n_devices, 1)

    # RNG keys: (n_members_pad, 2) -> (n_devices, n_members_per, 2)
    rng = jax.random.PRNGKey(bde.seed)
    keys_e = jax.random.split(rng, n_members_pad)
    keys_de = (
        keys_e.reshape(n_devices, n_members_per, 2)
        if n_devices > 0
        else keys_e.reshape(1, n_members_pad, 2)
    )

    # Shard params to (D, n_members_per, ...)
    params_de = tree_map(
        lambda a: a.reshape(n_devices, n_members_per, *a.shape[1:]), params_e
    )

    # Per-device function: vmap over local chunk
    def run_chunk(keys_chunk, positions_chunk):
        return jax.vmap(run_member, in_axes=(0, 0))(keys_chunk, positions_chunk)

    # pmap across devices; params/keys are sharded (in_axes=0)
    try:
        states_de, mclmc_params_de = jax.pmap(
            run_chunk, in_axes=(0, 0), out_axes=(0, 0)
        )(keys_de, params_de)

        # Back to (n_members_pad, ...)
        states_e, mclmc_params_e = tree_map(
            lambda a: a.reshape(n_members_pad, *a.shape[2:]),
            (states_de, mclmc_params_de),
        )

        # Drop padding
        if pad:
            states_e = tree_map(lambda a: a[:n_members], states_e)
            mclmc_params_e = tree_map(lambda a: a[:n_members], mclmc_params_e)

        return AdaptationResults(states_e, mclmc_params_e)
    finally:
        if progress is not None:
            progress.close()
