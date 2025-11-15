"""Callbacks used in training."""

import logging
from pathlib import Path
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from bde.sampler.types import ParamTree
from bde.sampler.utils import get_flattened_keys

logger = logging.getLogger(__name__)


def save_position(position: ParamTree, base: Path, idx: jnp.ndarray, n: int):
    """Save the position of the model.

    Parameters:
    -----------
    position: ParamTree
        Position of the model to save.
    base: Path
        Base path to save the samples.
    idx: jnp.ndarray
        Index of the current chain.
    n: int
        Index of the current sample.

    Notes:
    -----
    - This callback is used as io_callback during sampling to
        save the position after each sample.
    """
    leafs, _ = jax.tree.flatten(position)
    param_names = get_flattened_keys(position)
    path = base / f"{idx.item()}/sample_{n}.npz"
    if not path.parent.exists():
        path.parent.mkdir(parents=True)
    np.savez_compressed(
        path, **{name: np.array(leaf) for name, leaf in zip(param_names, leafs)}
    )
    return position


def progress_bar_scan(
    n_steps: int, name: str, position: int, leave: bool = True
) -> Callable:
    pbar = tqdm(total=int(n_steps), desc=name, position=position, leave=leave)

    def _progress_bar_scan(f: Callable):
        def inner(carry, xs):
            i = xs[0] if isinstance(xs, tuple) else xs
            carry2, ys = f(carry, xs)
            jax.debug.callback(lambda _i: pbar.update(1), i, ordered=True)
            return carry2, ys

        return inner

    def finish(x):
        jax.block_until_ready(x)
        pbar.close()
        return x

    _progress_bar_scan.finish = finish
    return _progress_bar_scan
