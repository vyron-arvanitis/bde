"""Callbacks used in training."""
import logging
from functools import partial
from pathlib import Path
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from bde.sampler.my_types import ParamTree
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
    path = base / f'{idx.item()}/sample_{n}.npz'
    if not path.parent.exists():
        path.parent.mkdir(parents=True)
    np.savez_compressed(
        path, **{name: np.array(leaf) for name, leaf in zip(param_names, leafs)}
    )
    return position


from functools import partial
from tqdm.auto import tqdm
import jax
import jax.numpy as jnp

def progress_bar_scan(n_steps: int, name: str, position: int, leave: bool = True) -> Callable:
    """Progress bar designed for lax.scan.

    Parameters:
    -----------
    n_steps: int
        Number of steps in the scan to show progress for.
    name: str
        Name of the progress bar to display.

    Notes:
    -----
    - This function is used to create a progress bar for a scan operation.
    - It is used in sampling to track the progress how many samples have been
        generated and how many are left.
    """
    progress_bar = tqdm(total=n_steps, desc=name)

    def _update_progress_bar(iter_num: int):
        # Update the progress bar
        _ = jax.lax.cond(
            iter_num >= 0,
            lambda _: jax.debug.callback(partial(progress_bar.update, 1)),
            lambda _: None,
            operand=None,
        )

        _ = jax.lax.cond(
            iter_num == n_steps - 1,
            lambda _: jax.debug.callback(partial(progress_bar.close)),
            lambda _: None,
            operand=None,
        )

    def _progress_bar_scan(f: Callable):
        def inner(carry, xs):
            if isinstance(xs, tuple):
                n, *_ = xs
            else:
                n = xs
            result = f(carry, xs)
            _update_progress_bar(n)
            return result

        return inner

    return _progress_bar_scan

