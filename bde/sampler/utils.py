"""Sampler-side pytree helpers shared across
the package, taken from
https://github.com/EmanuelSommer/MILE"""

import logging
import operator
from functools import reduce

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from jax.tree_util import tree_map

from bde.sampler.types import ParamTree

logger = logging.getLogger(__name__)


def get_flattened_keys(d: dict, sep=".") -> list[str]:
    """Recursively get `sep` delimited path to the leaves of a tree.

    Parameters:
    -----------
    d: dict
        Parameter Tree to get the names of the leaves from.
    sep: str
        Separator for the tree path.

    Returns:
    --------
        list of names of the leaves in the tree.
    """
    keys = []
    for k, v in d.items():
        if isinstance(v, dict):
            keys.extend([f"{k}{sep}{kk}" for kk in get_flattened_keys(v)])
        else:
            keys.append(k)
    return keys


def get_by_path(tree: dict, path: list):
    """Access a nested object in root by item sequence."""
    return reduce(operator.getitem, path, tree)


def set_by_path(tree: dict, path: list, value):
    """Set a value in a pytree by path."""
    reduce(operator.getitem, path[:-1], tree)[path[-1]] = value
    return tree


def count_chains(samples: ParamTree) -> int:
    """Find number of chains in the samples.

    Raises:
        ValueError: If the number of chains is not consistent across layers.
    """
    n = set([x.shape[0] for x in jax.tree.leaves(samples)])
    if len(n) > 1:
        raise ValueError(f"Ambiguous chain dimension across layers. Found {n}")
    return n.pop()


def count_samples(samples: ParamTree) -> int:
    """Find number of samples in the samples.

    Raises:
        ValueError: If the number of samples is not consistent across layers.
    """
    n = set([x.shape[1] for x in jax.tree.leaves(samples)])
    if len(n) > 1:
        raise ValueError(f"Ambiguous sample dimension across layers. Found {n}")
    return n.pop()


def get_mem_size(x: ParamTree) -> int:
    """Get the memory size of the model."""
    return sum([x.nbytes for x in jax.tree_leaves(x)])


def count_params(params: ParamTree) -> int:
    """Count the number of parameters in the model."""
    return sum([x.size for x in jax.tree.leaves(params)])


def count_nan(params: ParamTree) -> ParamTree:
    """Count the number of NaNs in the parameter tree."""
    return jax.tree.map(lambda x: jnp.isnan(x).sum().item(), params)


def impute_nan(params: ParamTree, value: float = 0.0) -> ParamTree:
    """Impute NaNs in the parameter tree with a value."""
    return jax.tree.map(lambda x: jnp.where(jnp.isnan(x), value, x), params)


def infer_dim_from_position_example(pos_e):
    """Return the flattened dimensionality of a single ensemble element.

    Parameters
    ----------
    pos_e : ParamTree
        Parameter tree with leading ensemble axis.

    Returns
    -------
    int
        Flattened dimension of one member.
    """
    ex = tree_map(lambda a: a[0], pos_e)
    flat, _ = ravel_pytree(ex)
    return flat.shape[0]


def pad_axis0(a, pad):
    """Pad the leading axis by repeating the first element.

    Parameters
    ----------
    a : jax.Array
        Array whose leading axis enumerates ensemble members.
    pad : int
        Number of additional entries to append.

    Returns
    -------
    jax.Array
        Array with `pad` extra elements appended on axis 0.
    """
    if pad == 0:
        return a
    return jnp.concatenate([a, jnp.repeat(a[:1], pad, axis=0)], axis=0)


def _reshape_to_devices(a, D, E_per):
    """Reshape a padded array to `(devices, local_members, ...)` layout.

    Parameters
    ----------
    a : jax.Array
        Array whose leading axis length equals `D * E_per`.
    D : int
        Number of local devices.
    E_per : int
        Members per device after padding.

    Returns
    -------
    jax.Array
        Reshaped array ready for `pmap` consumption.
    """
    return a.reshape(D, E_per, *a.shape[1:])
