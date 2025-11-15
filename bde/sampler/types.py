"""Central type aliases shared across the sampler subpackage.

`ParamTree` captures the nested pytree of model parameters, `FileTree` mirrors the
directory layout when persisting samples (kept for potential future use), and
`PRNGKey` annotates JAX pseudo-random keys.
"""

import enum
import sys
import typing
from pathlib import Path

import jax

LayerParams: typing.TypeAlias = typing.Tuple[jax.Array, jax.Array]
ParamList: typing.TypeAlias = typing.List[LayerParams]
ParamTree: typing.TypeAlias = dict[str, typing.Union[jax.Array, "ParamTree"]]
# `FileTree` mirrors the on-disk nesting when persisting checkpoints;
# retained for API stability.
FileTree: typing.TypeAlias = dict[str, typing.Union[Path, "FileTree"]]
# A `PRNGKey` is a uint32 array of shape (2,) produced by `jax.random.PRNGKey`.
PRNGKey = jax.Array


class CustomEnumMeta(enum.EnumMeta):
    """Custom Enum Meta Class for Better Error Handling."""

    def __call__(cls, value, **kwargs):
        """Extend the __call__ method to raise a ValueError."""
        if value not in cls._value2member_map_:
            raise ValueError(
                f"{cls.__name__} must be one of {[*cls._value2member_map_.keys()]}"
            )
        return super().__call__(value, **kwargs)


class BaseEnum(enum.Enum, metaclass=CustomEnumMeta):
    """BaseEnum Class for implementing custom Enum classes."""


if sys.version_info >= (3, 11):

    class BaseStrEnum(enum.StrEnum, metaclass=CustomEnumMeta):
        """BaseEnum Class for implementing custom Enum classes."""

        def __str__(self):
            """Return the string representation of the Enum."""
            return self.value

    class BaseIntEnum(enum.IntEnum, metaclass=CustomEnumMeta):
        """BaseEnum Class for implementing custom Enum classes."""

        def __str__(self):
            """Return the string representation of the Enum."""
            return self.value.__str__()

else:

    class BaseStrEnum(str, BaseEnum):
        """BaseEnum Class for implementing custom Enum classes."""

        def __str__(self):
            """Return the string representation of the Enum."""
            return self.value

    class BaseIntEnum(int, BaseEnum):
        """BaseEnum Class for implementing custom Enum classes."""

        def __str__(self):
            """Return the string representation of the Enum."""
            return self.value.__str__()
