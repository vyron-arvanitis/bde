"""The class DataGen, generates arbitrary data!"""

from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np


# class TaskType:
#     """Holder for task types."""
#     REGRESSION = "regression"
#     CLASSIFICATION = "classification"


class DataLoader:
    def __init__(
            self,
            *,
            seed: int = 0,
            n_samples: int = 1024,
            n_features: int = 10,
            task: Optional[str] = None,
            data: Optional[dict[str, any]] = None
    ):
        self.seed = int(seed)
        self.n_samples = int(n_samples)
        self.n_features = int(n_features)

        # Generate data by default
        # TODO: delete below no ned for these attributes because we introduced a __getattr__
        # self.x = self.data["x"]
        # self.y = self.data["y"]
        # self.w = self.data["w"]

        # self.task = self.infer_task() # TODO: figure out where you are going with this

        if data is None:
            self.data = self._data_gen()
        else:
            self.data = {k: jnp.asarray(v) for k, v in data.items()}

    def _data_gen(self) -> dict[str, jnp.ndarray]:
        """Generate a simple linear regression dataset: y = x @ w + noise."""
        key = jax.random.PRNGKey(self.seed)
        k_x, k_w, k_eps = jax.random.split(key, 3)
        x = jax.random.normal(k_x, (self.n_samples, self.n_features))
        w = jax.random.normal(k_w, (self.n_features, 1))
        y = x @ w + jax.random.normal(k_eps, (self.n_samples, 1))
        return {"x": x, "w": w, "y": y}

    @classmethod
    def _from_dict(cls, data: dict[str, any]) -> "DataLoader":
        """Schema-agnostic constructor: any keys you like."""
        return cls(data=data)

    @classmethod
    def from_arrays(
            cls,
            X,
            y=None,
            *,
            task: Optional[str] = None,
            kind: str = "user",
    ) -> "DataLoader":
        """
        Construct a loader from user-provided arrays.

        Notes
        -----
        - Implementation should normalize to jax.numpy arrays,
          ensure x shape (N, D) and y shape (N, 1) if provided,
          and store `task` or leave it None for later inference.
        """
        # TODO: implement normalization & shape checks; set fields accordingly
        pass

    def __validate(self):
        """This method ensures the shapes and the types of the data


        Returns
        -------

        """

        # TODO: maybe ensure teh dimensions are equal
        # TODO: if user does not give data give warning and the gen data!
        pass

    def keys(self):
        """This method exposes the keys or available fields
        Returns
        -------

        """
        # TODO: figure out where we are going with this
        return list(self.data.keys())

    def load(self):
        """This method loads the correct type of data
        Returns
        -------

        """
        # TODO: delete
        pass

    @property
    def feature_dim(self) -> int:
        return int(self.x.shape[1])

    def __getitem__(self, item):
        if isinstance(item, str):
            return self.data[item]

        if isinstance(item, (np.ndarray, jnp.ndarray, list, tuple, slice)):
            n = len(self)  # original row count
            new_data = {}
            for k, v in self.data.items():
                if hasattr(v, "shape") and v.ndim >= 1 and int(v.shape[0]) == n:
                    new_data[k] = v[item]  # row-aligned -> subset
                else:
                    new_data[k] = v  # e.g. (D,1) weights -> keep same
            return DataLoader._from_dict(new_data)

        raise TypeError(f"Unsupported index type for DataLoader.__getitem__: {type(item)}")

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getattr__(self, item):
        if item in self.data:
            return self.data[item]
        else:
            raise AttributeError(f"{self.__class__.__name__} has no attribute '{item}' !")
