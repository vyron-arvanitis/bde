# Authors: Arvanitis V., Aslanidis A., Sommer E. and scikit-learn-contrib developers
# License: BSD 3 clause

# from ._version import __version__
from .bde import Bde, BdeClassifier, BdePredictor, BdeRegressor

__version__ = "0.1"

__all__ = [
    "BdeRegressor",
    "BdeClassifier",
    "Bde",
    "BdePredictor",
    "__version__",
]
