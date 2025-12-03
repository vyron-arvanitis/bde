"""This file shows how to write test based on the scikit-learn common tests."""
# Authors: Arvanitis V., Aslanidis A., Sommer E. and scikit-learn-contrib developers
# License: BSD 3 clause
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.utils.estimator_checks import parametrize_with_checks

from bde.bde import Bde  # import the base class
from bde.utils.discovery import all_estimators

FAST_BDE_KWARGS = dict(
    n_members=2,
    hidden_layers=[2],
    epochs=1,
    warmup_steps=5,
    n_samples=1,
    n_thinning=1,
    patience=1,
)

estimators = []
for _, Est in all_estimators():
    if Est in (MLPClassifier, MLPRegressor):
        continue
    if issubclass(Est, Bde):
        estimators.append(Est(**FAST_BDE_KWARGS))
    else:
        estimators.append(Est())


@parametrize_with_checks(estimators)
def test_estimators(estimator, check, request):
    check(estimator)
