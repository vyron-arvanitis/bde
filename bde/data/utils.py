"""Validation helpers shared across estimators and data utilities."""

import warnings

import numpy as np
from sklearn.utils._tags import get_tags
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import (
    DataConversionWarning,
    check_array,
    check_is_fitted,
    check_X_y,
)

from bde.task import TaskType


def validate_fit_data(estimator, X, y):
    """Run sklearn-style checks for training data and update estimator metadata."""
    y_array = np.asarray(y)
    try:
        supports_multi_output = get_tags(estimator).target_tags.multi_output
    except Exception:
        supports_multi_output = False

    multi_output = False
    if y_array.ndim > 1:
        if supports_multi_output:
            if y_array.shape[1] == 1:
                y_array = y_array.reshape(-1)
            else:
                multi_output = True
        else:
            warnings.warn(
                (
                    "A column-vector y was passed when a 1d array was expected. Please"
                    " change the shape of y to (n_samples,), for example using ravel()."
                ),
                category=DataConversionWarning,
                stacklevel=2,
            )
            y_array = (
                y_array.reshape(y_array.shape[0])
                if y_array.shape[1] == 1
                else y_array[:, 0]
            )

    feature_names = getattr(X, "columns", None)
    X_checked, y_checked = check_X_y(
        X,
        y_array,
        accept_sparse=False,
        ensure_2d=True,
        ensure_min_samples=1,
        ensure_all_finite=True,
        y_numeric=(estimator.task == TaskType.REGRESSION),  # only True for regression
        multi_output=multi_output,
    )

    if feature_names is not None:
        estimator.feature_names_in_ = np.asarray(feature_names, dtype=object)
    elif hasattr(estimator, "feature_names_in_"):
        delattr(estimator, "feature_names_in_")

    estimator.n_features_in_ = X_checked.shape[1]

    if estimator.task == TaskType.CLASSIFICATION:
        check_classification_targets(y_checked)
        estimator.classes_ = np.unique(y_checked)
        estimator.n_outputs_ = 1 if y_checked.ndim == 1 else y_checked.shape[1]

    if estimator.task == TaskType.REGRESSION:
        estimator.n_outputs_ = 1 if y_checked.ndim == 1 else y_checked.shape[1]
        if y_checked.ndim == 1:
            y_checked = y_checked[:, None]  # restore column form for the network

    return X_checked, y_checked


def validate_predict_data(estimator, X):
    """Validate inputs for predict-like methods and enforce feature metadata."""

    check_is_fitted(estimator, attributes=["n_features_in_"])

    feature_names = getattr(X, "columns", None)
    X_checked = check_array(
        X,
        accept_sparse=False,
        ensure_2d=True,
        ensure_min_samples=1,
        ensure_all_finite=True,
    )

    if X_checked.shape[1] != estimator.n_features_in_:
        raise ValueError(
            f"X has {X_checked.shape[1]} features, but"
            f" {estimator.__class__.__name__} is expecting"
            f" {estimator.n_features_in_} features as input"
        )

    if hasattr(estimator, "feature_names_in_"):
        if feature_names is None:
            raise ValueError(
                "X has no feature names, but this estimator was fitted with feature"
                " names."
            )
        if list(feature_names) != list(estimator.feature_names_in_):
            raise ValueError("Feature names of X do not match those seen during fit.")

    return X_checked
