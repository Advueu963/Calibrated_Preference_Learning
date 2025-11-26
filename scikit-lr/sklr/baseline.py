"""The :mod:`sklr.baseline` module includes simple baseline estimators."""


# =============================================================================
# Imports
# =============================================================================

# Standard
from abc import ABC
from abc import abstractmethod

# Third party
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.base import clone
from sklearn.utils.validation import check_is_fitted, _check_sample_weight

# Local application
from .base import LabelRankerMixin, PartialLabelRankerMixin


# =============================================================================
# Classes
# =============================================================================

class BaseDummy(BaseEstimator, ABC):
    """Base class for dummy estimators."""

    @abstractmethod
    def __init__(self, base_estimator, method, decimals):
        """Constructor."""
        self.base_estimator = base_estimator
        self.method = method
        self.decimals = decimals

    def fit(self, X, Y, sample_weight=None):
        """Fit the dummy estimator on the training data and rankings."""
        (X, Y) = self._validate_data(X, Y, multi_output=True)
        sample_weight = _check_sample_weight(sample_weight, X)

        self.n_classes_in_ = Y.shape[1]

        mask = Y == -1
        Y[mask] = np.inf

        y = np.argmin(Y, 1)

        self.estimator_ = clone(self.base_estimator).fit(X, y, sample_weight)

        return self

    def predict(self, X):
        """Predict rankings for the provided data."""
        check_is_fitted(self)

        X = self._validate_data(X, reset=False)
        n_samples = X.shape[0]

        shape = (n_samples, self.n_classes_in_)
        Y = np.zeros(shape, dtype=int)

        probabilities = self.estimator_.predict_proba(X).round(self.decimals)

        Y = pd.DataFrame(Y)
        Y.iloc[:, self.estimator_.classes_] = -probabilities

        return Y.rank(1, self.method).to_numpy(int)


class DummyLabelRanker(LabelRankerMixin, BaseDummy):
    """A dummy label ranker that make predictions using simple rules."""

    def __init__(self, base_estimator, *, decimals=5):
        """Constructor."""
        super(DummyLabelRanker, self).__init__(base_estimator, "min", decimals)


class DummyPartialLabelRanker(PartialLabelRankerMixin, BaseDummy):
    """A dummy partial label ranker that make predictions using simple rules."""

    def __init__(self, base_estimator, *, decimals=5):
        """Constructor."""
        super(DummyPartialLabelRanker, self).__init__(base_estimator, "dense", decimals)
