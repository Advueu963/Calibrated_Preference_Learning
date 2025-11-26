"""Base functions for ensemble-based estimators."""


# =============================================================================
# Imports
# =============================================================================

# Third party
import numpy as np


# =============================================================================
# Functions
# =============================================================================

def _predict_ensemble(estimator, X, sample_weight):
    """Predict using an ensemble estimator."""
    Y = [estimator.predict(X) for estimator in estimator.estimators_]

    # Join the predictions of all estimators
    axes = (1, 0, 2)
    Y = np.transpose(Y, axes)

    Y = [estimator._rank_algorithm.aggregate(y, sample_weight) for y in Y]

    return np.array(Y)
