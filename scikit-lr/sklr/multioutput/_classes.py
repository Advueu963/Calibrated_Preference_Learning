"""This module implements partial label ranking estimators."""


# =============================================================================
# Imports
# =============================================================================

# Standard
from collections import defaultdict
import itertools

# Third party
from scipy.special import logsumexp
from sklearn.base import clone
from sklearn.base import BaseEstimator
from sklearn.naive_bayes import CategoricalNB, GaussianNB
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted
from sklr.base import PartialLabelRankerMixin
import numpy as np

# Local application
from .utils import aggregate, borda


# =============================================================================
# Classes
# =============================================================================

class PartialLabelRankerChain(PartialLabelRankerMixin, BaseEstimator):
    """A partial label ranking model that arranges classifiers into a chain."""

    def __init__(self,
                 gaussian_estimator,
                 categorical_estimator,
                 *,
                 order="heuristic",
                 random_state=None):
        self.gaussian_estimator = gaussian_estimator
        self.categorical_estimator = categorical_estimator
        self.order = order
        self.random_state = random_state

    def _validate_order(self, Y, order, random_state):
        """"""
        n_classes = Y.shape[1]

        if order == "random":
            order = random_state.permutation(n_classes)
        else:
            count = np.zeros(n_classes)

            # Get the total number of points to assign
            total = np.arange(1, n_classes + 1)
            total = total[::-1]

            # Count the number of classes at each position for each sample
            binary = [np.bincount(y, None, n_classes + 1) for y in Y]
            binary = np.array(binary)
            binary = binary[:, 1:]

            borda(Y, total, binary, count)
            order = np.argsort(-count)

        return order

    def fit(self, X, Y):
        """Fit the model to the training data and rankings."""
        X, Y = self._validate_data(X, Y, multi_output=True)

        n_samples, n_classes = Y.shape
        self.n_classes_in_ = n_classes

        random_state = check_random_state(self.random_state)

        self.order_ = self._validate_order(Y, self.order, random_state)

        shape = (n_samples, n_classes * (n_classes - 1) // 2)
        X_aug = np.zeros(shape, dtype=int)

        combinations = itertools.combinations(self.order_, 2)
        combinations = list(combinations)

        gaussian = defaultdict(dict)
        categorical = defaultdict(dict)
        self.estimators_ = {"gaussian": gaussian, "categorical": categorical}

        for index, combination in enumerate(combinations):
            f_class, s_class = combination
            f_position, s_position = Y[:, f_class], Y[:, s_class]

            precedes = f_position < s_position
            ties = f_position == s_position
            succedes = f_position > s_position
            condlist = [precedes, ties, succedes]

            length = len(condlist)
            choicelist = np.arange(length)

            y = np.select(condlist, choicelist)

            estimator = clone(self.gaussian_estimator).fit(X, y)
            self.estimators_["gaussian"][combination] = estimator

            if index > 0:
                # Do not require the chain for the first estimator
                X_new = X_aug[:, :index]
                estimator = clone(self.categorical_estimator).fit(X_new, y)
                self.estimators_["categorical"][combination] = estimator

            X_aug[:, index] = y

        # Initialize the rank aggregation algorithm for inference
        self._rank_algorithm.init(n_classes)

        return self

    def predict(self, X):
        """Predict on the testing data using the model."""
        check_is_fitted(self)

        X = self._validate_data(X, reset=False)

        n_samples, n_classes = X.shape[0], self.n_classes_in_

        shape = (n_samples, n_classes * (n_classes - 1) // 2)
        X_aug = np.zeros(shape, dtype=int)

        shape = (n_samples, n_classes, n_classes, 2)
        precedences_matrices = np.zeros(shape)

        shape = (n_samples, n_classes)
        Y_pred = np.zeros(shape, dtype=int)

        combinations = itertools.combinations(self.order_, 2)

        for index, combination in enumerate(combinations):
            f_class, s_class = combination

            estimator = self.estimators_["gaussian"][combination]
            gaussian_jll = estimator._joint_log_likelihood(X)

            if index > 0:
                # Do not require the chain for the first estimator
                X_new = X_aug[:, :index]
                estimator = self.estimators_["categorical"][combination]
                categorical_jll = estimator._joint_log_likelihood(X_new)

                # Substract the empirical logarithm probability for each class
                # since it has been already included by the gaussian estimator
                categorical_jll = categorical_jll - estimator.class_log_prior_
            else:
                categorical_jll = 0

            # Normalize by the joint probability distribution
            jll = gaussian_jll + categorical_jll
            normalization = logsumexp(jll, axis=1)
            probabilities = jll - normalization[:, None]
            probabilities = np.exp(probabilities)

            precedes = probabilities[:, 0]
            ties = probabilities[:, 1]
            succedes = probabilities[:, 2]

            precedences_matrices[:, f_class, s_class, 0] = precedes
            precedences_matrices[:, f_class, s_class, 1] = ties
            precedences_matrices[:, s_class, f_class, 0] = succedes
            precedences_matrices[:, s_class, f_class, 1] = ties

            indexes = np.argmax(jll, axis=1)
            y = estimator.classes_[indexes]

            X_aug[:, index] = y

        aggregate(Y_pred, precedences_matrices, self._rank_algorithm)

        return Y_pred


class BivariatePartialLabelRanker(PartialLabelRankerMixin, BaseEstimator):
    """A partial label ranker that arranges pairs of pairwise preferences."""

    def __init__(self, gaussian_estimator):
        self.gaussian_estimator = gaussian_estimator

    def fit(self, X, Y):
        """Fit the model to the training data and rankings."""
        X, Y = self._validate_data(X, Y, multi_output=True)

        self.n_classes_in_ = Y.shape[1]
        n_classes = self.n_classes_in_

        self.estimators_ = defaultdict(dict)

        iterable = range(self.n_classes_in_)
        combinations = itertools.combinations(iterable, 2)
        combinations = list(combinations)

        for index, f_pair in enumerate(combinations):
            f_f_class, f_s_class = f_pair
            f_f_position, f_s_position = Y[:, f_f_class], Y[:, f_s_class]

            for s_pair in combinations[index + 1:]:
                s_f_class, s_s_class = s_pair
                s_f_position, s_s_position = Y[:, s_f_class], Y[:, s_s_class]

                # Compute the pairs of pairwise preferences among classes
                conditions = (np.less, np.equal, np.greater)
                iterable = itertools.product(conditions, repeat=2)
                iterable = list(iterable)
                length = len(iterable)
                condlist = [f_condition(f_f_position, f_s_position) &
                            s_condition(s_f_position, s_s_position)
                            for f_condition, s_condition, in iterable]

                choicelist = np.arange(length)

                y = np.select(condlist, choicelist)

                estimator = clone(self.gaussian_estimator)
                self.estimators_[f_pair][s_pair] = estimator.fit(X, y)

        # Initialize the rank aggregation algorithm for inference
        self._rank_algorithm.init(n_classes)

        return self
        
    def predict(self, X):
        """Predict on the testing data using the model."""
        check_is_fitted(self)

        X = self._validate_data(X, reset=False)

        n_samples = X.shape[0]

        shape = (n_samples, self.n_classes_in_, self.n_classes_in_, 3)
        likelihood = np.zeros(shape)

        # Hold the likelihood for the pairs of pairwise preferences
        shape = (n_samples, 9)
        auxiliar = np.zeros(shape)

        shape = (n_samples, self.n_classes_in_, self.n_classes_in_, 2)
        precedences_matrices = np.zeros(shape)

        shape = (n_samples, self.n_classes_in_)
        Y_pred = np.zeros(shape, dtype=int)

        combinations = itertools.combinations(range(self.n_classes_in_), 2)
        combinations = list(combinations)

        for index, f_pair in enumerate(combinations):
            f_f_class, f_s_class = f_pair

            for s_pair in combinations[index + 1:]:
                s_f_class, s_s_class = s_pair

                indexes = self.estimators_[f_pair][s_pair].classes_
                estimator = self.estimators_[f_pair][s_pair]
                jll = estimator._joint_log_likelihood(X)
                auxiliar[:, indexes] = np.exp(jll)

                # Marginalize the first and second pair of pairwise preferences
                reshape = auxiliar.reshape(n_samples, 3, 3)
                likelihood[:, f_f_class, f_s_class] += reshape.sum(axis=2)
                likelihood[:, s_f_class, s_s_class] += reshape.sum(axis=1)

        for f_class, s_class in combinations:
            # Normalize by adding the probabilities obtained from all pairs
            normalization = np.sum(likelihood[:, f_class, s_class], axis=1)
            normalization = normalization[:, None]
            probabilities = likelihood[:, f_class, s_class] / normalization

            precedes = probabilities[:, 0]
            ties = probabilities[:, 1]
            succedes = probabilities[:, 2]

            precedences_matrices[:, f_class, s_class, 0] = precedes
            precedences_matrices[:, f_class, s_class, 1] = ties
            precedences_matrices[:, s_class, f_class, 0] = succedes
            precedences_matrices[:, s_class, f_class, 1] = ties

        aggregate(Y_pred, precedences_matrices, self._rank_algorithm)

        return Y_pred
