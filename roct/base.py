from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_array

from groot.util import convert_numpy

import numpy as np

from numbers import Number

import json

import warnings


def check_features_scaled(X):
    min_values = np.min(X, axis=0)
    max_values = np.max(X, axis=0)
    if np.any(min_values < 0) or np.any(max_values) > 1:
        warnings.warn(
            "found feature values outside of the [0, 1] range, "
            "features should be scaled to [0, 1] or ROCT will ignore their "
            "values for splitting."
        )


class BaseOptimalRobustTree(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        max_depth=3,
        attack_model=None,
        add_impossible_combinations=False,
        time_limit=None,
        record_progress=False,
        verbose=False,
    ):
        self.max_depth = max_depth
        self.attack_model = attack_model
        self.add_impossible_combinations = add_impossible_combinations
        self.time_limit = time_limit
        self.record_progress = record_progress
        self.verbose = verbose

        self.optimal_ = False

    def __parse_attack_model(self, attack_model):
        Delta_l = []
        Delta_r = []
        for attack in attack_model:
            if isinstance(attack, Number):
                Delta_l.append(attack)
                Delta_r.append(attack)
            elif isinstance(attack, tuple):
                Delta_l.append(attack[0])
                Delta_r.append(attack[1])
            elif attack == "":
                Delta_l.append(0)
                Delta_r.append(0)
            elif attack == ">":
                Delta_l.append(0)
                Delta_r.append(1)
            elif attack == "<":
                Delta_l.append(1)
                Delta_r.append(0)
            elif attack == "<>":
                Delta_l.append(1)
                Delta_r.append(1)
        return Delta_l, Delta_r

    def fit(self, X, y):
        """
        Fit the optimal robust decision tree on the given dataset.
        """
        check_features_scaled(X)

        self.n_samples_, self.n_features_in_ = X.shape

        # If no attack model is given then train a regular decision tree
        if self.attack_model is None:
            self.attack_model = [0.0] * X.shape[1]

        self.Delta_l, self.Delta_r = self.__parse_attack_model(self.attack_model)
        self.T = (2 ** (self.max_depth + 1)) - 1

        self._fit_solver_specific(X, y)

    def predict_proba(self, X):
        """
        Predict class probabilities of the input samples X.

        The class probability is the fraction of samples of the same class in
        the leaf.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples to predict.

        Returns
        -------
        proba : array of shape (n_samples,)
            The probability for each input sample of being malicious.
        """

        X = check_array(X)

        predictions = []
        for sample in X:
            predictions.append(self.root_.predict(sample))

        return np.array(predictions)

    def predict(self, X):
        """
        Predict the classes of the input samples X.

        The predicted class is the most frequently occuring class label in a
        leaf.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples to predict.

        Returns
        -------
        y : array-like of shape (n_samples,)
            The predicted class labels
        """

        return np.argmax(self.predict_proba(X), axis=1)

    def to_string(self):
        result = ""
        result += f"Parameters: {self.get_params()}\n"

        if hasattr(self, "root_"):
            result += f"Tree:\n{self.root_.pretty_print()}"
        else:
            result += "Tree has not yet been fitted"

        return result

    def to_xgboost_json(self, output_file="tree.json"):
        if hasattr(self, "root_"):
            dictionary, _ = self.root_.to_xgboost_json(0, 0)
        else:
            raise Exception("Tree is not yet fitted")

        if output_file is None:
            return dictionary
        else:
            with open(output_file, "w") as fp:
                # If saving to file then surround dict in list brackets
                json.dump([dictionary], fp, indent=2, default=convert_numpy)

    def prune(self):
        bounds = np.tile(np.array([0, 1], dtype=np.float32), (self.n_features_, 1))

        for _ in range(self.max_depth):
            # Without decision nodes we do not have to prune
            if self.root_.is_leaf():
                break

            self.root_ = self.root_.prune(bounds)
