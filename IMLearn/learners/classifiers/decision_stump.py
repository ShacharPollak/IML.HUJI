from __future__ import annotations
from typing import Tuple, NoReturn
from ...base import BaseEstimator
import numpy as np
from itertools import product

from ...metrics import misclassification_error


class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is about the threshold
    """

    def __init__(self) -> DecisionStump:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a decision stump to the given data

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        n_features = X.shape[1]
        best_loss = -1
        for j in range(n_features):
            t_plus, loss_plus = self._find_threshold(X[:, j], y, 1)
            t_minus, loss_minus = self._find_threshold(X[:, j], y, -1)
            if best_loss == -1 or loss_plus < best_loss or loss_minus < best_loss:
                if loss_plus < loss_minus:
                    best_loss = loss_plus
                    self.threshold_ = t_plus
                    self.sign_ = 1
                    self.j_ = j
                else:
                    best_loss = loss_minus
                    self.threshold_ = t_minus
                    self.sign_ = -1
                    self.j_ = j

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """
        values = X[:, self.j_]
        return np.where(values >= self.threshold_, self.sign_, -self.sign_)

    def _find_threshold(self, values: np.ndarray, labels: np.ndarray, sign: int) -> Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        labels: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Misclassificaiton error of returned threshold

        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """
        best_error = -1
        best_threshold = 0
        for val in np.r_[values,np.asarray([np.inf])]:
            predicted = np.where(values >= val, sign, -sign)
            indicator = np.where(predicted != np.sign(labels), 1, 0)
            error = np.abs(labels) @ indicator
            if best_error == -1 or error < best_error:
                best_error = error
                best_threshold = val

        return best_threshold, best_error

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        predicted = self._predict(X)
        indicator = np.where(predicted != np.sign(y), 1, 0)
        result = np.abs(y) @ indicator
        return result
        # return sum([abs(y[i]) for i in range(len(y)) if np.sign(y[i]) != predicted[i]])
