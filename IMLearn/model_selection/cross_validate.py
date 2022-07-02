from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable

import numpy
import numpy as np
import pandas as pd

from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    X_chunks = np.array_split(X, cv)
    y_chunks = np.array_split(y, cv)
    chunk_size = len(X_chunks[0])
    train_scores = []
    validation_scores = []
    for i in range(cv):
        X_all_but_i = np.asarray(np.concatenate([X[0:i*chunk_size], X[(i+1)*chunk_size:]], axis=0))
        y_all_but_i = np.asarray(np.concatenate([y[0:i*chunk_size], y[(i+1)*chunk_size:]], axis=0))
        estimator.fit(X_all_but_i, y_all_but_i)
        prediction_on_train = estimator.predict(X_all_but_i)
        prediction_on_validation = estimator.predict(X_chunks[i])
        train_score = scoring(y_all_but_i, prediction_on_train)
        validation_score = scoring(y_chunks[i], prediction_on_validation)
        train_scores.append(train_score)
        validation_scores.append(validation_score)
    avg_train_score = np.average(np.asarray(train_scores))
    avg_validation_score = np.average(np.asarray(validation_scores))

    return avg_train_score, avg_validation_score
