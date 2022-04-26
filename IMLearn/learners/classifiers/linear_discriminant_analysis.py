from typing import NoReturn

from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv

from ...metrics import misclassification_error


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
    """

    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.fitted_ = True
        self.classes_, nk = np.unique(y, return_counts=True)
        k = len(self.classes_)
        m = X.shape[0]
        n_features = X.shape[1]
        self.mu_ = np.asarray([np.mean(X[y == c], axis=0) for c in self.classes_])
        y = y.astype(int)
        self.cov_ = np.zeros((n_features, n_features))
        for i in range(m):
            x_mu = X[i] - self.mu_[y[i]]
            x_mu_multiply = np.outer(x_mu, x_mu)
            self.cov_ += (1.0 / (m-k)) * x_mu_multiply
        self._cov_inv = inv(self.cov_)
        self.pi_ = nk / m

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        posterior_probability = self.likelihood(X)
        result = np.asarray([np.argmax(prob) for prob in posterior_probability])
        return result

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")

        result = []
        for i in range(X.shape[0]):
            cur = []
            for k in range(len(self.classes_)):
                prob = np.log(self.pi_[k]) + X[i].T @ self._cov_inv @ self.mu_[k] - 0.5 * self.mu_[k].T @ self._cov_inv\
                       @ self.mu_[k]
                cur.append(prob)
            result.append(np.asarray(cur))
        result = np.asarray(result)
        return result

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
        predicted = self.predict(X)
        return misclassification_error(y, predicted)
