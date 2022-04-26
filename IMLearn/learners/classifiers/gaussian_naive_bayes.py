from typing import NoReturn

from . import LDA
from ...base import BaseEstimator
import numpy as np

from ...metrics import misclassification_error


class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """
    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        lda = LDA()
        lda.fit(X, y)
        self.classes_ = lda.classes_
        k = len(self.classes_)
        n_features = X.shape[1]
        self.mu_ = lda.mu_
        self.pi_ = lda.pi_
        self.vars_ = np.asarray([np.var(X[y == c], axis=0, ddof=1) for c in self.classes_])
        # self.vars_ = np.zeros((k, n_features))
        # for i, c in enumerate(self.classes_):
        #     X_c = X[y == c]
        #     nk = X_c.shape[0]
        #     for j in range(n_features):
        #         self.vars_[i][j] = (1/(nk - 1)) * np.sum(X[y == c][j] - self.mu_[i][j])

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

        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        result = np.zeros((n_samples, n_classes))
        for sample in range(n_samples):
            for k in range(len(self.classes_)):
                for feature in range(X.shape[1]):
                    result[sample][k] += np.log(self.pi_[k]) - 0.5 * (((X[sample][feature] - self.mu_[k][feature]) **
                                                                       2) /
                                                                     self.vars_[k][feature])
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
