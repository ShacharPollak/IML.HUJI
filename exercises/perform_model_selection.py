from __future__ import annotations
import numpy as np
import pandas as pd
import sklearn.linear_model
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    noises = np.random.normal(0, noise, n_samples)
    X = np.linspace(-1.2, 2, n_samples)
    f = lambda x: (x + 3) * (x + 2) * (x + 1) * (x - 1) * (x - 2)
    f_x = f(X)
    y = f_x + noises
    X_train, y_train, X_test, y_test = split_train_test(pd.DataFrame(X), pd.Series(y), float(2.0 / 3.0))
    X_train_numpy, y_train_numpy, X_test_numpy, y_test_numpy = X_train.to_numpy().flat, y_train.to_numpy(), \
                                                               X_test.to_numpy().flat, y_test.to_numpy()
    fig = go.Figure([go.Scatter(x=X, y=f_x, mode="markers", name="true model", marker=dict(size=12, color="Black")),
                     go.Scatter(x=list(X_train_numpy), y=list(y_train_numpy), mode="markers", name="train",
                                marker=dict(size=12, color="Blue")),
                     go.Scatter(x=list(X_test_numpy), y=list(y_test_numpy), mode="markers", name="test",
                                marker=dict(size=12, color="Red"))])
    fig.update_layout(
        yaxis_title='y',
        xaxis_title='x',
        title='The True (Noiseless) Model - f(x)=(x+3)(x+2)(x+1)(x-1)(x-2), And The Two Sets - Train And Test Samples.'
              '     num samples: {}, noise: {}'.format(n_samples, noise)
    )
    fig.show()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    train_scores = []
    validation_scores = []
    for k in range(11):
        polynom_model = PolynomialFitting(k)
        avg_train_score, avg_validation_score = cross_validate(polynom_model, np.asarray(X_train_numpy),
                                                               np.asarray(y_train_numpy), mean_square_error)
        train_scores.append(avg_train_score)
        validation_scores.append(avg_validation_score)
    fig = go.Figure([go.Scatter(x=list(range(11)), y=train_scores, mode='markers+lines', name="train score",
                                marker=dict(color="Green"), line=dict(width=10)),
                     go.Scatter(x=list(range(11)), y=validation_scores, mode='markers+lines', name="validation score",
                                marker=dict(color="Blue"), line=dict(width=10))])
    fig.update_layout(
        yaxis_title='MSE',
        xaxis_title='degree',
        title='Average Training And Validation Errors As Function Of Polynomial Degree.'
              '     num samples: {}, noise: {}'.format(n_samples, noise)
    )
    fig.show()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    k_best = int(np.argmin(validation_scores))
    polynom_model = PolynomialFitting(k_best)
    polynom_model.fit(X_train_numpy, y_train_numpy)
    test_error = round(polynom_model.loss(X_test_numpy, y_test_numpy), 2)
    print("best k: {}, test error: {}".format(k_best, test_error))


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    X, y = datasets.load_diabetes(return_X_y=True)
    X, y = pd.DataFrame(X), pd.Series(y)
    X_train = X[:n_samples]
    y_train = y[:n_samples]
    X_test = X[n_samples:]
    y_test = y[n_samples:]

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    ridge_train_errors, ridge_validation_errors, lasso_train_errors, lasso_validation_errors = [], [], [], []
    ridge_lam_range = np.linspace(0, 2.6, n_evaluations)
    lasso_lam_range = np.linspace(0, 2.6, n_evaluations)
    for lam in ridge_lam_range:
        ridge = RidgeRegression(lam, True)
        train_err, valid_error = cross_validate(ridge, X_train, y_train, mean_square_error, 5)
        ridge_train_errors.append(train_err)
        ridge_validation_errors.append(valid_error)
    for lam in lasso_lam_range:
        lasso = Lasso(alpha=lam, fit_intercept=True)
        train_err, valid_error = cross_validate(lasso, X_train, y_train, mean_square_error, 5)
        lasso_train_errors.append(train_err)
        lasso_validation_errors.append(valid_error)

    fig = go.Figure([go.Scatter(x=ridge_lam_range, y=ridge_train_errors, mode='markers+lines', name="train score",
                                marker=dict(color="Green"), line=dict(width=10)),
                     go.Scatter(x=ridge_lam_range, y=ridge_validation_errors, mode='markers+lines', name="validation "
                                                                                                         "score",
                                marker=dict(color="Blue"), line=dict(width=10))])
    fig.update_layout(
        yaxis_title='MSE',
        xaxis_title='lamda',
        title='Average Training And Validation Errors As Function Of The Regularization Parameter Lamda (Ridge)'
    )
    fig.show()
    fig = go.Figure([go.Scatter(x=lasso_lam_range, y=lasso_train_errors, mode='markers+lines', name="train score",
                                marker=dict(color="Green"), line=dict(width=10)),
                     go.Scatter(x=lasso_lam_range, y=lasso_validation_errors, mode='markers+lines', name="validation "
                                                                                                         "score",
                                marker=dict(color="Blue"), line=dict(width=10))])
    fig.update_layout(
        yaxis_title='MSE',
        xaxis_title='lamda',
        title='Average Training And Validation Errors As Function Of The Regularization Parameter Lamda (Lasso)'
    )
    fig.show()

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    best_lamda_ridge = ridge_lam_range[np.argmin(ridge_validation_errors)]
    best_lamda_lasso = lasso_lam_range[np.argmin(lasso_validation_errors)]
    ridge = RidgeRegression(best_lamda_ridge)
    lasso = Lasso(best_lamda_lasso)
    lin_reg = LinearRegression()
    ridge.fit(X_train, y_train)
    lasso.fit(X_train, y_train)
    lin_reg.fit(X_train, y_train)
    ridge_test_error = ridge.loss(X_test, y_test)
    lasso_predict = lasso.predict(X_test)
    lasso_test_error = mean_square_error(y_test, lasso_predict)
    lin_reg_test_error = lin_reg.loss(X_test, y_test)
    print("best lamda for ridge: {}, ridge test error: {}".format(best_lamda_ridge, ridge_test_error))
    print("best lamda for lasso: {}, lasso test error: {}".format(best_lamda_lasso, lasso_test_error))
    print("linear regression test error: {}".format(lin_reg_test_error))


if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree()
    select_polynomial_degree(noise=0)
    select_polynomial_degree(n_samples=1500, noise=10)
    select_regularization_parameter()
