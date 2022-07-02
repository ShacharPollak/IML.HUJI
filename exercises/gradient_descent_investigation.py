import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type

from sklearn.metrics import roc_curve, auc

from IMLearn import BaseModule
from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from IMLearn.desent_methods.modules import L1, L2
from IMLearn.learners.classifiers.logistic_regression import LogisticRegression
from IMLearn.metrics import misclassification_error
from IMLearn.model_selection import cross_validate
from IMLearn.utils import split_train_test

import plotly.graph_objects as go

from utils import custom


def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm

    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted

    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path

    title: str, default=""
        Setting details to add to plot title

    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange] over which regularization path is shown

    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """

    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange, density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1], mode="markers+lines",
                                 marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=f"GD Descent Path {title}"))


def get_gd_state_recorder_callback() -> Tuple[Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's value and parameters at each iteration

    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, recoding the objective's value and parameters
        at each iteration of the algorithm

    values: List[np.ndarray]
        Recorded objective values

    weights: List[np.ndarray]
        Recorded parameters
    """
    ret_values = []
    ret_weights = []

    def new_callback(solver, weights, val, grad, t, eta, delta):
        ret_values.append(val)
        ret_weights.append(weights)

    return new_callback, ret_values, ret_weights


def compare_fixed_learning_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                 etas: Tuple[float] = (1, .1, .01, .001)):
    for eta in etas:
        f_l1 = L1(init)
        f_l2 = L2(init)
        X = np.zeros(1)
        y = np.zeros(1)
        fixed_rate = FixedLR(eta)
        callback_l1, values_l1, weights_l1 = get_gd_state_recorder_callback()
        l1_descent = GradientDescent(learning_rate=fixed_rate, max_iter=1000, callback=callback_l1)
        callback_l2, values_l2, weights_l2 = get_gd_state_recorder_callback()
        l2_descent = GradientDescent(learning_rate=fixed_rate, max_iter=1000, callback=callback_l2)
        sol_l1 = l1_descent.fit(f_l1, X, y)
        sol_l2 = l2_descent.fit(f_l2, X, y)
        path_plot_l1 = plot_descent_path(L1, np.asarray(weights_l1), "The Descent Path Of L1 With Eta={}".format(eta))
        path_plot_l1.show()
        path_plot_l2 = plot_descent_path(L2, np.asarray(weights_l2), "The Descent Path Of L2 With Eta={}".format(eta))
        path_plot_l2.show()

        print("lowest loss of L1 with eta = {} is: {}".format(eta, np.min(values_l1)))
        print("lowest loss of L2 with eta = {} is: {}".format(eta, np.min(values_l2)))

        fig = go.Figure([go.Scatter(x=list(range(1, len(values_l1) + 1)), y=values_l1, mode="markers+lines",
                                    name="convergence rate", marker=dict(color="green"))])
        fig.update_layout(
            yaxis_title='value',
            xaxis_title='iteration',
            title='Convergence Rate As Function Of Iteration, L1 Norm, Eta={}'.format(eta),
            hovermode="x"
        )
        fig.show()

        fig = go.Figure([go.Scatter(x=list(range(1, len(values_l2) + 1)), y=values_l2, mode="markers+lines",
                                    name="convergence rate", marker=dict(color="green"))])
        fig.update_layout(
            yaxis_title='value',
            xaxis_title='iteration',
            title='Convergence Rate As Function Of Iteration, L2 Norm, Eta={}'.format(eta),
            hovermode="x"
        )
        fig.show()


def compare_exponential_decay_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                    eta: float = .1,
                                    gammas: Tuple[float] = (.9, .95, .99, 1)):
    # Optimize the L1 objective using different decay-rate values of the exponentially decaying learning rate
    colors = ['red', 'green', 'blue', 'yellow']
    fig = go.Figure()
    path_plot_l1 = go.Figure()
    for i, gamma in enumerate(gammas):
        f_l1 = L1(init)
        X = np.zeros(1)
        y = np.zeros(1)
        lr_rate = ExponentialLR(eta, gamma)
        callback_l1, values_l1, weights_l1 = get_gd_state_recorder_callback()
        l1_grad = GradientDescent(learning_rate=lr_rate, callback=callback_l1)
        sol = l1_grad.fit(f_l1, X=X, y=y)
        print("lowest L1 norm achieved using exp decay with gamma = {} is: {}".format(gamma, np.min(values_l1)))
        fig.add_traces([go.Scatter(x=list(range(1, len(values_l1) + 1)), y=values_l1, mode="markers+lines",
                                   name="gamma {}".format(gamma), marker=dict(color=colors[i]))])
        if gamma == 0.95:
            path_plot_l1 = plot_descent_path(L1, np.asarray(weights_l1),
                                             "The Descent Path Of L1 With gamma={}".format(gamma))

    # Plot algorithm's convergence for the different values of gamma
    fig.update_layout(
        yaxis_title='value',
        xaxis_title='iteration',
        title='Convergence Rate As Function Of Iteration, L2 Norm, Eta={}, Various gammas'.format(eta),
        hovermode="x")
    fig.show()

    # Plot descent path for gamma=0.95
    path_plot_l1.show()


def load_data(path: str = "../datasets/SAheart.data", train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train- and test portion

    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset

    train_portion: float, default=0.8
        Portion of dataset to use as a training set

    Return:
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd, train_portion)


def fit_logistic_regression():
    # Load and split SA Heard Disease dataset
    X_train, y_train, X_test, y_test = load_data()
    X_train, y_train, X_test, y_test = np.asarray(X_train), np.asarray(y_train), np.asarray(X_test), np.asarray(y_test)
    # Plotting convergence rate of logistic regression over SA heart disease data
    log_reg = LogisticRegression(solver=GradientDescent(FixedLR(1e-4), max_iter=20000))
    log_reg.fit(X_train, y_train)
    y_prob = log_reg.predict_proba(X_train)
    fpr, tpr, thresholds = roc_curve(y_train, y_prob)

    fig = go.Figure(
        data=[go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="black", dash='dash'),
                         name="Random Class Assignment"),
              go.Scatter(x=fpr, y=tpr, mode='markers+lines', text=thresholds, name="LogisticRegression Assignment",
                         showlegend=False, marker_size=5,
                         marker_color=custom[-1],  # c[1][1]
                         hovertemplate="<b>Threshold:</b>%{text:.3f}<br>FPR: %{x:.3f}<br>TPR: %{y:.3f}")],
        layout=go.Layout(title=rf"$\text{{ROC Curve Of Fitted Model - AUC}}={auc(fpr, tpr):.6f}$",
                         xaxis=dict(title=r"$\text{False Positive Rate (FPR)}$"),
                         yaxis=dict(title=r"$\text{True Positive Rate (TPR)}$")))
    fig.show()
    alphas = tpr - fpr
    alpha_star = thresholds[np.argmax(alphas)]
    print("optimal ROC value achieved by alpha = {}".format(alpha_star))
    log_reg = LogisticRegression(solver=GradientDescent(FixedLR(1e-4), max_iter=20000), alpha=alpha_star)
    log_reg.fit(X_train, y_train)
    test_error = log_reg.loss(X_test, y_test)
    print("model test error with best alpha: {}".format(test_error))

    # Fitting l1- and l2-regularized logistic regression models, using cross-validation to specify values
    # of regularization parameter
    lamdas = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
    for regularization in ["l1", "l2"]:
        train_scores, validation_scores = [], []
        for lamda in lamdas:
            log_reg = LogisticRegression(solver=GradientDescent(learning_rate=FixedLR(1e-4), max_iter=20000),
                                         penalty=regularization, lam=lamda, alpha=0.5)
            train_score, validation_score = cross_validate(log_reg, X_train, y_train, misclassification_error)
            train_scores.append(train_score)
            validation_scores.append(validation_score)
        lamda_star = lamdas[np.argmin(validation_scores)]
        log_reg = LogisticRegression(solver=GradientDescent(learning_rate=FixedLR(1e-4), max_iter=20000),
                                     penalty=regularization, lam=lamda_star, alpha=0.5)
        log_reg.fit(X_train, y_train)
        test_error = log_reg.loss(X_test, y_test)
        print("({} regularized) value of lamda selected: {}, model test error: {}".format(regularization, lamda_star,
                                                                                          test_error))


if __name__ == '__main__':
    np.random.seed(0)
    compare_fixed_learning_rates()
    compare_exponential_decay_rates()
    fit_logistic_regression()
