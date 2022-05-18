import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from IMLearn.metrics import accuracy
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    adaboost = AdaBoost(lambda: DecisionStump(), n_learners)
    adaboost.fit(train_X, train_y)
    train_errors = [adaboost.partial_loss(train_X, train_y, i) for i in range(1, n_learners+1)]
    test_errors = [adaboost.partial_loss(test_X, test_y, i) for i in range(1, n_learners+1)]
    fig = go.Figure([go.Scatter(x=list(range(n_learners)), y=train_errors, mode="markers+lines", name="train error",
                                marker=dict(color="green")),
                     go.Scatter(x=list(range(n_learners)), y=test_errors, mode="markers+lines", name="test error",
                                marker=dict(color="red"))])
    fig.update_layout(
        yaxis_title='Error',
        xaxis_title='number of learners',
        title='Adaboost Error As Function Of Number Of Learners On Train Set And On Test Set',
        hovermode="x"
    )
    fig.show()

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])
    symbols = np.array(["circle", "x"])

    def _decision_surface(partial_predict, xrange, yrange, num, density=120, dotted=False, colorscale=custom,
                          showscale=True):
        xrange, yrange = np.linspace(*xrange, density), np.linspace(*yrange, density)
        xx, yy = np.meshgrid(xrange, yrange)
        pred = partial_predict(np.c_[xx.ravel(), yy.ravel()], num)

        if dotted:
            return go.Scatter(x=xx.ravel(), y=yy.ravel(), opacity=1, mode="markers",
                              marker=dict(color=pred, size=1, colorscale=colorscale, reversescale=False),
                              hoverinfo="skip", showlegend=False)
        return go.Contour(x=xrange, y=yrange, z=pred.reshape(xx.shape), colorscale=colorscale, reversescale=False,
                          opacity=.7, connectgaps=True, hoverinfo="skip", showlegend=False, showscale=showscale)
    fig = make_subplots(rows=2, cols=2, subplot_titles=[rf"$\textbf{{{m}}}$" for m in T],
                        horizontal_spacing=0.01, vertical_spacing=.03)
    for i, m in enumerate(T):
        fig.add_traces([_decision_surface(adaboost.partial_predict, lims[0], lims[1], m, showscale=False),
                        go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                                   marker=dict(color=test_y.astype(int), symbol=symbols[test_y.astype(int)],
                                               colorscale=[custom[0], custom[-1]],
                                               line=dict(color="black", width=1)))],
                       rows=(i // 2) + 1, cols=(i % 2) + 1)

    fig.update_layout(title=rf"$\textbf{{Decision Boundaries Obtained By Using The Ensemble Up To Iteration 5, 50, 100 And 250}}$", margin=dict(t=100)) \
        .update_xaxes(visible=False).update_yaxes(visible=False)
    fig.show()

    # Question 3: Decision surface of best performing ensemble
    best_ensemble = int(np.argmin(test_errors))+1
    predicted = adaboost.partial_predict(test_X, best_ensemble)
    acc = accuracy(test_y, predicted)

    fig = make_subplots(rows=1, cols=1)
    fig.add_traces([_decision_surface(adaboost.partial_predict, lims[0], lims[1], best_ensemble, showscale=False),
                    go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                               marker=dict(color=test_y.astype(int), symbol=symbols[test_y.astype(int)],
                                           colorscale=[custom[0], custom[-1]],
                                           line=dict(color="black", width=1)))],
                   rows=1, cols=1)
    fig.update_layout(
        title="Decision Surface Obtained By Best Ensemble ({}) With {} Accuracy".format(best_ensemble, acc),
        margin=dict(t=100)) \
        .update_xaxes(visible=False).update_yaxes(visible=False)
    fig.show()

    # Question 4: Decision surface with weighted samples
    weights = adaboost.D_/np.max(adaboost.D_) * 5
    predicted = adaboost.partial_predict(test_X, best_ensemble)
    acc = accuracy(test_y, predicted)

    fig = make_subplots(rows=1, cols=1)
    fig.add_traces([decision_surface(adaboost.predict, lims[0], lims[1], showscale=False),
                    go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode="markers", showlegend=False,
                               marker=dict(size=weights, color=train_y.astype(int), symbol=symbols[train_y.astype(int)],
                                           colorscale=[custom[0], custom[-1]],
                                           line=dict(color="black", width=1)))],
                   rows=1, cols=1)
    fig.update_layout(
        title="Decision Surface Obtained By Full Ensemble, Point Size Proportional To Weight".format(best_ensemble, acc),
        margin=dict(t=100)) \
        .update_xaxes(visible=False).update_yaxes(visible=False)
    fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(0)
    fit_and_evaluate_adaboost(0.4)
