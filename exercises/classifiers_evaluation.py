from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px
from plotly.subplots import make_subplots
from math import atan2, pi

pio.templates.default = "simple_white"


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "linearly_separable.npy"),
                 ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        X, y = load_dataset("../datasets/" + f)

        # Fit Perceptron and record loss in each fit iteration
        losses = []

        def loss_callback(fit: Perceptron, X_: np.ndarray, y_: int):
            losses.append(fit.loss(X, y))

        perceptron = Perceptron(callback=loss_callback)
        perceptron.fit(X, y)

        # Plot figure
        fig = px.line(losses,
                      title="Perceptron Algorithm's Training Loss As Function Of Training Iterations Over "
                            "{} Dataset".format(n),
                      x=range(len(losses)),
                      y=losses,
                      labels={'x': 'number of iterations', 'y': 'training loss'})
        fig.show()


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, showlegend=False, mode="lines", marker_color="black")


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        X, y = load_dataset("../datasets/" + f)

        # Fit models and predict over training set
        lda = LDA()
        gaussian = GaussianNaiveBayes()
        lda.fit(X, y)
        gaussian.fit(X, y)
        lda_predict = lda.predict(X)
        guassian_predict = gaussian.predict(X)

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        # Create subplots
        from IMLearn.metrics import accuracy
        lda_accuracy = round(accuracy(y, lda_predict), 2)
        guassian_accuracy = round(accuracy(y, guassian_predict), 2)
        lda_ellipses = []
        gaussian_ellipses = []
        for i in range(len(lda.mu_)):
            lda_ellipses.append(get_ellipse(lda.mu_[i], lda.cov_))
            gaussian_ellipses.append(get_ellipse(gaussian.mu_[i], np.diag(gaussian.vars_[i])))

        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=("Guassian Naive Bayes (Accuracy = {})".format(guassian_accuracy), "LDA (Accuracy = {})".format(lda_accuracy)))

        fig.add_trace(
            go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers', showlegend=False, marker=dict(size=10,
                                                                                           color=lda_predict,
                                                                                           line=dict(width=2,
                                                                                                color='DarkSlateGrey'),
                                                                                           symbol=y)),
            row=1, col=2
        )
        for i in range(len(lda_ellipses)):
            fig.add_trace(lda_ellipses[i], row=1, col=2)
            fig.add_trace(go.Scatter(x=[lda.mu_[i][0]], y=[lda.mu_[i][1]], mode='markers', showlegend=False,
                                     marker_color="Black",
                                     marker=dict(color="Black", line=dict(width=2,color='Black'),
                                                 symbol='x-thin', line_width=7, size=18)), row=1, col=2)

        fig.add_trace(
            go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers', showlegend=False, marker=dict(size=10,
                                                                                           color=guassian_predict,
                                                                                           line=dict(width=2,
                                                                                                color='DarkSlateGrey'),
                                                                                           symbol=y)),
            row=1, col=1
        )

        fig.update_layout(title_text="LDA prediction (right) and Guassian Naive Bayes "
                                     "Prediction (left) on {} Dataset".format(f))
        for i in range(len(gaussian_ellipses)):
            fig.add_trace(gaussian_ellipses[i], row=1, col=1)
            fig.add_trace(go.Scatter(x=[gaussian.mu_[i][0]], y=[gaussian.mu_[i][1]], mode='markers', showlegend=False,
                                     marker_color="Black",
                                     marker=dict(color="Black", symbol='x-thin',
                                                 line=dict(width=2,color='Black'), line_width=7, size=18)), row=1,
                          col=1)
        fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    # run_perceptron()
    compare_gaussian_classifiers()
