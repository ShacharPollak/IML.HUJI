from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
import numpy as np
from typing import Tuple
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px
from plotly.subplots import make_subplots

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
    data = np.load("../datasets/" + filename)
    y = data[:, 2]
    X = data[:, :2]
    return X, y


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "linearly_separable.npy"),
                 ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        X, y = load_dataset(f)

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


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        X, y = load_dataset(f)

        # Fit models and predict over training set
        lda = LDA()
        guassian = GaussianNaiveBayes()
        lda.fit(X, y)
        guassian.fit(X, y)
        lda_predict = lda.predict(X)
        guassian_predict = guassian.predict(X)

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        from IMLearn.metrics import accuracy
        lda_accuracy = round(accuracy(y, lda_predict), 2)
        guassian_accuracy = round(accuracy(y, guassian_predict), 2)

        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=("LDA (Accuracy = {})".format(lda_accuracy),
                                            "Guassian Naive Bayes (Accuracy = {})".format(guassian_accuracy)))

        fig.add_trace(
            go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers', showlegend=False, marker=dict(size=10,
                                                                                           color=lda_predict,
                                                                                           line=dict(width=2,
                                                                                                     color='DarkSlateGrey'),
                                                                                           symbol=y)),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers', showlegend=False, marker=dict(size=10,
                                                                                           color=guassian_predict,
                                                                                           line=dict(width=2,
                                                                                                     color='DarkSlateGrey'),
                                                                                           symbol=y)),
            row=1, col=2
        )

        fig.update_layout(title_text="LDA prediction (left) and Guassian Naive Bayes "
                                     "Prediction (right) on {} Dataset".format(f))
        fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    # run_perceptron()
    compare_gaussian_classifiers()
