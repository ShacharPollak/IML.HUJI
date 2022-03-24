import numpy.random
import plotly.data
from plotly.subplots import make_subplots

from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

DEFAULT_GRAPH_HEIGHT = 600

DEFAULT_NUM_OF_SAMPLES = 1000

pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    samples = np.random.normal(10, 1, DEFAULT_NUM_OF_SAMPLES)
    uni_gau = UnivariateGaussian()
    uni_gau.fit(samples)
    print("(" + str(uni_gau.mu_) + ", " + str(uni_gau.var_) + ")")

    # Question 2 - Empirically showing sample mean is consistent
    distance_mu = np.zeros(100)
    ms = np.linspace(10, DEFAULT_NUM_OF_SAMPLES, 100).astype(int)
    i = 0
    for m in ms:
        cur_samples = samples[0:m]
        uni_gau.fit(cur_samples)
        distance_mu[i] = np.absolute(uni_gau.mu_ - 10)
        i += 1
    go.Figure([go.Scatter(x=ms, y=distance_mu, mode='markers+lines', name=r'$mu$')],
              layout=go.Layout(title=r"$\text{Absolute Distance Between The Estimated And True Value Of The "
                                     r"Expectation As Function Of Sample Size}$",
                               xaxis_title="$\\text{Number of samples}$",
                               yaxis_title="r$\\text{Absolute distance between the estimated and true value of the "
                                           "expectation}$",
                               height=DEFAULT_GRAPH_HEIGHT)).show()

    # Question 3 - Plotting Empirical PDF of fitted model
    pdf = uni_gau.pdf(samples)
    go.Figure([go.Scatter(x=samples, y=pdf, mode='markers')],
              layout=go.Layout(title=r"$\text{PDF of the Previously Drawn Samples}$",
                               xaxis_title="$\\text{Ordered sample values}$",
                               yaxis_title="r$\\text{PDF of the samples}$",
                               height=DEFAULT_GRAPH_HEIGHT)).show()

def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mu = [0, 0, 4, 0]
    cov = [[1, 0.2, 0, 0.5], [0.2, 2, 0, 0], [0, 0, 1, 0], [0.5, 0, 0, 1]]
    samples = numpy.random.multivariate_normal(mu, cov, DEFAULT_NUM_OF_SAMPLES)
    mul_gau = MultivariateGaussian()
    mul_gau.fit(samples)
    print(mul_gau.mu_)
    print(mul_gau.cov_)

    # Question 5 - Likelihood evaluation
    lin_space = np.linspace(-10, 10, 200)
    result = []
    cov = np.asarray(cov)
    for f1 in lin_space:
        cur = []
        for f3 in lin_space:
            mu = np.transpose([f1, 0, f3, 0])
            log_likelihood = MultivariateGaussian.log_likelihood(mu, cov, samples)
            cur.append(log_likelihood)
        result.append(cur)
    result = np.asarray(result)
    go.Figure([go.Heatmap(x=lin_space, y=lin_space, z=result, colorscale='Viridis')],
               layout=go.Layout(title=r"$\text{Log-Likelihood As Function Of mu = [f1,0,f3,0]}$",
               xaxis_title="$\\text{f3}$",
               yaxis_title="r$\\text{f1}$",
               height=DEFAULT_GRAPH_HEIGHT*1.4,
               width=DEFAULT_GRAPH_HEIGHT*1.4)).show()

    # Question 6 - Maximum likelihood
    max_index = result.argmax()
    row = int(max_index / 200)
    col = max_index % 200
    print("f1: " + str(round(lin_space[row], 3)))
    print("f3: " + str(round(lin_space[col], 3)))

if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
    # for quiz:
    # q3 = np.array([1, 5, 2, 3, 8, -4, -2, 5, 1, 10, -10, 4, 5, 2, 7, 1, 1, 3, 2, -1, -3, 1, -4, 1, 2, 1,
    #       -4, -4, 1, 3, 2, 6, -6, 8, 3, -6, 4, 1, -2, 3, 1, 4, 1, 4, -2, 3, -1, 0, 3, 5, 0, -2])
    # print(UnivariateGaussian.log_likelihood(1, 1, q3))
    # print(UnivariateGaussian.log_likelihood(10, 1, q3))
