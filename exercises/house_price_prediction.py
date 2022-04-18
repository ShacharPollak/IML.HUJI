import pandas

from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    # read file:
    df = pandas.read_csv(filename)
    # drop columns:
    df = df.drop(columns=['id', 'date', 'lat', 'long'])
    # remove duplicates:
    df = df.drop_duplicates()
    # fill missing values:
    df['waterfront'] = df['waterfront'].fillna(0)
    df['view'] = df['view'].fillna(0)
    df['yr_renovated'] = df['yr_renovated'].fillna(0)
    df['sqft_basement'] = df['sqft_basement'].fillna(0)
    # drop rows with missing values:
    df = df.dropna(axis=0, how='any')
    # convert categorical column to dummies:
    df['zipcode'] = df['zipcode'].astype(int)
    df = pd.get_dummies(df, columns=['zipcode'])
    # remove weird samples:
    df = df[df['bedrooms'] < 15]
    df = df[df['price'] > 0]
    # split to X and y:
    y = df['price']
    X = df.drop(columns=['price'])
    return X, y


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    stdy = np.std(y)
    for feature in X:
        x = X[feature]
        cov = np.cov(x, y)[0][1]
        stdx = np.std(x)
        correlation = cov / (stdx * stdy)
        go.Figure([go.Scatter(x=x, y=y, mode='markers')],
            layout=go.Layout(title="Price with relation to {}. ".format(feature)+"Pearson Correlation = {}".format(
                round(correlation,4)),
                                   xaxis_title="{}".format(feature),
                                   yaxis_title="price",
                                   height=500, width=1000)).write_image("{}/correlation_{}.png".format(output_path,
                                                   feature))
if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    X, y = load_data("../datasets/house_prices.csv")

    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(X, y)

    # Question 3 - Split samples into training- and testing sets.
    train_X, train_y, test_X, test_y = split_train_test(X, y)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    lin_reg = LinearRegression()
    percentage = []
    means = []
    c_up = []
    c_down = []
    for i in range(10, 101):
        loss_samples = []
        frac = float(i / 100)
        for j in range(10):
            sample_x = train_X.sample(frac=frac)
            sample_y = train_y.reindex_like(sample_x)
            lin_reg.fit(np.asarray(sample_x), np.asarray(sample_y))
            loss = lin_reg.loss(np.asarray(test_X), np.asarray(test_y))
            loss_samples.append(loss)
        mean = np.mean(loss_samples)
        std = np.std(loss_samples)
        confidence_upper = mean + 2 * std
        confidence_lower = mean - 2 * std
        percentage.append(i)
        means.append(mean)
        c_up.append(confidence_upper)
        c_down.append(confidence_lower)

    fig = go.Figure([go.Scatter(x=percentage, y=means, mode="markers+lines", name="loss", line=dict(
        dash="dash"),
                marker=dict(color="green", opacity=.7)),
     go.Scatter(x=percentage, y=c_up, fill=None, mode="lines", line=dict(color="lightgrey"),
                showlegend=False),
     go.Scatter(x=percentage, y=c_down, fill='tonexty', mode="lines", line=dict(color="lightgrey"),
                showlegend=False)])
    fig.update_layout(
        yaxis_title='Average loss',
        xaxis_title='percentage of sample',
        title='Average Loss As Function Of Training Size With Error Ribbon Of Size (mean-2*std, mean+2*std)',
        hovermode="x"
    )
    fig.show()



