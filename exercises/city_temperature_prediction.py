import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    # read data:
    df = pd.read_csv(filename, parse_dates=['Date'])
    # remove duplicates:
    df = df.drop_duplicates()
    # drop rows with missing values:
    df = df.dropna(axis=0, how='any')
    # remove weird samples:
    df = df[df['Temp'] > -15]
    # add day of year:
    df["DayOfYear"] = [x.day_of_year for x in df['Date']]

    # make temp the response:
    df.insert(df.columns.size-1, column='Temp', value=df.pop('Temp'))

    return df


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    dataset = load_data("../datasets/City_Temperature.csv")

    # Question 2 - Exploring data for specific country
    subset = dataset[dataset['Country'] == 'Israel']
    subset['Year'] = subset['Year'].astype(str)
    fig = px.scatter(subset, title="Temperature As Function of DayOfYear (Israel)", x="DayOfYear", y="Temp",
                     color="Year")
    fig.show()

    by_month = subset.groupby('Month').agg({'Temp': np.std})
    fig = px.bar(by_month, title="Standard Deviation of Daily Temperatures of Of Each Month", x=range(1, 13), y="Temp",
                 labels={'x': 'Month', 'Temp': 'Standard Deviation of Daily Temperatures'})
    fig.show()

    # Question 3 - Exploring differences between countries
    group = dataset.groupby(['Country', 'Month']).agg({'Temp': [np.mean, np.std]})
    group.columns = ['mean', 'std']
    group = group.reset_index()
    fig = px.line(group, x='Month', y='mean', line_group='Country', error_y='std', color='Country',
                  title="Average Monthly Temperature (with error bars - using std) By Country",
                  labels={'mean': 'Average Monthly Temperature'})
    fig.show()

    # Question 4 - Fitting model for different values of `k`
    y = subset['Temp']
    X = subset['DayOfYear']
    train_X, train_y, test_X, test_y = split_train_test(X, y)
    losses = []
    for k in range(1, 11):
        poly_fitting = PolynomialFitting(k)
        poly_fitting.fit(np.asarray(train_X), np.asarray(train_y))
        loss = poly_fitting.loss(np.asarray(test_X), np.asarray(test_y))
        print("test error recorded for k=" + str(k) + ": " + str(round(loss, 2)))
        losses.append(round(loss, 2))
    fig = px.bar(losses, title="Test Error Recorded for Each Degree k of the Polynom", x=range(1, 11), y=losses,
                 labels={'x': 'k - degree of polynom', 'y': 'test error'})
    fig.show()

    # Question 5 - Evaluating fitted model on different countries
    poly = PolynomialFitting(5)
    poly.fit(np.asarray(X), np.asarray(y))
    losses = []
    countries = [country for country in dataset['Country'].unique() if country != "Israel"]
    for country in countries:
        data = dataset[dataset['Country'] == country]
        y_test = data['Temp']
        X_test = data['DayOfYear']
        loss = poly.loss(np.asarray(X_test), np.asarray(y_test))
        losses.append(loss)
    fig = px.bar(losses, title="Model's Error Over Each Country", x=countries, y=losses,
                 labels={'x': 'country', 'y': "Model's Error"})
    fig.show()

