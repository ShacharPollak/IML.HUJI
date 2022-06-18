import math
from math import nan

import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import *
import sklearn.feature_selection
from sklearn.model_selection import train_test_split, GridSearchCV


def dummy_df(df, todummy_list):
    for x in todummy_list:
        dummies = pd.get_dummies(df[x], prefix=x, dummy_na=False)
        df = df.drop(x, 1)
        df = pd.concat([df, dummies], axis=1)
    return df


# preparing X y #
def prep_X_y(df):
    X = df.drop("cancellation_datetime", 1)
    y = [1 if x is not np.nan else 0 for x in df['cancellation_datetime']]
    z = df['cancellation_datetime']
    return X, y, z


# adding columns #
def add_columns(X):
    func = lambda t: (int(t[:4]) - 2017) * 365 + int(t[5:7]) * 31 + int(
        t[8:10])
    func_month = lambda y: int(y[5:7])
    func_day = lambda y: int(y[5:7]) * 31 + int(y[8:10])
    vec = np.vectorize(func)
    vec_month = np.vectorize(func_month)
    vec_day = np.vectorize(func_day)
    arr = []
    for start, end in zip(X['checkin_date'], X['checkout_date']):
        arr.append(np.busday_count(start[:10], end[:10]))

    X['start_day'] = vec_day(X['checkin_date'])
    X['end_day'] = vec_day(X['checkout_date'])
    X['month_of_stay'] = vec_month(X['checkin_date'])
    X['time_to_stay'] = vec(X['checkin_date']) - vec(X['booking_datetime'])
    X['duration'] = vec(X['checkout_date']) - vec(X['checkin_date'])
    X['same_country_invite'] = np.where(
        X["origin_country_code"] == X["hotel_country_code"], 1, 0)
    week_end_days = []
    for weekdays, total_days in zip(arr, X['duration']):
        week_end_days.append(total_days - weekdays)
    X['week_end_days'] = week_end_days
    return X


# staring droping from X #
def drop_columns(X):
    return X.drop(
        ["h_booking_id", "hotel_id", "hotel_live_date", "h_customer_id",
         "hotel_area_code", "hotel_brand_code", "hotel_chain_code",
         "customer_nationality", "origin_country_code", "hotel_city_code",
         'checkin_date', 'booking_datetime', 'checkout_date'], 1)


# changing FALSE to 0 TRUE to 1 #
def bool_changer(X):
    X['is_user_logged_in'] = [0 if x is False else 1 for x in
                              X['is_user_logged_in']]
    X['is_first_booking'] = [0 if x is False else 1 for x in
                             X['is_first_booking']]
    return X


# filling zero's where needed #
def fill_zeros(X):
    X['request_nonesmoke'] = X['request_nonesmoke'].fillna(0)
    X['request_latecheckin'] = X['request_latecheckin'].fillna(0)
    X['request_highfloor'] = X['request_highfloor'].fillna(0)
    X['request_largebed'] = X['request_largebed'].fillna(0)
    X['request_twinbeds'] = X['request_twinbeds'].fillna(0)
    X['request_airport'] = X['request_airport'].fillna(0)
    X['request_earlycheckin'] = X['request_earlycheckin'].fillna(0)
    return X


# making dummies #
def make_dummies(X):
    todummy_list = ['hotel_country_code', 'accommadation_type_name',
                    'charge_option', 'guest_nationality_country_name',
                    'language',
                    'original_payment_method', 'original_payment_type',
                    'original_payment_currency', 'cancellation_policy_code']

    hotel_country_cod_set = {'JP', 'TH', 'MY', 'TW', 'ID', 'KR', 'PH', 'VN',
                             'US',
                             'HK', 'CN', 'SG', 'AU'}
    accommadation_type_set = {'Hotel', 'Resort',
                              'Guest House / Bed & Breakfast',
                              'Hostel'}

    country_set = {'South Korea', 'Malaysia', 'Taiwan', 'Thailand', 'China',
                   'Japan', 'Hong Kong', 'Indonesia',
                   'United States of America'}
    language_set = {'English', 'Korean', 'T. Chinese / Taiwan',
                    'S.Chinese / Mainland',
                    'Japanese', 'Thai'}

    original_payment_method_set = {'Visa', 'MasterCard', 'UNKNOWN',
                                   'American Express'}
    original_payment_currency_set = {'MYR', 'KRW', 'TWD', 'USD', 'THB', 'CNY',
                                     'JPY', 'HKD'}
    cancellation_policy_code_set = {'365D100P_100P', '1D1N_1N', '3D1N_1N',
                                    '1D100P', '3D1N_100P', '1D100P_100P',
                                    '7D100P_100P',
                                    '3D100P_100P', '2D100P', '7D1N_100P',
                                    '3D100P',
                                    '2D1N_1N', '0D0N', '14D100P_100P'}

    X['hotel_country_code'] = [x if x in hotel_country_cod_set else 'Other' for
                               x
                               in X['hotel_country_code']]
    X['accommadation_type_name'] = [
        x if x in accommadation_type_set else 'Other'
        for x in X['accommadation_type_name']]
    X['guest_nationality_country_name'] = [x if x in country_set else 'Other'
                                           for x
                                           in
                                           X['guest_nationality_country_name']]
    X['language'] = [x if x in language_set else 'Other' for x in
                     X['language']]
    X['original_payment_method'] = [
        x if x in original_payment_method_set else 'Other' for x in
        X['original_payment_method']]
    X['original_payment_currency'] = [
        x if x in original_payment_currency_set else 'Other' for x in
        X['original_payment_currency']]

    X['cancellation_policy_code'] = [
        x if x in cancellation_policy_code_set else 'Other' for x in
        X['cancellation_policy_code']]

    X = dummy_df(X, todummy_list)
    # X_test does not have 'charge_option_Pay at Check-in'
    if 'charge_option_Pay at Check-in' in X:
        X = X.drop(['charge_option_Pay at Check-in'], 1)
    return X


def generate_prediction(X_train, y_train, X_test):
    """
    helper function to test if we want select, so far we don't
    @param X_train:
    @param y_train:
    @param X_test:
    """
    select = sklearn.feature_selection.SelectKBest(k=10)
    selected_features = select.fit(X_train, y_train)
    indices_selected = selected_features.get_support(indices=True)
    colnames_selected = [X_train.columns[i] for i in indices_selected]
    X_train_selected = X_train[colnames_selected]
    X_test_selected = X_test[colnames_selected]
    predict(X_train_selected, y_train, X_test_selected,
                            "318196839_318670379_208781005.csv")


def process_data(X):
    X = add_columns(X)
    X = drop_columns(X)
    X = bool_changer(X)
    X = fill_zeros(X)
    X = make_dummies(X)
    return X


def test_model(X, y, y_days):
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        train_size=0.70,
                                                        random_state=1)

    y_days_train, y_days_test = train_test_split(y_days, train_size=0.70,
                                                        random_state=1)

    parameters = {
        "n_estimators": [100],
        "learning_rate": [1.],
    }

    model_adaboost = AdaBoostClassifier(
        random_state=27,
    )

    model_adaboost = GridSearchCV(
        model_adaboost,
        parameters,
        cv=5,
        scoring='accuracy',
    )
    # model_adaboost = LinearRegression()
    model_adaboost.fit(X_train, y_train)
    y_hat = model_adaboost.predict(X_test)
    count = 0

    model_days = LinearRegression()
    X_days = X_train
    X_days['cancel_day'] = y_days_train
    X_days = X_days.loc[X_days['cancel_day'] > 0]
    y_days_train = X_days['cancel_day']
    X_days = X_days.drop(['cancel_day'], 1)
    model_days.fit(X_days, y_days_train)
    X_days_test = X_test
    X_days_test['cancel_day'] = y_days_test
    X_days_test = X_days_test.loc[X_days_test['cancel_day'] > 0]
    y_days_test = X_days_test['cancel_day']
    X_days_test = X_days_test.drop(['cancel_day'], 1)

    y_days_predict = model_days.predict(X_days_test)

    count_yes_can = 0

    y_res = []
    for item in y_hat:
        if not item:
            y_res.append(0)
        else:
            y_res.append(y_days_predict[count_yes_can])
            count_yes_can += 1

    for y_ex, y_real in zip(y_res, y_test):
        if y_ex == y_real == 0:
            count += 1

        elif abs(y_ex - y_real) < 7:  # i give a 7 day diff as good
            count += 1

    print("the real acc we got is: ", count / len(y_test))


def predict(X_train, y_train, X_test, y_days, filename):
    parameters = {
        "n_estimators": [100],
        "learning_rate": [1.],
    }

    model_adaboost = AdaBoostClassifier(
        random_state=27,
    )

    model_adaboost = GridSearchCV(
        model_adaboost,
        parameters,
        cv=5,
        scoring='accuracy',
    )
    model_adaboost.fit(X_train, y_train)  # model that says if cancel or not
    y_hat = model_adaboost.predict(X_test)  # y_hat == canceled or not

    model_days = LinearRegression()
    X_days = X_train
    X_days['cancel_day'] = y_days
    X_days = X_days.loc[X_days['cancel_day'] > 0]
    y_days_train = X_days['cancel_day']
    X_days = X_days.drop(['cancel_day'], 1)
    model_days.fit(X_days, y_days_train)
    # this model tells when cancel will happen
    y_days_predict = model_days.predict(X_test)

    count_yes_can = 0

    y_res = []
    for item in y_hat:
        if not item:
            y_res.append(0)

        elif 742 <= y_days_predict[count_yes_can] <= 752:
            y_res.append(1)
            count_yes_can += 1
        else:
            y_res.append(0)
            count_yes_can += 1
    print(count_yes_can, len(y_days_predict))
    print("exporting y_res")
    pd.DataFrame(y_res, columns=["predicted_values"]).to_csv(
        filename, index=False)


# helper function to check how good is our model prediction
def check_percentage(X_train, y_train, X_test, y_test, a):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_hat = [x[1] for x in model.predict_proba(X_test)]
    y_res = [0 if x < a else 1 for x in y_hat]
    count = 0
    y_predict = model.predict(X_test)
    for y_expected, y_real in zip(y_res, y_test):
        if y_expected == y_real:
            count += 1
    print("our way", count / len(y_test))
    count = 0

    for y_expected, y_real in zip(y_predict, y_test):
        if y_expected == y_real:
            count += 1
    print("with function", count / len(y_test))
    # auc = roc_auc_score(y_test, y_hat)
    # print("acc", auc)
    return count / len(y_test)

def create_day_model(X):
    func = lambda t: (int(t[:4]) - 2017) * 365 + int(t[5:7]) * 31 + int(
        t[8:10])

    y = []
    for day in X['cancellation_datetime']:
        if isinstance(day, str):
            y.append(func(day))
        else:
            y.append(0)
    return y


if __name__ == '__main__':
    df_train = pd.read_csv('../datasets/agoda_cancellation_train.csv')  # 57000
    df_test = pd.read_csv('week_9_test_data.csv') # 800
    X_train, y_train, z = prep_X_y(df_train)
    X_train = process_data(X_train)
    X_test = process_data(df_test)
    y_days = create_day_model(df_train)
    # test_model(X_train, y_train, y_days)
    predict(X_train, y_train, X_test, y_days, "318196839_318670379_208781005.csv")