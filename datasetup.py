
import pandas as pd
import numpy as np
import sklearn.cross_validation
import sklearn.preprocessing

def load_data(filename):
    """ Loads the data from the given filename in CSV format

        Things to consider here:
        -drop unnecessary columns (team names, etc)
        -feature scaling?

    :param filename: of the CSV file

    :return: y, X_scaled
    """

    df = pd.read_csv(filename)

    return df


def split_df_labels_features(df):
    """ splits a dataframe into y and X provided it conforms to the
        required input colums of id, labels, features...
    """

    #TODO: also split and return id/index, to later do odds comaprison
    y = df['FTR']
    X = df.iloc[:, 5:]  # skip team names and goals
    #print X.head()
    return X.values, y.values


def get_stratified_data(X, y, test_size):
    """ Splits the data into a training set and test set as defined in config
    :param y:
    :param X:
    :return:
    """
    sss = sklearn.cross_validation.StratifiedShuffleSplit(y, n_iter=1, test_size=test_size)

    train_index, test_index = next(iter(sss))

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    return X_train, y_train, X_test, y_test
