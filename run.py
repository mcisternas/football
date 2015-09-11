import os
import importlib
import logging
import csv

from sklearn.grid_search import GridSearchCV
import sklearn.metrics
import sklearn.cross_validation
import sklearn.ensemble
import docopt
import matplotlib.pyplot as plt

import datasetup

datafile = "data/England/E0_1215_feats.csv"

classifiers = {
    'Logistic Regression': sklearn.linear_model.LogisticRegression(),
    'Random Forests': sklearn.ensemble.RandomForestClassifier(n_estimators=100, n_jobs=config["num_cores"]),
    # 'SVC': sklearn.svm.SVC(C=1.0, probability=True),
    # 'Gradient Boosting': sklearn.ensemble.GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=2),
}


def run_benchmark(classifiers, classifiers_gridparameters):

    #load data
    df = datasetup.load_data(datafile)
    #split data, X/y train/test
    
    #for loop, iterate over classifiers
