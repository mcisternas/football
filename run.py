import os
import importlib
import logging
import csv
import numpy as np
from sklearn.grid_search import GridSearchCV
import sklearn.metrics
import sklearn.cross_validation
import sklearn.ensemble
import sklearn.multiclass

import sklearn.linear_model
import sklearn.neural_network
import sklearn.pipeline
import sklearn.ensemble

import docopt
import matplotlib.pyplot as plt

import datasetup

#Config params- consider moving it to a config file
datafile = "data/England/E0_1115_feats.csv"
test_size = 0.3

classifiers = {
    'Logistic Regression': sklearn.linear_model.LogisticRegression(),
    #'Random Forests': sklearn.ensemble.RandomForestClassifier(n_estimators=100, n_jobs=1),
    #'SVC': sklearn.svm.SVC(C=1.0, probability=True),
    'Gradient Boosting': sklearn.ensemble.GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3),
}


def run_benchmark(classifiers): #, classifiers_gridparameters):

    #load data
    df = datasetup.load_data(datafile)

    #split data, X/y train/test
    X, y = datasetup.split_df_labels_features(df)
    X_train, y_train, X_test, y_test = datasetup.get_stratified_data(X, y, 0.25)

    #for loop, iterate over classifiers
    for clf_name, clf_notoptimized in classifiers.iteritems():
        print "****************************************"
        print "Running: %s" %clf_name

        #clf_notoptimized = sklearn.multiclass.OneVsRestClassifier(clf_notoptimized)
        clf_fitted = clf_notoptimized.fit(X_train, y_train)
        y_pred = clf_fitted.predict_proba(X_test)#[:, 1]
        y_pred_label = clf_fitted.predict(X_test)

        #for i in range(len(y_pred_label)):
        #    h_prob = y_pred[i,2]
        #    d_prob = y_pred[i,1]
        #    a_prob = y_pred[i,0]
        #    print "%5.3f %5.3f %5.3f %s %s" %(h_prob,
        #            d_prob, a_prob, y_pred_label[i], y_test[i])

        # to test the home-wins benchmark
        #y_pred_label = np.repeat(['H'],len(y_pred_label))
        print "Accuracy: %5.2f" %sklearn.metrics.accuracy_score(y_test, y_pred_label)
        #print "Jaccard: %5.2f " %sklearn.metrics.jaccard_similarity_score(y_test, y_pred_label)
        print(sklearn.metrics.classification_report(y_test, y_pred_label))#, target_names=target_names))
        #print zip(y_test, y_pred_label)

if __name__ == '__main__':
    run_benchmark(classifiers)
