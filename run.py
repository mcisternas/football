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
import sklearn.naive_bayes

import docopt
import matplotlib.pyplot as plt

import datasetup
import metrics

#Config params- consider moving it to a config file
datafile = "data/England/E0_0815_feats.csv"
test_size = 0.3

classifiers = {
    'Logistic Regression': sklearn.linear_model.LogisticRegression(C=1),#, class_weight='auto'),
    #'Random Forests': sklearn.ensemble.RandomForestClassifier(n_estimators=100, n_jobs=1),
    #'SVC': sklearn.svm.SVC(C=1.0, probability=True),
    'Gradient Boosting': sklearn.ensemble.GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=2),
    'Naive Bayes - Gaussian': sklearn.naive_bayes.GaussianNB(),
    #'Naive Bayes - Multinomial': sklearn.naive_bayes.MultinomialNB(),
}


def run_benchmark(classifiers): #, classifiers_gridparameters):

    #load data
    df = datasetup.load_data(datafile)

    #split data X/y, scale, split train/test
    X, y = datasetup.split_df_labels_features(df)
    X = datasetup.scale_features(X)
    X_train, y_train, X_test, y_test = datasetup.get_stratified_data(X, y, test_size)

    #for loop, iterate over classifiers
    for clf_name, clf_notoptimized in classifiers.iteritems():
        print "****************************************"
        print "Running: %s" %clf_name
        clf_results = {'clf_name': clf_name}

        #clf_notoptimized = sklearn.multiclass.OneVsRestClassifier(clf_notoptimized)
        clf_fitted = clf_notoptimized.fit(X_train, y_train)
        y_pred = clf_fitted.predict_proba(X_test)#[:, 1]
        y_pred_label = clf_fitted.predict(X_test)


        clf_results.update({
            'y_pred': y_pred,
            'y_pred_label': y_pred_label,
            'clf': clf_fitted,
            'clf_notoptimized': clf_notoptimized,
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test,
            #'param_grid': param_grid,
        })

        # to test the home-wins benchmark
        #y_pred_label = np.repeat(['H'],len(y_pred_label))
        
        print "Accuracy: %5.2f" %sklearn.metrics.accuracy_score(y_test, y_pred_label)
        #print "Jaccard: %5.2f " %sklearn.metrics.jaccard_similarity_score(y_test, y_pred_label)
        print(sklearn.metrics.classification_report(y_test, y_pred_label))#, target_names=target_names))

        metrics.confusion_matrix(clf_results)

if __name__ == '__main__':
    run_benchmark(classifiers)
