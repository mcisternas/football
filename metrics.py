import pandas
import matplotlib.pyplot as plt
import sklearn.calibration
import sklearn.preprocessing
import sklearn.metrics
import sklearn.decomposition
from sklearn.grid_search import GridSearchCV
import numpy as np
import seaborn
import os.path

out_path = 'output'

def confusion_matrix(clf_results, threshold=0.5):

    y_test = clf_results["y_test"]
    clf_name = clf_results["clf_name"]
    y_pred_label = clf_results["y_pred_label"]

    cm = sklearn.metrics.confusion_matrix(y_pred_label, y_test)

    seaborn.set_style("white")
    plt.figure(figsize=(10, 10))
    ax1 = plt.subplot2grid((2, 2), (0, 0))
    ax2 = plt.subplot2grid((2, 2), (0, 1))
    cmap = plt.cm.Blues

    ax1.set_title("{} - Confusion Matrix".format(clf_name))
    ax1.set_ylabel('True label')
    ax1.set_xlabel('Predicted label')
    ax1.locator_params(nbins=4)
    ax1.set_xticklabels(['', 'A', 'D', 'H'])
    ax1.set_yticklabels(['', 'A', 'D', 'H'])

    confMatrix1 = ax1.imshow(cm, interpolation='nearest', cmap=cmap)

    # Display the values of the conf matrix on the plot
    cm_bbox = {'facecolor': 'white', 'alpha': 0.5, 'pad': 10}
    for i in range(len(set(y_test))):
        for j in range(len(set(y_test))):
            ax1.text(i, j, "%d" % cm[i, j], size=14, ha='center', bbox=cm_bbox)

    plt.colorbar(confMatrix1, ax=ax1)

    ax2.set_title("Normalised Confusion Matrix")
    ax2.set_ylabel('True label')
    ax2.set_xlabel('Predicted label')
    ax1.locator_params(nbins=4)
    ax1.set_xticklabels(['', 'A', 'D', 'H'])
    ax1.set_yticklabels(['', 'A', 'D', 'H'])

    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    confMatrix2 = ax2.imshow(cm_normalized, interpolation='nearest', cmap=cmap)

    for i in range(len(set(y_test))):
        for j in range(len(set(y_test))):
            ax2.text(i, j, "%4.2f" % cm_normalized[i, j], ha='center', size=14, bbox=cm_bbox)

    plt.colorbar(confMatrix2, ax=ax2)
    plt.savefig(os.path.join(out_path, 'conf_matrix__{}'.format(clf_name.replace(' ', ''))), bbox_inches='tight')
