import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def calc_link_prediction_roc_auc(E,
                                 train_edges,
                                 train_neg_edges,
                                 test_edges,
                                 test_neg_edges,
                                 indexing_offset=0):
    clf_X = np.zeros((len(train_edges) + len(train_neg_edges), E.shape[1]))
    clf_y = np.zeros((len(train_edges) + len(train_neg_edges), ))
    i = 0
    for edge in train_edges:
        clf_X[i] = E[int(edge[0]) - indexing_offset] * E[int(edge[1]) - indexing_offset]
        clf_y[i] = 1
        i += 1
    for edge in train_neg_edges:
        clf_X[i] = E[edge[0] - indexing_offset] * E[edge[1] - indexing_offset]
        clf_y[i] = 0
        i += 1
    test_X = np.zeros((len(test_edges) + len(test_edges), E.shape[1]))
    test_y = np.zeros((len(test_edges) + len(test_neg_edges), ))
    i = 0
    for edge in test_edges:
        test_X[i] = E[edge[0] - indexing_offset] * E[edge[1] - indexing_offset]
        test_y[i] = 1
        i += 1
    for edge in test_neg_edges:
        test_X[i] = E[edge[0] - indexing_offset] * E[edge[1] - indexing_offset]
        test_y[i] = 0
        i += 1
    clf = LogisticRegression()
    clf.fit(clf_X, clf_y)
    pred_y = clf.predict_proba(test_X)[:, 1]
    return roc_auc_score(y_true=test_y, y_score=pred_y)
