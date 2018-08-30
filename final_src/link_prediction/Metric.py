import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score


def calc_link_prediction_score(E,
                                 train_edges,
                                 train_neg_edges,
                                 test_edges,
                                 test_neg_edges,
                                 score='roc-auc'):
    nodes_set = set(e[0] for e in train_edges).union(set(e[1] for e in train_edges))
    assert len(nodes_set) == E.shape[0], str(len(nodes_set)) + ' != ' + str(E.shape[0])

    sorted_nodes = sorted(nodes_set, key=int)
    nodes_dict = {sorted_nodes[i]: i for i in range(len(sorted_nodes))}

    clf_X = np.zeros((len(train_edges) + len(train_neg_edges), E.shape[1]))
    clf_y = np.zeros((len(train_edges) + len(train_neg_edges), ))
    i = 0
    for edge in train_edges:
        clf_X[i] = E[nodes_dict[edge[0]]] * E[nodes_dict[edge[1]]]
        clf_y[i] = 1
        i += 1
    for edge in train_neg_edges:
        clf_X[i] = E[nodes_dict[edge[0]]] * E[nodes_dict[edge[1]]]
        clf_y[i] = 0
        i += 1
    test_X = np.zeros((len(test_edges) + len(test_edges), E.shape[1]))
    test_y = np.zeros((len(test_edges) + len(test_neg_edges), ))
    i = 0
    for edge in test_edges:
        test_X[i] = E[nodes_dict[edge[0]]] * E[nodes_dict[edge[1]]]
        test_y[i] = 1
        i += 1
    for edge in test_neg_edges:
        test_X[i] = E[nodes_dict[edge[0]]] * E[nodes_dict[edge[1]]]
        test_y[i] = 0
        i += 1
    clf = LogisticRegression()
    clf.fit(clf_X, clf_y)
    pred_y = clf.predict_proba(test_X)[:, 1]
    if score == 'roc-auc':
        return roc_auc_score(y_true=test_y, y_score=pred_y)
    elif score == 'f1':
        return f1_score(y_true=test_y, y_pred=pred_y)
