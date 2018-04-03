from sklearn.multiclass import OneVsRestClassifier
import numpy as np


class MultilabelOVRClassifier(OneVsRestClassifier):
    def __init__(self, estimator, n_jobs):
        super(MultilabelOVRClassifier, self).__init__(estimator, n_jobs)
        self.label_count = None
        self.y_label_count = None
        self.X_index_dict = None

    def set_labels(self, X_indexes, y):
        self.label_count = y.shape[0]
        self.y_label_count = y.sum(axis=1).astype(np.int)
        self.X_index_dict = dict(zip(X_indexes, range(len(y))))

    def predict(self, X):
        pred_probas = self.predict_proba(X)
        res = np.zeros_like(pred_probas)
        for i, index in enumerate(X.index.values):
            y_index = self.X_index_dict[index]
            label_count = self.y_label_count[y_index]
            best_indexes = np.argsort(pred_probas[i])[-label_count:]
            res[i, best_indexes] = 1
        return res
