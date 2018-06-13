from sklearn.multiclass import OneVsRestClassifier
import numpy as np


class MultiLabelOVRClassifier(OneVsRestClassifier):
    label_count = None
    y_label_count = None
    X_index_dict = None

    def __init__(self, estimator, n_jobs):
        super(MultiLabelOVRClassifier, self).__init__(estimator, n_jobs)

    @classmethod
    def set_labels(cls, X_indexes, y):
        cls.label_count = y.shape[0]
        cls.y_label_count = y.sum(axis=1).astype(np.int)
        cls.X_index_dict = dict(zip(X_indexes, range(len(y))))

    def predict(self, X):
        pred_probas = self.predict_proba(X)
        res = np.zeros_like(pred_probas)
        for i, index in enumerate(X.index.values):
            y_index = self.X_index_dict[index]
            label_count = self.y_label_count[y_index]
            best_indexes = np.argsort(pred_probas[i])[-label_count:]
            res[i, best_indexes] = 1
        return res
