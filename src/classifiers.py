from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
import numpy as np


class MultilabelOVRClassifier(OneVsRestClassifier):
    @classmethod
    def set_labels(cls, X_indexes, y):
        cls.label_count = len(y[0])
        cls.y_label_count = np.array([sum(raw) for raw in y])
        cls.X_index_dict = dict(zip(X_indexes, range(len(y))))

    # def __init__(self, estimator, X_indexes, y, n_jobs=1):
    #     super().__init__(estimator, n_jobs)

    # def fit(self, X, y):
    #     return super().fit(X, y)

    def predict(self, X):
        pred_probas = self.predict_proba(X)
        res = np.zeros((len(pred_probas), self.label_count))
        for i, index in enumerate(X.index.values):
            y_index = self.X_index_dict[index]
            label_count = self.y_label_count[y_index] + 1
            i_pred_probas = pred_probas[i]
            best_indexes = np.argsort(i_pred_probas)[-label_count:]
            for j in best_indexes:
                res[i][j] = 1
        return res
