from itertools import product

from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from load_data import load_blog_catalog
from transformers import transformers
from classifiers import MultilabelOVRClassifier

from settings import PATH_TO_DUMPS

qual_measures = {
    'f1-macro': lambda y_true, y_pred: metrics.f1_score(y_true, y_pred, average='macro'),
    'f1-micro': lambda y_true, y_pred: metrics.f1_score(y_true, y_pred, average='micro'),
    'accuracy': lambda y_true, y_pred: metrics.accuracy_score(y_true, y_pred),
}


def scorer(estimator, X, y):
    return qual_measures['f1-micro'](y, estimator.predict(X))

n_jobs = -1

# available methods:
# 'stoc_hist', 'node2vec', 'stoc_opt', 'stocsk_opt',
# 'bigclam', 'gamma', 'deepwalk', 'svd', 'nmf'

methods = ['deepwalk', 'node2vec']
dimensions = [32, 64, 128]

graph = load_blog_catalog()

for method, dimension in product(methods, dimensions):
    print("Starting experiment with model={} dim={}".format(method, str(dimension)))
    embedding_transformer = transformers[method](
        graph, 'BlogCatalog', dimension, seed=43, path_to_dumps=PATH_TO_DUMPS
    )

    embedding_transformer.fit()

    # params_grid = {
    # }
    # params_grid.update(embedding_transformer.cvparams)

    clf = MultilabelOVRClassifier(LogisticRegression(n_jobs=n_jobs))

    # ppl = Pipeline([('embedding', embedding_transformer), ('classifier', clf)])

    ss = ShuffleSplit(n_splits=4, random_state=43, train_size=0.4)
    scores = cross_val_score(clf, graph, y_multilabel, cv=ss.split(X), scoring='f1_micro', verbose=1)
    print(scores)
    print(scores.mean())
