import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit, cross_val_score

from io_utils.graph import load_graph
from settings import PATH_TO_BLOG_CATALOG
from src.classifiers import MultilabelOVRClassifier
from transformation.RunConfiguration import RunConfiguration
from transformers.Adapter import calc_embedding


def run_blog_catalog(run_configuration: RunConfiguration, path_to_dumps, seed=43):

    assert run_configuration.graph_name == 'blog_catalog'

    graph = load_graph(
        run_configuration.graph_name,
        weighted=True if run_configuration.method == 'node2vec' else False
    )

    E = calc_embedding(
        run_configuration.method,
        graph,
        run_configuration.graph_name,
        run_configuration.dimension,
        path_to_dumps,
        seed=seed,
        use_cached=True
    )

    X = pd.DataFrame(E)

    y = np.zeros((10312, 39), dtype=np.int32)
    for x in pd.read_csv(
            PATH_TO_BLOG_CATALOG + '/data/group-edges.csv',
            delimiter=' ',
            header=None
    ).values:
        y[x[0] - 1, x[1] - 1] = 1

    clf = MultilabelOVRClassifier(LogisticRegression(), n_jobs=-1)
    MultilabelOVRClassifier.set_labels(X.index, y)
    ss = ShuffleSplit(n_splits=4, random_state=43, train_size=0.5, test_size=0.5)
    scores = cross_val_score(clf, X, y, cv=ss.split(X), scoring='f1_micro', verbose=0)
    return scores
