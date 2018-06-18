import pandas as pd
from sklearn.cluster import AgglomerativeClustering

from sklearn.metrics import normalized_mutual_info_score

from io_utils.graph import load_graph
from settings import PATH_TO_FOOTBALL
from transformation.RunConfiguration import RunConfiguration
from transformers.Adapter import calc_embedding


def run_football(run_configuration: RunConfiguration, path_to_dumps, seed=43):

    assert run_configuration.graph_name == 'football'

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

    y = pd.read_csv(
        PATH_TO_FOOTBALL + '/football_labels.txt',
        header=None,
        squeeze=True
    )

    y_pred = AgglomerativeClustering(n_clusters=12).fit_predict(X)

    return normalized_mutual_info_score(y, y_pred)


def run_sbm(run_configuration: RunConfiguration, path_to_dumps, seed=43):
    assert run_configuration.graph_name.startswith('sbm')

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

    y = [1] * 300 + [2] * 300 + [3] * 300

    y_pred = AgglomerativeClustering(n_clusters=3).fit_predict(X)

    return normalized_mutual_info_score(y, y_pred)
