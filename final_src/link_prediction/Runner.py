from pathlib import Path

import numpy as np

from io_utils.graph import load_graph, save_graph, read_graph2
from link_prediction.GraphSampler import GraphSampler
from link_prediction.Metric import calc_link_prediction_roc_auc
from transformation.RunConfiguration import RunConfiguration

from settings import PATH_TO_LINK_PREDICTION_DATASETS
from transformers.Adapter import calc_embedding


def should_stop():
    with open('early_stopping.txt', 'r') as file:
        res = file.read() == '1'
    if res:
        with open('early_stopping.txt', 'w') as file:
            file.write('0')
    return res


def run(run_configuration: RunConfiguration, path_to_dumps, ratio=0.5, seed=43):
    graph = load_graph(run_configuration.graph_name)
    edges = graph.edges()

    path = Path(PATH_TO_LINK_PREDICTION_DATASETS) / run_configuration.graph_name / "train.edgelist"

    if not path.exists():
        train_graph = GraphSampler(graph, ratio).fit_transform()
        if not (Path(PATH_TO_LINK_PREDICTION_DATASETS) / run_configuration.graph_name).exists():
            (Path(PATH_TO_LINK_PREDICTION_DATASETS) / run_configuration.graph_name).mkdir()
        save_graph(
            train_graph,
            str(path)
        )
    else:
        train_graph = read_graph2(str(path))

    E = calc_embedding(
        run_configuration.method,
        graph,
        run_configuration.graph_name + '-lp-{}-{}'.format(ratio, seed),
        run_configuration.dimension,
        path_to_dumps,
        should_stop=should_stop
    )

    train_edges = train_graph.edges()

    edges_set = set(edges)
    train_edges_set = set(train_edges)

    test_edges_set = edges_set - train_edges_set

    np.random.seed(seed)
    test_neg_edges_set = set()
    while len(test_neg_edges_set) < len(test_edges_set):
        edge = (np.random.randint(0, len(graph.nodes())), np.random.randint(0, len(graph.nodes())))
        if edge not in edges_set:
            test_neg_edges_set.add(edge)
    assert len(test_neg_edges_set & edges_set) == 0

    train_neg_edges_set = set()
    while len(train_neg_edges_set) < len(train_edges_set):
        edge = (np.random.randint(0, len(graph.nodes())), np.random.randint(0, len(graph.nodes())))
        if edge not in train_edges_set:
            train_neg_edges_set.add(edge)
    assert len(train_neg_edges_set & train_edges_set) == 0

    return calc_link_prediction_roc_auc(E,
                                        train_edges,
                                        list(train_neg_edges_set),
                                        list(test_edges_set),
                                        list(test_neg_edges_set),
                                        indexing_offset=1)
