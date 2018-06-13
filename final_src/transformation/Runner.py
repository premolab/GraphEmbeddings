from io_utils.graph import load_graph
from transformation.RunConfiguration import RunConfiguration
from transformers.Adapter import calc_embedding


def run(run_configuration: RunConfiguration, path_to_dumps, seed=43):

    graph = load_graph(
        run_configuration.graph_name,
        weighted=True if run_configuration.method == 'node2vec' else False
    )

    calc_embedding(
        run_configuration.method,
        graph,
        run_configuration.graph_name,
        run_configuration.dimension,
        path_to_dumps,
        seed=seed,
        use_cached=True
    )
