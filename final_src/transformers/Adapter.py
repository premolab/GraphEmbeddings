from io_utils.embedding import read_embedding, path_to_embedding
from transformation.HistLossConfiguration import HistLossConfiguration
from transformers.DeepWalkTransformer import DeepWalkTransformer
from transformers.HistLossTransformer import HistLossTransformer
from transformers.Node2VecTransformer import Node2VecTransformer
from transformers.HopeTransformer import HopeTransformer
from transformers.SdneTransformer import SdneTransformer


def calc_embedding(method,
                   graph,
                   graph_name,
                   dimension,
                   path_to_dumps,
                   seed=43,
                   use_cached=True,
                   should_stop=None):
    if method == 'deepwalk':
        DeepWalkTransformer(
            graph,
            graph_name,
            dimension,
            seed=seed,
            path_to_dumps=path_to_dumps,
            dump_model=True,
            load_dumped_model=use_cached
        ).fit()
        E = read_embedding(path_to_embedding(
            root=path_to_dumps,
            method=method,
            name=graph_name,
            dim=dimension
        ))

    elif method == 'node2vec':
        Node2VecTransformer(
            graph,
            graph_name,
            dimension,
            seed=seed,
            path_to_dumps=path_to_dumps,
            dump_model=True
        ).fit()
        E = read_embedding(path_to_embedding(
            root=path_to_dumps,
            method=method,
            name=graph_name,
            dim=dimension
        ))

    elif method == 'hope':
        HopeTransformer(
            graph,
            graph_name,
            dimension,
            path_to_dumps=path_to_dumps
        ).fit()
        E = read_embedding(path_to_embedding(
            root=path_to_dumps,
            method=method,
            name=graph_name,
            dim=dimension
        ))

    elif method == 'sdne':
        SdneTransformer(
            graph,
            graph_name,
            dimension,
            path_to_dumps=path_to_dumps,
            use_cached=True,
            d=dimension, beta=5, alpha=1e-5, nu1=1e-6, nu2=1e-6, K=3,
            n_units=[500, 300], rho=0.3, n_iter=30, xeta=0.001,
            n_batch=500
        ).fit()
        E = read_embedding(path_to_embedding(
            root=path_to_dumps,
            method=method,
            name=graph_name,
            dim=dimension
        ))

    elif method.startswith('hist_loss_'):
        HistLossTransformer(
            graph,
            graph_name,
            dimension,
            seed,
            HistLossConfiguration.from_string(method[10:]),
            path_to_dumps=path_to_dumps,
            use_cached=use_cached,
            should_stop=should_stop
        ).fit()
        E = read_embedding(path_to_embedding(
            root=path_to_dumps,
            method=method,
            name=graph_name,
            dim=dimension
        ))
    else:
        raise Exception("Unknown method: " + method)

    return E