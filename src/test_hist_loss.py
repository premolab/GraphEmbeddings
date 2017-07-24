import theano
from theano import tensor as T
import networkx as nx
import pandas as pd

from lib.hist_loss.HistLoss import HistLoss
from load_data import load_blog_catalog, load_karate
from settings import PATH_TO_DUMPS


def test_calc_hist():
    # input for loss
    A = T.dmatrix('A')  # shape: (n, n)
    E = T.dmatrix('E')  # shape: (n, dim)
    E_norm = E/E.norm(2, axis=1).reshape((E.shape[0], 1))  # shape: (n, n)
    E_corr = T.dot(E_norm, E_norm.T)  # shape: (n, n)

    pos_mask = A  # shape: (n, n)
    neg_mask = 1 - pos_mask - T.eye(pos_mask.shape[0])  # shape: (n, n)

    pos_samples = E_corr[pos_mask.nonzero()]
    neg_samples = E_corr[neg_mask.nonzero()]

    # if neg_sampling:
    #     neg_samples = neg_samples[srng.choice(size=(2*pos_samples.shape[0], ), a=neg_samples.shape[0])]

    pos_hist = HistLoss.calc_hist(pos_samples, bin_num=64)
    neg_hist = HistLoss.calc_hist(neg_samples, bin_num=64)

    agg_pos = T.extra_ops.cumsum(pos_hist)
    loss = T.sum(T.dot(agg_pos, neg_hist))

    import time

    print('Compiling')
    t = time.time()
    f = theano.function(
        [A, E], [E_norm, E_corr, pos_mask, neg_mask, pos_samples, neg_samples, pos_hist, neg_hist, loss]
    )
    print(time.time() - t)

    print('Reading embedding')
    t = time.time()
    X = pd.read_csv(
        '{}/models/deepwalk_BlogCatalog_d32.csv'.format(PATH_TO_DUMPS),
        delim_whitespace=True, header=None,
        skiprows=1,
        index_col=0
    ).sort_index()
    E_input = X.values
    print(time.time() - t)

    print('Reading graph')
    t = time.time()
    graph = load_blog_catalog()
    nodes = graph.nodes()
    adjacency_matrix = nx.adjacency_matrix(graph, nodes).todense().astype("float32")
    print(time.time() - t)
    # adjacency_matrix = [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
    # E_input = [[1, 2, 3], [1, 3, 4], [-1, -2, -4]]
    print('Solving')
    t = time.time()
    res = f(adjacency_matrix, E_input)
    print(time.time() - t)

    print(*res, sep='\n')

if __name__ == '__main__':
    test_calc_hist()
