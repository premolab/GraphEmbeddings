import downhill
import networkx as nx
import pickle

from itertools import product

from lib.hist_loss.HistLoss import HistLoss
from lib.hist_loss.HistLossWeighted import HistLossWeighted
from load_data import load_blog_catalog, load_karate, load_football, load_stars, load_polbooks, load_protein, load_email, \
    load_amazon
from settings import PATH_TO_DUMPS, PATH_TO_BLOG_CATALOG

import numpy as np
import time


def run_downhill(adjacency_matrix, N, dim, l, neg_sampling, batch_size, batch_count, sec_ord):
    if sec_ord:
        hist_loss = HistLossWeighted(N, l=l, dim=dim, neg_sampling=neg_sampling, bin_num=10)
        hist_loss.setup()
    else:
        hist_loss = HistLoss(N, l=l, dim=dim, neg_sampling=neg_sampling)
        hist_loss.setup()

    def get_batch():
        batch_indxs = np.random.choice(a=N, size=batch_size).astype('int32')
        A_batched = adjacency_matrix[batch_indxs].astype('float32')

        pos_count = np.count_nonzero(A_batched[:, batch_indxs])
        # pos_count = len(A_batched[:, batch_indxs].nonzero()[0])
        neg_count = batch_size * (batch_size - 1) - pos_count
        if sec_ord:
            pass
        else:
            neg_sampling_indxs = np.random.choice(a=neg_count,
                                              size=pos_count*2).astype('int32')

        return (
            batch_indxs,
            # neg_sampling_indxs,
            A_batched
        )

    downhill.minimize(
        hist_loss.loss,
        algo='adagrad',
        train=get_batch,
        inputs=[
            hist_loss.batch_indxs,
            # hist_loss.neg_sampling_indxs,
            hist_loss.A_batched
        ],
        params=[hist_loss.b, hist_loss.w],
        monitor_gradients=True,
        learning_rate=0.01,
        train_batches=batch_count,
        valid_batches=batch_count*2,
        patience=5
    )
    return hist_loss.w.get_value(), hist_loss.b.get_value()


if __name__ == '__main__':
    print('Reading graph')
    t = time.time()
    dims = [32]
    ls = [0]

    neg_sampling = False
    sec_ord, k = True, 1
    batch_size = 1000
    batch_count = 10

    graph, name = load_blog_catalog()
    nodes = graph.nodes()
    adjacency_matrix_sparse = nx.to_scipy_sparse_matrix(graph, nodes, format='csr')
    N = adjacency_matrix_sparse.shape[0]
    print(time.time() - t)
    adjacency_matrix = adjacency_matrix_sparse.toarray()
    print(time.time() - t)
    # if sec_ord:
    #     # S = np.dot(adjacency_matrix, adjacency_matrix)
    #     # S = np.linalg.matrix_power(adjacency_matrix, 2)
    #     S = np.load('{}/sec_ord.npy'.format(PATH_TO_BLOG_CATALOG))
    #     print(time.time() - t)
    #     K = np.linalg.norm(adjacency_matrix, axis=0)
    #     print(time.time() - t)
    #     R = np.divide(np.transpose(np.divide(S, K)), K) - np.eye(N)
    #     print(time.time() - t)
    #     adjacency_matrix = (adjacency_matrix + R * k)/(1+k)

    # adj_array = adjacency_matrix.toarray()
    print(time.time() - t)
    print(adjacency_matrix[:5, :5])

    for dim, l in product(dims, ls):
        print(("Generating embedding with method=hist_loss, dim={}, " +
              "dataset={}, batch_size={}, batch_count={}")
              .format(dim, name, batch_size, batch_count))
        w, b = run_downhill(adjacency_matrix, N, dim, l, neg_sampling, batch_size, batch_count, sec_ord)
        E = np.dot(adjacency_matrix, w) + b
        E_norm = E / np.linalg.norm(E, axis=1).reshape((E.shape[0], 1))
        filename = '{}/models/hist_loss'.format(PATH_TO_DUMPS)
        if l != 0:
            filename += '_l{}'.format(l)
        if sec_ord:
            filename += '_so{}'.format(k)
        filename += '_{}_d{}.csv'.format(name, dim)
        print('Saving results to {}'.format(filename))
        with open(filename, 'w') as file:
            file.write('{} {}\n'.format(N, dim))
            for i in range(N):
                file.write(str(i+1) + ' ' + ' '.join([str(x) for x in E_norm[i]]) + '\n')
