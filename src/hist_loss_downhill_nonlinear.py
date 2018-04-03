import downhill
import networkx as nx
import pickle

from itertools import product

from lib.hist_loss.HistLoss import HistLoss
from lib.hist_loss.HistLossNonlinear import HistLossNonlinear
from load_data import *
from settings import *

import numpy as np
import time


def run_downhill(adjacency_matrix, N, dim, l, neg_sampling, batch_size, batch_count, detail=False):
    hist_loss = HistLossNonlinear(N, l=l, dim=dim, neg_sampling=neg_sampling)
    hist_loss.setup()

    def get_batch():
        batch_indxs = np.random.choice(a=N, size=batch_size).astype('int32')
        A_batched = adjacency_matrix[batch_indxs].astype('float32')

        pos_count = np.count_nonzero(A_batched[:, batch_indxs])
        # pos_count = len(A_batched[:, batch_indxs].nonzero()[0])
        neg_count = batch_size * (batch_size - 1) - pos_count
        neg_sampling_indxs = np.random.choice(a=neg_count,
                                              size=pos_count*2).astype('int32')
        return (
            batch_indxs,
            neg_sampling_indxs,
            A_batched
        )

    opt = downhill.build(algo='adagrad',
                         loss=hist_loss.loss,
                         inputs=[
                             hist_loss.batch_indxs,
                             hist_loss.neg_sampling_indxs,
                             hist_loss.A_batched
                         ],
                         params=[hist_loss.b1, hist_loss.w1,
                                 hist_loss.b2, hist_loss.w2,
                                 hist_loss.b3, hist_loss.w3],
                         monitor_gradients=True)

    train = downhill.Dataset(
        get_batch,
        name='train',
    )

    for i, _ in enumerate(opt.iterate(train=train,
                                  patience=5,
                                  learning_rate=0.01)):
        if detail:
            E = hist_loss.calc_embedding().eval({
                hist_loss.A_batched: adjacency_matrix.astype('float32'),
                hist_loss.batch_indxs: np.arange(N).astype('int32')
            })
            E_norm = E / np.linalg.norm(E, axis=1).reshape((E.shape[0], 1))
            filename = '{}/models/nonlinear2/{}'.format(PATH_TO_DUMPS, i)
            if l != 0:
                filename += '_l{}'.format(l)
            filename += '_{}_d{}.csv'.format(name, dim)
            print('Saving results to {}'.format(filename))
            with open(filename, 'w') as file:
                file.write('{} {}\n'.format(N, dim))
                for i in range(N):
                    file.write(str(i + 1) + ' ' + ' '.join([str(x) for x in E_norm[i]]) + '\n')

    return hist_loss.w1.get_value(), hist_loss.b1.get_value(), hist_loss.w2.get_value(), hist_loss.b2.get_value()


if __name__ == '__main__':
    print('Reading graph')
    t = time.time()
    dims = [3, 4, 5, 6, 7, 8, 16]
    ls = [0]

    neg_sampling = True
    batch_size = 100
    batch_count = 10

    graph, name = generate_sbm([100, 100, 100], 0.1, 0.01, 43)
    nodes = graph.nodes()
    adjacency_matrix_sparse = nx.to_scipy_sparse_matrix(graph, nodes, format='csr')
    N = adjacency_matrix_sparse.shape[0]
    print(time.time() - t)
    adjacency_matrix = adjacency_matrix_sparse.toarray()

    print(time.time() - t)
    print(adjacency_matrix[:5, :5])

    for dim, l in product(dims, ls):
        print(("Generating embedding with method=hist_loss, dim={}, " +
              "dataset={}, batch_size={}, batch_count={}")
              .format(dim, name, batch_size, batch_count))
        w1, b1, w2, b2 = run_downhill(adjacency_matrix, N, dim, l, neg_sampling, batch_size, batch_count, detail=True)
        E = np.dot(adjacency_matrix, w) + b
        E_norm = E / np.linalg.norm(E, axis=1).reshape((E.shape[0], 1))
        filename = '{}/models/vavs/hist_loss'.format(PATH_TO_DUMPS)
        if l != 0:
            filename += '_l{}'.format(l)
        filename += '_{}_d{}.csv'.format(name, dim)
        print('Saving results to {}'.format(filename))
        with open(filename, 'w') as file:
            file.write('{} {}\n'.format(N, dim))
            for i in range(N):
                file.write(str(i+1) + ' ' + ' '.join([str(x) for x in E_norm[i]]) + '\n')
