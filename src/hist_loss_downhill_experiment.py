import downhill
import numpy as np
import networkx as nx
import pickle

from itertools import product

from lib.hist_loss.HistLoss import HistLoss
from load_data import load_blog_catalog, load_karate, load_football, load_stars, load_polbooks, load_protein, load_email, \
    load_amazon
from settings import PATH_TO_DUMPS

import time


def run_downhill(adj_array, N, dim, l, neg_sampling, batch_size, batch_count):
    hist_loss = HistLoss(N, l=l, dim=dim, neg_sampling=neg_sampling)
    hist_loss.setup()

    def get_batch():
        batch_indxs = np.random.choice(a=N, size=batch_size)
        pos_mask = adj_array[batch_indxs][:, batch_indxs]
        pos_count = np.count_nonzero(pos_mask)
        neg_count = batch_size * (batch_size - 1) - pos_count
        neg_sampling_indxs = np.random.choice(a=neg_count, size=pos_count*2)
        return batch_indxs, neg_sampling_indxs, adj_array

    downhill.minimize(
        hist_loss.loss,
        train=get_batch,
        inputs=[hist_loss.batch_indxs, hist_loss.neg_sampling_indxs, hist_loss.A],
        params=[hist_loss.b, hist_loss.w],
        monitor_gradients=True,
        learning_rate=0.1,
        train_batches=batch_count
    )
    return hist_loss.w.get_value(), hist_loss.b.get_value()


if __name__ == '__main__':
    print('Reading graph')
    t = time.time()
    dims = [2, 32, 64, 128]
    ls = [0]

    neg_sampling = True
    batch_size = 100
    batch_count = 100

    graph, name = load_amazon()
    nodes = graph.nodes()
    adjacency_matrix = nx.adjacency_matrix(graph, nodes).astype('float64')
    N = adjacency_matrix.shape[0]
    adj_array = adjacency_matrix.toarray()
    print(time.time() - t)

    for dim, l in product(dims, ls):
        w, b = run_downhill(adj_array, N, dim, l, neg_sampling, batch_size, batch_count)
        E = np.dot(adj_array, w) + b
        E_norm = E / np.linalg.norm(E, axis=1).reshape((E.shape[0], 1))
        if l != 0:
            filename = '{}/models/hist_loss_l{}_{}_d{}.csv'\
                .format(PATH_TO_DUMPS, l, name, dim)
        else:
            filename = '{}/models/hist_loss_{}_d{}.csv'\
                .format(PATH_TO_DUMPS, name, dim)
        print('Saving results to {}'.format(filename))
        with open(filename, 'w') as file:
            file.write('{} {}\n'.format(N, dim))
            for i in range(N):
                file.write(str(i+1) + ' ' + ' '.join([str(x) for x in E_norm[i]]) + '\n')
