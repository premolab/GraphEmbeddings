import downhill
import networkx as nx
import pickle

from itertools import product

from lib.hist_loss.HistLoss import HistLoss
from load_data import *
from settings import *

import numpy as np


def run_downhill(
        A=None,
        dim=128,
        l=0,
        neg_sampling=False,
        batch_size=100,
        batch_count=10,
        detail=False
):
    N = A.shape[0]
    hist_loss = HistLoss(N, l=l, dim=dim, neg_sampling=neg_sampling)
    hist_loss.setup()

    def get_batch():
        batch_indxs = np.random.choice(a=N, size=batch_size).astype('int32')
        A_batched = A[batch_indxs].astype('float32')

        pos_count = np.count_nonzero(A_batched[:, batch_indxs])
        # pos_count = len(A_batched[:, batch_indxs].nonzero()[0])
        neg_count = batch_size * (batch_size - 1) - pos_count
        neg_sampling_indxs = np.random.choice(a=neg_count,
                                              size=pos_count*2).astype('int32')
        return (
            batch_indxs,
            # neg_sampling_indxs,
            A_batched
        )

    opt = downhill.build(algo='adagrad',
                         loss=hist_loss.loss,
                         inputs=[
                             hist_loss.batch_indxs,
                             # hist_loss.neg_sampling_indxs,
                             hist_loss.A_batched
                         ],
                         params=[hist_loss.b, hist_loss.w],
                         monitor_gradients=True)

    train = downhill.Dataset(
        get_batch,
        name='train',
    )

    for i, _ in enumerate(opt.iterate(
            train=train,
            patience=5,
            learning_rate=0.01
    )):
        if detail:
            w, b = hist_loss.w.get_value(), hist_loss.b.get_value()
            E = np.dot(A, w) + b
            E_norm = E / np.linalg.norm(E, axis=1).reshape((E.shape[0], 1))
            filename = '{}/models/d3/{}'.format(PATH_TO_DUMPS, i)
            if l != 0:
                filename += '_l{}'.format(l)
            filename += '_{}_d{}.csv'.format(name, dim)
            print('Saving results to {}'.format(filename))
            with open(filename, 'w') as file:
                file.write('{} {}\n'.format(N, dim))
                for i in range(N):
                    file.write(str(i + 1) + ' ' + ' '.join([str(x) for x in E_norm[i]]) + '\n')

    return hist_loss.w.get_value(), hist_loss.b.get_value()


if __name__ == '__main__':
    print('Reading graph')
    dims = [3, 4, 5, 6, 7, 8, 12, 16, 24, 32, 64, 128]

    # graph, name = generate_sbm([300, 300, 300], 0.1, 0.01, 43)
    graph, name = load_email()
    A = nx.adjacency_matrix(graph).toarray()
    N = A.shape[0]
    D = np.zeros(A.shape)
    for i in range(A.shape[0]):
        D[i, i] = np.sum(A[i])

    S = np.dot(np.linalg.inv(D), np.dot(A, A))

    print(S[:5, :5])

    for dim, in product(dims,):
        print("Generating embedding with method=hist_loss, dim={}, dataset={}".format(dim, name))
        w, b = run_downhill(A=S, dim=dim)
        E = np.dot(S, w) + b
        E_norm = E / np.linalg.norm(E, axis=1).reshape((E.shape[0], 1))
        filename = '{}/models/hist_loss_sim'.format(PATH_TO_DUMPS)
        filename += '_{}_d{}.csv'.format(name, dim)
        print('Saving results to {}'.format(filename))
        with open(filename, 'w') as file:
            file.write('{} {}\n'.format(N, dim))
            for i in range(N):
                file.write(str(i+1) + ' ' + ' '.join([str(x) for x in E_norm[i]]) + '\n')
