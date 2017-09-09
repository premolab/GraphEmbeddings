import numpy as np

import theano
from theano import tensor as T
from theano.tensor.shared_randomstreams import RandomStreams


class HistLossWeighted:
    def __init__(self, N, dim, l=0, bin_num=64, neg_sampling=True, seed=234):
        self.N, self.dim = N, dim
        self.l = l
        self.bin_num = bin_num
        self.neg_sampling = neg_sampling
        # self.srng = RandomStreams(seed=seed)

        # self.A = T.dmatrix('A')
        self.A_batched = T.matrix('A_batched')
        # self.pos_mask = T.matrix('pos_mask')
        self.batch_indxs = T.vector('batch_indxs', dtype='int32')
        self.neg_sampling_indxs = T.vector('neg_sampling_indxs', dtype='int32')

        self.w = theano.shared(np.random.normal(size=(N, dim)), name='w')
        self.b = theano.shared(np.zeros((N, dim)), name='b')
        self.loss = None

    def setup(self):
        self.loss = self.compile_loss()

    def compile_loss(self):
        b_batched = self.b[self.batch_indxs]  # shape: (batch_n, dim)

        E = T.dot(self.A_batched, self.w) + b_batched  # shape: (batch_n, dim)
        E_norm = E / E.norm(2, axis=1).reshape((E.shape[0], 1))  # shape: (batch_n, batch_n)
        E_corr = T.dot(E_norm, E_norm.T)  # shape: (batch_n, batch_n)
        # E_corr = theano.printing.Print('E_corr')(E_corr)

        samples = E_corr.flatten()
        weights_matrix = self.A_batched[:, self.batch_indxs]
        pos_weights = weights_matrix.flatten()
        neg_weights = (1 - weights_matrix - T.eye(weights_matrix.shape[0])).flatten()
        # pos_weights = theano.printing.Print('pos_weights')(pos_weights)
        # neg_weights = theano.printing.Print('neg_weights')(neg_weights)

        pos_hist = HistLossWeighted.calc_hist_weighted(samples, pos_weights, bin_num=self.bin_num)
        neg_hist = HistLossWeighted.calc_hist_weighted(samples, neg_weights, bin_num=self.bin_num)
        # pos_hist = theano.printing.Print('pos_hist')(pos_hist)
        # neg_hist = theano.printing.Print('neg_hist')(neg_hist)

        agg_pos = T.extra_ops.cumsum(pos_hist)
        loss = T.sum(T.dot(agg_pos, neg_hist))
        return loss

    @staticmethod
    def calc_hist_weighted(samples, weights, bin_num=64):
        delta = 2 / (bin_num - 1)
        grid_row = T.arange(-1, 1 + delta, delta)
        grid = T.tile(grid_row, (samples.shape[0], 1))
        samples_grid = T.tile(samples, (grid_row.shape[0], 1)).T
        dif = T.abs_(samples_grid - grid)
        mask = dif < delta
        # mask = theano.printing.Print('mask')(mask)
        hist = T.dot(mask.T * weights, delta - dif).diagonal()
        return hist / hist.sum()
        # return T.dot(theano.printing.Print('weighted_mask')(mask.T * weights), delta - dif).diagonal() / (delta * samples.shape[0] + 0.0001)
