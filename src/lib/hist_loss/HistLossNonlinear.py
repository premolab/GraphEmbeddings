import numpy as np

import theano
from theano import tensor as T
from theano.tensor.shared_randomstreams import RandomStreams


class HistLossNonlinear:
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

        self.w1 = theano.shared(np.random.normal(size=(N, dim)), name='w1')
        self.b1 = theano.shared(np.zeros((N, dim)), name='b1')

        self.w2 = theano.shared(np.random.normal(size=(dim, dim)), name='w2')
        self.b2 = theano.shared(np.zeros((N, dim)), name='b2')

        self.w3 = theano.shared(np.random.normal(size=(dim, dim)), name='w3')
        self.b3 = theano.shared(np.zeros((N, dim)), name='b3')

        self.loss = None

    def setup(self):
        self.loss = self.compile_loss()

    def compile_loss(self):
        E = self.calc_embedding()  # shape: (batch_n, dim)
        E_norm = E / E.norm(2, axis=1).reshape((E.shape[0], 1))  # shape: (batch_n, batch_n)
        E_corr = T.dot(E_norm, E_norm.T)  # shape: (batch_n, batch_n)

        pos_mask = self.A_batched[:, self.batch_indxs]  # shape: (batch_n, batch_n)
        neg_mask = 1 - pos_mask - T.eye(pos_mask.shape[0])  # shape: (batch_n, batch_n)

        pos_samples = E_corr[pos_mask.nonzero()]
        neg_samples = E_corr[neg_mask.nonzero()]

        if self.neg_sampling:
            neg_samples = neg_samples[self.neg_sampling_indxs]

        pos_hist = HistLossNonlinear.calc_hist(pos_samples, bin_num=self.bin_num)
        neg_hist = HistLossNonlinear.calc_hist(neg_samples, bin_num=self.bin_num)

        agg_pos = T.extra_ops.cumsum(pos_hist)
        loss = T.sum(T.dot(agg_pos, neg_hist)) - self.l * T.sum(pos_samples)
        return loss

    def calc_embedding(self):
        b1_batched = self.b1[self.batch_indxs]  # shape: (batch_n, dim)
        b2_batched = self.b2[self.batch_indxs]  # shape: (batch_n, dim)
        b3_batched = self.b3[self.batch_indxs]  # shape: (batch_n, dim)

        l1 = T.nnet.nnet.sigmoid(T.dot(self.A_batched, self.w1) + b1_batched)
        l2 = T.nnet.nnet.sigmoid(T.dot(l1, self.w2) + b2_batched)
        E = T.dot(l2, self.w3) + b3_batched

        return E

    @staticmethod
    def calc_hist(samples, bin_num=64):
        """Строит гистограмму с треугольным ядром, формула есть в статье"""
        delta = 2 / (bin_num - 1)
        grid_row = T.arange(-1, 1 + delta, delta)
        grid = T.tile(grid_row, (samples.shape[0], 1))
        samples_grid = T.tile(samples, (grid_row.shape[0], 1)).T
        dif = T.abs_(samples_grid - grid)
        mask = dif < delta
        return T.dot(mask.T, delta - dif).diagonal() / (delta * samples.shape[0] + 0.0001)
