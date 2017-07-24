import numpy as np

import theano
from theano import tensor as T
from theano.tensor.shared_randomstreams import RandomStreams


class HistLoss:
    def __init__(self, N, dim, bin_num=64, neg_sampling=True, seed=None):
        self.N, self.dim = N, dim
        self.bin_num = bin_num
        self.neg_sampling = neg_sampling
        self.srng = RandomStreams(seed=seed)

        self.A = T.dmatrix('A')
        self.A_batched = T.dmatrix('A_batched')
        self.pos_mask = T.dmatrix('pos_mask')
        self.batch_indxs = T.vector('batch_indxs', dtype='int64')

        self.w = theano.shared(np.random.normal(size=(N, dim)), name='w')
        self.b = theano.shared(np.zeros((N, dim)), name='b')
        self.loss = None

    def setup(self):
        self.loss = self.compile_loss(self.A, self.w, self.b, self.batch_indxs)

    def compile_loss(self, A, w, b, batch_indxs):
        A_batched = A[batch_indxs]  # shape: (batch_n, N)
        b_batched = b[batch_indxs]  # shape: (batch_n, dim)

        E = T.dot(A_batched, w) + b_batched  # shape: (batch_n, dim)
        E_norm = E / E.norm(2, axis=1).reshape((E.shape[0], 1))  # shape: (batch_n, batch_n)
        E_corr = T.dot(E_norm, E_norm.T)  # shape: (batch_n, batch_n)

        pos_mask = A_batched[:, self.batch_indxs]  # shape: (batch_n, batch_n)
        neg_mask = 1 - pos_mask - T.eye(pos_mask.shape[0])  # shape: (batch_n, batch_n)

        pos_samples = E_corr[pos_mask.nonzero()]
        neg_samples = E_corr[neg_mask.nonzero()]

        if self.neg_sampling:
            neg_samples = neg_samples[self.srng.choice(
                size=(2 * pos_samples.shape[0],),
                a=neg_samples.shape[0]
            )]
        pos_hist = HistLoss.calc_hist_map(pos_samples, bin_num=self.bin_num)
        neg_hist = HistLoss.calc_hist_map(neg_samples, bin_num=self.bin_num)

        agg_pos = T.extra_ops.cumsum(pos_hist)
        loss = T.sum(T.dot(agg_pos, neg_hist))
        return loss

    # works slower but gives nonzero gradient
    @staticmethod
    def calc_hist_map(samples, bin_num=64):
        delta = 2 / (bin_num - 1)
        # ts =  -1 + delta * T.arange(bin_num)
        rs = T.floor((samples + 1) / delta) + 1  # r is natural index
        ts = -1 + delta * (rs - 1)  # t_r is value between -1 and 1
        samples_sub_t_prev = samples - ts
        t_next_sub_samples = delta - samples_sub_t_prev

        H, _ = theano.map(
            lambda r: (
                T.sum(samples_sub_t_prev[T.eq(rs, r).nonzero()]) +
                T.sum(t_next_sub_samples[T.eq(rs - 1, r).nonzero()])
            ),
            np.arange(1, bin_num + 1)
        )
        return H / (delta * samples.shape[0])

    @staticmethod
    def calc_hist(samples, bin_num=64):
        delta = 2 / (bin_num - 1)
        grid_row = T.arange(-1, 1 + delta, delta)
        grid = T.tile(grid_row, (samples.shape[0], 1))
        samples_grid = T.tile(samples, (grid_row.shape[0], 1)).T
        dif = T.abs_(samples_grid - grid)
        return T.sum(dif < delta, axis=0)
