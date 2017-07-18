import lasagne
from theano import tensor as T
from theano.tensor.shared_randomstreams import RandomStreams


class HistLossLayer(lasagne.layers.MergeLayer):
    def __init__(self, similarities, mask, bin_num=64, **kwargs):
        super(HistLossLayer, self).__init__([similarities, mask], **kwargs)
        self.bin_num = bin_num
        self.min_cov = 1e-6
        self.srng = RandomStreams(seed=234)

    def get_output_shape_for(self, input_shapes):
        # (rows of first input x columns of second input)
        return 1

    def get_output_for(self, inputs, **kwargs):
        sim, pos_mask = inputs

        neg_mask = 1 - pos_mask - T.eye(pos_mask.shape[0])

        sim_pos = sim[pos_mask.nonzero()]
        sim_neg = sim[neg_mask.nonzero()]

        neg_sampling = self.srng.permutation(n=sim_neg.shape[0], size=(1,))[0, :2 * sim_pos.shape[0]]
        # neg_sampling = self.srng.choice(size=(2*sim_pos.shape[0], ), a=sim_neg.shape[0])
        sim_neg = sim_neg[neg_sampling]

        return self.loss(sim_pos, sim_neg)

    def make_hist(self, sim):
        hist = self.calc_hist_vals_vector(sim, -1.0, 1.0)
        hist /= hist.sum()
        return hist

    def loss(self, sim_pos, sim_neg):
        hist_pos = self.make_hist(sim_pos)
        hist_neg = self.make_hist(sim_neg)
        agg_pos = T.extra_ops.cumsum(hist_pos)
        return T.sum(T.dot(agg_pos, hist_neg))

    def calc_hist_vals_vector(self, sim, hist_min, hist_max):
        sim_mat = T.tile(sim[:, None], (1, self.bin_num))
        w = max((hist_max - hist_min) / self.bin_num, self.min_cov)
        grid_vals = T.arange(0, self.bin_num) * (hist_max - hist_min) / self.bin_num + hist_min + w / 2.0
        grid = T.tile(grid_vals, (sim.shape[0], 1))
        w_triang = 4.0 * w
        D = T._tensor_py_operators.__abs__(grid - sim_mat)
        mask = (D <= w_triang / 2)
        D_fin = w_triang * (D * (-2.0 / w_triang ** 2) + 1.0 / w_triang) * mask
        hist_corr = T.sum(D_fin, 0) + self.min_cov
        return hist_corr

