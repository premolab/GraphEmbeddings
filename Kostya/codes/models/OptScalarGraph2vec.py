import lasagne
from theano import tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from codes.models.Graph2vec import Graph2vec


class OptScalarLossGraph2vec(Graph2vec):
    def __init__(self, graph, d, pos_scale=10, neg_scale=0.001, fit=True,
                 save_intern_embeddings_step=0,
                 save_intern_name_prefix=None, **kwargs):
        self.srng = RandomStreams(seed=234)
        self.pos_scale = pos_scale
        self.neg_scale = neg_scale
        super().__init__(graph, d, fit=fit,
                         save_intern_embeddings_step=save_intern_embeddings_step,
                         save_intern_name_prefix=save_intern_name_prefix, **kwargs)

    def dist_mtx(self, emb_minibatch):
        sim = emb_minibatch.dot(emb_minibatch.T)
        return sim

    def get_loss(self, sim_pos, sim_neg):
        part_pos = self.pos_scale * T.sum((sim_pos - 1) ** 2)
        part_neg = self.neg_scale * T.sum((sim_neg + 1) ** 2) + T.sum(-T.log(1e-20 + 1 - sim_neg))

        return part_pos + part_neg

    def init_loss(self, deterministic):
        network_out = lasagne.layers.get_output(self.network, deterministic=deterministic)
        sim = self.dist_mtx(network_out)

        pos_mask = self.adj_minibatch
        neg_mask = 1 - pos_mask - T.eye(pos_mask.shape[0])

        sim_pos = sim[pos_mask.nonzero()]
        sim_neg = sim[neg_mask.nonzero()]
        neg_sampling = self.srng.permutation(n=sim_neg.shape[0], size=(1,))[0, :2 * sim_pos.shape[0]]
        sim_neg = sim_neg[neg_sampling]

        return self.get_loss(sim_pos, sim_neg)

    def update_step(self, lr, last_loss, train_loss, bucket, volume):
        mul = 0.999
        return mul * lr
