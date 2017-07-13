import lasagne
from models.DotLayer import DotLayer
from models.Graph2vec import Graph2vec

from codes.models.HistLossLayer import HistLossLayer


class HistLossGraph2vec(Graph2vec):
    def __init__(self, graph, d, fit=True, bin_num=128,
                 save_intern_embeddings_step=0,
                 save_intern_name_prefix=None, **kwargs):
        self.min_cov = 1e-6
        self.bin_num = bin_num
        super().__init__(graph, d,
                         fit=fit,
                         save_intern_embeddings_step=save_intern_embeddings_step,
                         save_intern_name_prefix=save_intern_name_prefix, **kwargs)

    def init_loss(self, deterministic):
        l_in_adj_minibatch = lasagne.layers.InputLayer(shape=(None, None), input_var=self.adj_minibatch)
        similarities = DotLayer(self.network, self.network)
        return lasagne.layers.get_output(HistLossLayer(similarities, l_in_adj_minibatch), deterministic=deterministic)

    def update_step(self, lr, last_loss, train_loss, bucket, volume):
        mul = (0.5 if lr > 1e-6 else 1) if last_loss < train_loss else 1.2 if bucket < 0.5 * volume and lr < 0.1 else 1
        return mul * lr