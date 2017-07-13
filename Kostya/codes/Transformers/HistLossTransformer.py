import os

from codes.models.HistLossGraph2vec import HistLossGraph2vec
from .BaseTransformer import BaseTransformer


class HistLossTransformer(BaseTransformer):
    def __init__(self, nx_G, name, d, seed=None, load_dumped_model=True, dump_model=True, stoc=None):
        super().__init__(nx_G, name, d, seed, load_dumped_model, dump_model)
        self.node2indx = None
        self.nodes = None
        self.stoc = stoc
        self.dir_name = "./emb/interns/{}/".format(name)
        if not os.path.exists(self.dir_name):
            os.mkdir(self.dir_name)
        self.model = HistLossGraph2vec(self.nx_G, self.d, fit=False, stoc=self.stoc,
                                       save_intern_name_prefix=self.dir_name + "hist_" + name)

    def get_dump_model_filename(self):
        return './dumps/models/{}hist2v_{}_d{}.dump'.format('stoc_' if self.stoc else '', self.name, self.d)

    def load_model(self, dump_name):
        self.nodes, self.embedding = self.model.load(dump_name)
        self.calc_node2indx(self.nodes)

    def save_model(self, dump_name):
        self.model.save(dump_name)

    def _fit_model(self, **fit_params):
        self.model.optimize(**fit_params)
        self.nodes = self.model.nodes
        self.embedding = self.model.embedding
        self.calc_node2indx(self.nodes)
