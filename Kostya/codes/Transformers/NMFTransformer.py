import networkx as nx
from sklearn.decomposition import NMF

from .BaseTransformer import BaseTransformer


class NMFTransformer(BaseTransformer):
    def __init__(self, nx_G, name, d, seed=None, load_dumped_model=True, dump_model=True):
        super().__init__(nx_G, name, d, seed, load_dumped_model, dump_model)
        self.node2indx = None
        self.nodes = None

        if isinstance(self.nx_G, nx.Graph):
            self.n = len(self.nx_G)
            self.nodes = self.nx_G.nodes()
            self.adjacency_matrix = nx.adjacency_matrix(self.nx_G, self.nodes).todense().astype(bool)
        else:
            self.n = self.nx_G.shape[0]
            self.nodes = list(range(self.n))
            self.adjacency_matrix = self.nx_G

        self.model = NMF(self.d)

    def get_dump_model_filename(self):
        return './dumps/models/nmf2v_{}_d{}.dump'.format(self.name, self.d)

    def _fit_model(self, **fit_params):
        self.model.fit(self.adjacency_matrix, **fit_params)
        self.embedding = self.model.components_.T
        self.calc_node2indx(self.nodes)
