import networkx as nx
import numpy as np

from codes.models import BigClamGamma
from .BaseTransformer import BaseTransformer


class GammaSMTransformer(BaseTransformer):
    def __init__(self, nx_G, name, d, seed=None, load_dumped_model=True, dump_model=True):
        super().__init__(nx_G, name, d, seed, load_dumped_model, dump_model)
        self.nodes = nx_G.nodes()
        self.calc_node2indx(self.nodes)
        self.model = BigClamGamma(np.array(nx.to_numpy_matrix(nx_G)), d, initF='rand', processesNo=1)

    def get_dump_model_filename(self):
        return './dumps/models/gammaSM2v_{}_d{}.dump'.format(self.name, self.d)

    def _fit_model(self, **fit_params):
        self.embedding = self.model.fit(self.d)[0]
