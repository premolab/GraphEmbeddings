import warnings

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim.models import KeyedVectors
from codes.models import node2vec
from .BaseTransformer import BaseTransformer


class Node2VecTransformer(BaseTransformer):
    def __init__(self, nx_G, name, d, p=1, q=1, seed=None, load_dumped_model=True, dump_model=True):
        super().__init__(nx_G, name, d, seed, load_dumped_model, dump_model)
        self.p = p
        self.q = q
        self.walks = None
        self.name = name
        pq_grid = [0.25, 0.5, 1, 2, 4]
        self.cvparams = {
            'embedding__p': pq_grid,
            'embedding__q': pq_grid,
        }

    def get_dump_model_filename(self):
        return './dumps/models/n2v_' + self.name + '_p{}_q{}_d{}.dump'.format(self.p, self.q, self.d)

    def args(self, output):
        return node2vec.parse_args(output=output, p=self.p, q=self.q, dimensions=self.d)

    def load_model(self, dump_name):
        self.embedding = KeyedVectors.load_word2vec_format(dump_name)

    def save_model(self, dump_name):
        pass  # it happens in node2vec.run

    def _fit_model(self, **fit_params):
        args = self.args(self.get_dump_model_filename())
        self.walks, model = node2vec.run(args, self.nx_G, self.name)
        self.embedding = model.wv

    def transform(self, nodes, **fit_params):
        str_nodes = nodes.T.astype(str).tolist()[0]
        return self.embedding[str_nodes]
