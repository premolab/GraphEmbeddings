import warnings

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim.models import KeyedVectors
from codes.models import deepwalk
from .Node2VecTransformer import Node2VecTransformer


class DeepWalkTransformer(Node2VecTransformer):
    def __init__(self, nx_G, name, d, seed=None, load_dumped_model=True, dump_model=True):
        super().__init__(nx_G, name, d, seed, load_dumped_model, dump_model)
        self.name = name
        self.cvparams = {}

    def get_dump_model_filename(self):
        return './dumps/models/deepwalk2v_' + self.name + '_d{}.dump'.format(self.d)

    def args(self, output):
        return deepwalk.parse_args(output=output, representation_size=self.d)

    def _fit_model(self, **fit_params):
        args = self.args(self.get_dump_model_filename())
        model = deepwalk.run(args, self.nx_G)
        self.embedding = model.wv
