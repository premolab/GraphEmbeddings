from codes.models.OptGraph2vec import OptLossGraph2vec
from .HistLossTransformer import HistLossTransformer


class OptTransformer(HistLossTransformer):
    def __init__(self, nx_G, name, d, pos_scale=10, neg_scale=1e-3, seed=None, load_dumped_model=True, dump_model=True, stoc=None):
        super().__init__(nx_G, name, d, seed, load_dumped_model, dump_model, stoc)
        self.model = OptLossGraph2vec(self.nx_G, d, fit=False, pos_scale=pos_scale, neg_scale=neg_scale,
                                      save_intern_name_prefix=self.dir_name + "opt_" + name)

        self.pos_scale = pos_scale
        self.neg_scale = neg_scale
        self.cvparams = {
            'embedding__pos_scale': [1, 10, 100, 1000],
            'embedding__neg_scale': [1e-4, 1e-3, 1e-2, 0.1],
        }

    def get_dump_model_filename(self):
        template = './dumps/models/{}opt2v_{}_d{}_ps{}_ns{}.dump'
        return template.format('stoc_' if self.stoc else '', self.name, self.d,
                               self.pos_scale, self.neg_scale)

    def _fit_model(self, **fit_params):
        self.model.neg_scale = self.neg_scale
        self.model.neg_scale = self.pos_scale
        self.model.optimize(**fit_params)
        self.nodes = self.model.nodes
        self.embedding = self.model.embedding
        self.calc_node2indx(self.nodes)