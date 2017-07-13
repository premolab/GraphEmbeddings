from codes.models import OptScalarLossGraph2vec
from .OptTransformer import OptTransformer


class OptScalarTransformer(OptTransformer):
    def __init__(self, nx_G, name, d, pos_scale=10, neg_scale=1e-3, seed=None, load_dumped_model=True, dump_model=True, stoc=None):
        super().__init__(nx_G, name, d, seed, load_dumped_model, dump_model, stoc)
        self.model = OptScalarLossGraph2vec(self.nx_G, d, fit=False, pos_scale=pos_scale, neg_scale=neg_scale,
                                            save_intern_name_prefix=self.dir_name + "opt_" + name)

    def get_dump_model_filename(self):
        template = './dumps/models/{}scalopt2v_{}_d{}_ps{}_ns{}.dump'
        return template.format('stoc_' if self.stoc else '', self.name, self.d,
                               self.pos_scale, self.neg_scale)