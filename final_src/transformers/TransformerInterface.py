import os
import sys
import random

from sklearn.base import BaseEstimator, TransformerMixin


class TransformerInterface(BaseEstimator, TransformerMixin):
    def __init__(self, nx_G, name, d, seed=None, load_dumped_model=True, dump_model=True):
        self.name = name
        self.nx_G = nx_G
        self.d = d
        self.load_dumped_model = load_dumped_model
        self.dump_model = dump_model
        self.embedding = None
        self.fitted = False
        self.cvparams = {}
        if seed is not None:
            random.seed(seed)

    def load_model(self, dump_name):
        raise NotImplemented()

    def save_model(self, dump_name):
        raise NotImplemented()

    def _fit_model(self, **fit_params):
        raise NotImplemented()

    def transform(self, nodes, **fit_params):
        raise NotImplemented()

    def get_dump_model_filename(self):
        raise NotImplemented()

    def fit(self, nodes=None, y=None, **fit_params):
        if self.fitted:
            return self
        dump_name = self.get_dump_model_filename()
        if self.load_dumped_model:
            try:
                if os.path.exists(dump_name):
                    self.load_model(dump_name)
                    self.fitted = True
                    print("Loaded cached embedding from " + dump_name)
                    return self
            except Exception as e:
                e.args += [dump_name]
                print('Model load failed: {}, err: {}'.format(dump_name, e), file=sys.stderr)
        self._fit_model(**fit_params)
        if self.dump_model:
            self.save_model(dump_name)
        self.fitted = True
        return self

    def fit_transform(self, nodes, y, **fit_params):
        self.fit(nodes, y, **fit_params)
        return self.transform(nodes)

    def calc_node2indx(self, nodes):
        self.node2indx = {node: indx for indx, node in enumerate(nodes)}
