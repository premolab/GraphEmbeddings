import pickle


from .TransformerInterface import TransformerInterface


class BaseTransformer(TransformerInterface):
    def __init__(self, nx_G, name, d, seed=None, load_dumped_model=True, dump_model=True):
        super().__init__(nx_G, name, d, seed, load_dumped_model, dump_model)

    def load_model(self, dump_name):
        self.nodes, self.embedding = pickle.load(open(dump_name, 'rb'))
        self.calc_node2indx(self.nodes)

    def save_model(self, dump_name):
        pickle.dump((self.nodes, self.embedding), open(dump_name, 'wb'))

    def transform(self, nodes, **fit_params):
        return self.embedding[[self.node2indx[x] for x in nodes[:, 0].tolist()]]
