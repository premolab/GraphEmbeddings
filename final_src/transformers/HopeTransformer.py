from pathlib import Path

import networkx as nx
import numpy as np
import scipy.sparse.linalg as lg

from io_utils.embedding import path_to_embedding, read_embedding, save_embedding


class HopeTransformer:

    def __init__(self,
                 graph,
                 graph_name,
                 dimension,
                 path_to_dumps,
                 beta=0.01,
                 use_cached=True):
        self.beta = beta
        self.graph = graph
        self.graph_name = graph_name
        self.dim = dimension
        self.use_cached = use_cached
        self.path_to_dumps = path_to_dumps

    def fit(self):
        path = path_to_embedding(
            root=self.path_to_dumps,
            method='hope',
            name=self.graph_name,
            dim=self.dim
        )
        if self.use_cached:
            if Path(path).exists():
                E = read_embedding(path)
                print("Loaded cached embedding from " + path)
                return E

        E = self.learn_embedding(
        )
        save_embedding(
            path,
            E=np.array(E)
        )

    def learn_embedding(self):
        if not self.graph:
            raise Exception('graph is None')

        A = nx.to_numpy_matrix(self.graph)
        M_g = np.eye(self.graph.number_of_nodes()) - self.beta * A
        M_l = self.beta * A
        S = np.dot(np.linalg.inv(M_g), M_l)

        u, s, vt = lg.svds(S, k=self.dim // 2)
        X1 = np.dot(u, np.diag(np.sqrt(s)))
        X2 = np.dot(vt.T, np.diag(np.sqrt(s)))
        E = np.concatenate((X1, X2), axis=1)

        p_d_p_t = np.dot(u, np.dot(np.diag(s), vt))
        eig_err = np.linalg.norm(p_d_p_t - S)
        print('SVD error (low rank): %f' % eig_err)
        return E
