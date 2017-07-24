import downhill
import numpy as np
import networkx as nx

from lib.hist_loss.HistLoss import HistLoss
from load_data import load_blog_catalog, load_karate


def run_downhill():
    dim = 32
    graph = load_blog_catalog()
    nodes = graph.nodes()
    adjacency_matrix = nx.adjacency_matrix(graph, nodes).astype('float64')
    N = adjacency_matrix.shape[0]
    adj_array = adjacency_matrix.toarray()

    hist_loss = HistLoss(N, dim=dim, neg_sampling=False)
    hist_loss.setup()

    def get_batch():
        return np.random.choice(a=N, size=N / 100), adj_array

    downhill.minimize(
        hist_loss.loss,
        train=get_batch,
        inputs=[hist_loss.batch_indxs, hist_loss.A],
        monitor_gradients=True,
    )
    print(hist_loss.w.get_value())
    print(hist_loss.b.get_value())

if __name__ == '__main__':
    run_downhill()