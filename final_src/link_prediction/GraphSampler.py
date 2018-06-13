import networkx as nx
import numpy as np


class GraphSampler:

    def __init__(self, graph, split_ratio):
        self.graph = graph
        self.split_ratio = split_ratio

    def fit_transform(self):
        sampled_graph = GraphSampler.sample_graph(
            self.graph,
            int(len(self.graph.edges()) * self.split_ratio)
        )
        return sampled_graph

    @staticmethod
    def sample_graph(graph: nx.Graph, expected_number_of_edges, seed=43):
        res_graph = graph.copy()
        assert nx.is_connected(graph)
        np.random.seed(seed)
        iters = 0
        while len(res_graph.edges()) > expected_number_of_edges:
            iters += 1
            edges = np.array(res_graph.edges())
            batch_size = min(300, len(res_graph.edges()) - expected_number_of_edges)
            edges_to_delete = edges[np.random.choice(np.arange(len(edges)), batch_size)]
            res_graph.remove_edges_from(edges_to_delete)
            if not nx.is_connected(res_graph):
                res_graph.add_edges_from(edges_to_delete)
            if iters % 100 == 1:
                print(iters, "left", len(res_graph.edges()) - expected_number_of_edges)
            if iters > 10000:
                break
        print("done", iters, "left", len(res_graph.edges()) - expected_number_of_edges)
        for edge in res_graph.edges():
            if 'weight' in graph[edge[0]][edge[1]]:
                res_graph[edge[0]][edge[1]]['weight'] = graph[edge[0]][edge[1]]['weight']
        return res_graph

