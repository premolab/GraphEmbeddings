import networkx as nx
import numpy as np
import pandas as pd

from codes.models import LancichinettiBenchmark


def get_easy_datasets():
    return [
        # LoadFootball(),
        LoadKarate(),
        # generate_graphs_2(200, 2, 0.1, 0.01),
        # generate_graphs_2(200, 2, 0.2, 0.01),
        # generate_graphs_2(200, 2, 0.2, 0.1),
        # generate_graphs_2(200, 5, 0.2, 0.01),
    ]


def get_hard_datasets():
    return [
            # LoadPPImat(),
            # LoadPPI(),
            # LoadLancichinettiBenchmark(),
            # LoadLancichinettiBenchmark2(),
            LoadBlogCatalog(),
        ]

def generate_graphs_2(n, m, p, qs, seed=100):
    np.random.seed(seed=seed)
    W = np.random.rand(n, n)
    s1 = int(n / m)

    if not isinstance(qs, list):
        q = qs
        block_model = np.kron(np.eye(m), np.ones((s1, s1))) * (p - q) + q
        A = 1.0 * (W < block_model)
        true_comm = pd.DataFrame(np.kron(np.arange(m), np.ones(s1)))
        return nx.from_numpy_matrix(A), true_comm, 'l_partition-n{}-m{}-p{}-q{}-s{}'.format(n, m, p, q, seed)


def generate_graphs(n, frac, p, qs, seed=100):
    np.random.seed(seed=seed)

    W = np.random.rand(n, n)
    s1 = int(n * frac)
    s2 = n - s1

    for q in qs:
        g_in1 = p * np.ones((s1, s1))
        g_in2 = p * np.ones((s2, s2))
        g_out1 = q * np.ones((s1, s2))
        g_out2 = q * np.ones((s2, s1))
        block_model = np.bmat([[g_in1, g_out1], [g_out2, g_in2]])
        A = 1.0 * (W < block_model)
        true_comm = pd.DataFrame(np.concatenate([np.ones((s1, 1)), -np.ones((s2, 1))]))
        yield nx.from_numpy_matrix(A), true_comm, 'l_partition-n{}-m2-p{}-q{}-s{}'.format(n, p, q, seed)


def read_graph(input_filepath, directed=False):
    '''
    Reads the input network in networkx.
    '''

    weighted = False
    with open(input_filepath) as f:
        for line in f:
            if len(line.split()) == 3:
                weighted = True
            break

    if weighted:
        G = nx.read_edgelist(input_filepath, nodetype=int, data=(('weight', float),), create_using=nx.DiGraph())
    else:
        G = nx.read_edgelist(input_filepath, nodetype=int, create_using=nx.DiGraph())
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1

    if not directed:
        G = G.to_undirected()

    return G

def LoadFootball():
    graph_filename = '../graphs/football.gml'
    G = nx.read_gml(graph_filename)
    comms = {key: G.node[key]['value'] for key in G.node}
    comms = pd.DataFrame.from_dict(comms, orient='index')
    for edge in G.edges():
        G[edge[0]][edge[1]]['weight'] = 1
    return G, comms, "football"

def LoadBlogCatalog():
    graph_filename = '../graphs/node2vec_exp_data/BlogCatalog-dataset/data/edges.csv'
    comms_filename = '../graphs/node2vec_exp_data/BlogCatalog-dataset/data/group-edges.csv'

    comms = pd.DataFrame.from_dict(dict(list(map(int, line.split())) for line in open(comms_filename)), orient='index')
    return read_graph(graph_filename), comms, "BlogCatalog"


def LoadPPI():
    graph_filename = '../graphs/node2vec_exp_data/PPI.edges'
    labels_filename = '../graphs/node2vec_exp_data/PPI.labels'

    labels = pd.DataFrame.from_csv(labels_filename, header=None, index_col=None)
    labels.index += 1
    return read_graph(graph_filename), labels, 'PPI'

def LoadPPImat():
    import scipy.io
    filename = '../graphs/node2vec_exp_data/Homo_sapiens.mat'
    mat = scipy.io.loadmat(filename)

    labels = pd.DataFrame(mat['group'].todense())
    return nx.Graph(mat['network']), labels, 'PPImat'

def LoadKarate():
    graph_filename = '../graphs/karate.edgelist'
    labels_filename = '../graphs/karate.mylabels'

    labels = pd.DataFrame.from_csv(labels_filename, header=None, index_col=None)
    # labels.index += 1
    return read_graph(graph_filename), labels, 'karate'

def LoadWikipedia():
    graph_filename = '../graphs/wiki.edges'
    labels_filename = '../graphs/karate.mylabels'

    labels = pd.DataFrame.from_csv(labels_filename, header=None, index_col=None)
    labels.index += 1
    return read_graph(graph_filename), labels, 'Wikipedia'

def LoadFacebook():
    graph_filename = '../graphs/wiki.edges'
    return read_graph(graph_filename), None, 'Facebook'

def LoadArXiv():
    graph_filename = '../graphs/wiki.edges'
    return read_graph(graph_filename), None, 'arXiv'

def LoadLancichinettiBenchmark(**data_params):
    seed = 21113222
    import os
    print(os.getcwd())
    with open(r'../external/Lancichinetti_benchmark/time_seed.dat', 'w') as f:
        f.write(str(seed))
    data_params_def = {
        'N': 1000,
        'mut': 0.1,
        'maxk': 150,
        'k': 75,
        'om': 2,
        'muw': 0.1,
        'beta': 2,
        't1': 2,
        't2': 2,
        'on': 100,
    }
    data_params_def.update(data_params)
    G, comms = LancichinettiBenchmark(**data_params_def)
    ans = dict()

    for i, x in enumerate(comms):
        for j in comms[x]:
            if j not in ans:
                ans[j] = [0]*len(comms)
            ans[j][i] = 1

    comms = pd.DataFrame.from_dict(ans, orient='index')
    return G, comms, 'lancichinetti'