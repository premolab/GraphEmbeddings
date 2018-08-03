from collections import namedtuple

import networkx as nx
import numpy as np
import parse
import scipy

from settings import *

GraphInfo = namedtuple('GraphInfo', ['graph', 'name'])


def read_graph(path):
    return nx.read_edgelist(path, nodetype=int)


def save_graph(graph: nx.Graph, path):
    nx.write_edgelist(graph, path, data=False)


def load_blog_catalog(weighted=False):

    G = nx.read_edgelist(
        '{}/data/edges.csv'.format(PATH_TO_BLOG_CATALOG), nodetype=int
    )

    if weighted:
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1
    return GraphInfo(G, 'BlogCatalog')


def load_karate(weighted=False):
    graph_filename = '{}/karate.edgelist'.format(PATH_TO_KARATE)
    G = nx.read_edgelist(graph_filename, nodetype=int)
    if weighted:
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1
    return GraphInfo(G, 'Karate')


def load_wikipedia(weighted=False):
    mat = scipy.io.loadmat('{}/POS.mat')


def load_football(weighted=False):
    graph_filename = '{}/football.txt'.format(PATH_TO_FOOTBALL)
    G = nx.read_edgelist(graph_filename, nodetype=int)
    if weighted:
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1
    return GraphInfo(G, 'Football')


def load_stars(weighted=False):
    graph_filename = '{}/stars.txt'.format(PATH_TO_STARS)
    G = nx.read_edgelist(graph_filename, nodetype=int)
    if weighted:
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1
    return GraphInfo(G, 'Stars')


def load_polbooks(weighted=False):
    graph_filename = '{}/polbooks.txt'.format(PATH_TO_POLBOOKS)
    G = nx.read_edgelist(graph_filename, nodetype=int)
    if weighted:
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1
    return GraphInfo(G, 'PolBooks')


def load_email(weighted=False):
    graph_filename = '{}/email-Eu-core.txt'.format(PATH_TO_EMAIL)
    G = nx.read_edgelist(graph_filename, nodetype=int)
    if weighted:
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1
    return GraphInfo(G, 'Email')


def load_cycles(weighted=False):
    graph_filename = '{}/cycles.txt'.format(PATH_TO_CYCLES)
    G = nx.read_edgelist(graph_filename, nodetype=int)
    if weighted:
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1
    return GraphInfo(G, 'Cycles')


def load_cliques(weighted=False):
    graph_filename = '{}/cliques.txt'.format(PATH_TO_CYCLES)
    G = nx.read_edgelist(graph_filename, nodetype=int)
    if weighted:
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1
    return GraphInfo(G, 'Cliques')


def load_scientists():
    graph_filename = '{}/scientists.txt'.format(PATH_TO_CYCLES)
    G = nx.read_weighted_edgelist(graph_filename, nodetype=int)
    # if weighted:
    #     for edge in G.edges():
    #         G[edge[0]][edge[1]]['weight'] = 1
    return GraphInfo(G, 'Scientists')


def load_amazon(weighted=False):
    graph_filename = '{}/amazon.txt'.format(PATH_TO_AMAZON)
    G = nx.read_edgelist(graph_filename, nodetype=int)
    if weighted:
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1
    return GraphInfo(G, 'Amazon')


def load_dblp(weighted=False):
    graph_filename = '{}/dblp.txt'.format(PATH_TO_AMAZON)
    G = nx.read_edgelist(graph_filename, nodetype=int)
    if weighted:
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1
    return GraphInfo(G, 'DBLP')


def load_facebook(weighted=False):
    graph_filename = '{}/facebook.txt'.format(PATH_TO_AMAZON)
    G = nx.read_edgelist(graph_filename, nodetype=int)
    if weighted:
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1
    return GraphInfo(G, 'Facebook')


def load_hepth(part='train', weighted=False):
    filename = '{}/{}.txt.npy'.format(PATH_TO_HEPTH, part)
    train_edges = np.load(filename)
    G = nx.Graph()
    G.add_edges_from(train_edges)
    if weighted:
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1
    return GraphInfo(G, 'HepTh_train_0.5')


def load_ppi(part='train', weighted=False):
    filename = '{}/{}.txt.npy'.format(PATH_TO_PPI, part)
    train_edges = np.load(filename)
    G = nx.Graph()
    G.add_edges_from(train_edges)
    if weighted:
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1
    return GraphInfo(G, 'PPI_train_0.5')


def load_facebook_partial(part='train', weighted=False):
    filename = '{}/{}.npy'.format(PATH_TO_FACEBOOK, part)
    train_edges = np.load(filename)
    G = nx.Graph()
    G.add_edges_from(train_edges)
    if weighted:
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1
    return GraphInfo(G, 'Facebook_train_0.5_seed_43')


def generate_sbm(sizes, p_in: float, p_out: float, seed, weighted=False):
    G = nx.random_partition_graph(sizes, p_in, p_out, seed)
    if weighted:
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1
    return GraphInfo(G, 'SBM_sizes_{}_p_in_{}_p_out_{}_seed_{}'.format(
        '_'.join(str(x) for x in sizes),
        p_in, p_out, seed
    ))


def generate_sbm_partition(name: str):
    sizes, p_in, p_out, seed = parse.parse('SBM_sizes_{}_p_in_{}_p_out_{}_seed_{}', name)
    return nx.random_partition_graph(
        [int(x) for x in sizes.split('_')],
        float(p_in),
        float(p_out),
        int(seed)
    ).graph['partition']


def load_graph(graph_name, weighted=False) -> nx.Graph:
    if graph_name == 'blog_catalog':
        return load_blog_catalog(weighted).graph
    elif graph_name == 'football':
        return load_football(weighted).graph
    elif graph_name == 'karate':
        return load_karate(weighted).graph
    elif graph_name == 'stars':
        return load_stars(weighted).graph
    elif graph_name == 'polbooks':
        return load_polbooks(weighted).graph
    elif graph_name == 'email':
        return load_email(weighted).graph
    elif graph_name == 'facebook':
        return load_facebook(weighted).graph
    elif graph_name == 'sbm-01-001':
        return generate_sbm([300, 300, 300], 0.1, 0.01, 43, weighted=weighted).graph
    elif graph_name == 'sbm-01-003':
        return generate_sbm([300, 300, 300], 0.1, 0.03, 43, weighted=weighted).graph
    elif graph_name == 'sbm-008-003':
        return generate_sbm([300, 300, 300], 0.08, 0.03, 43, weighted=weighted).graph
    elif graph_name == 'cycles':
        return load_cycles(weighted).graph
    elif graph_name == 'cliques':
        return load_cliques(weighted).graph
    elif graph_name == 'scientists':
        return load_scientists().graph
    else:
        raise Exception("Unknown graph name: " + graph_name)
