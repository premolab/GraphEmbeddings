import networkx as nx
import numpy as np
from settings import *
import parse


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


def load_blog_catalog(weighted=False):
    # graph_filename = '{}/data/edges.csv'.format(PATH_TO_BLOG_CATALOG)
    # graph = read_graph(graph_filename)
    # return graph
    G = nx.read_edgelist(
        '{}/data/edges.csv'.format(PATH_TO_BLOG_CATALOG), nodetype=int
    )
    if weighted:
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1
    return G, 'BlogCatalog'


def load_karate(weighted=False):
    graph_filename = '{}/karate.edgelist'.format(PATH_TO_KARATE)
    G = nx.read_edgelist(graph_filename, nodetype=int)
    if weighted:
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1
    return G, 'Karate'


def load_football(weighted=False):
    graph_filename = '{}/football.txt'.format(PATH_TO_FOOTBALL)
    G = nx.read_edgelist(graph_filename, nodetype=int)
    if weighted:
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1
    return G, 'Football'


def load_stars(weighted=False):
    graph_filename = '{}/stars.txt'.format(PATH_TO_STARS)
    G = nx.read_edgelist(graph_filename, nodetype=int)
    if weighted:
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1
    return G, 'Stars'


def load_polbooks(weighted=False):
    graph_filename = '{}/polbooks.txt'.format(PATH_TO_POLBOOKS)
    G = nx.read_edgelist(graph_filename, nodetype=int)
    if weighted:
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1
    return G, 'PolBooks'


def load_protein():
    graph_filename = '{}/protein_new.txt'.format(PATH_TO_PROTEIN)
    G = nx.read_weighted_edgelist(graph_filename, nodetype=int)
    return G, 'Protein'


def load_email(weighted=False):
    graph_filename = '{}/email-Eu-core.txt'.format(PATH_TO_EMAIL)
    G = nx.read_edgelist(graph_filename, nodetype=int)
    if weighted:
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1
    return G, 'Email'


def load_amazon(weighted=False):
    graph_filename = '{}/amazon.txt'.format(PATH_TO_AMAZON)
    G = nx.read_edgelist(graph_filename, nodetype=int)
    if weighted:
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1
    return G, 'Amazon'


def load_dblp(weighted=False):
    graph_filename = '{}/dblp.txt'.format(PATH_TO_AMAZON)
    G = nx.read_edgelist(graph_filename, nodetype=int)
    if weighted:
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1
    return G, 'DBLP'


def generate_sbm(sizes, p_in: float, p_out: float, seed, weighted=False):
    G = nx.random_partition_graph(sizes, p_in, p_out, seed)
    if weighted:
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1
    return G, 'SBM_sizes_{}_p_in_{}_p_out_{}_seed_{}'.format(
               '_'.join(str(x) for x in sizes),
               p_in, p_out, seed
           )


def generate_sbm_partition(name: str):
    sizes, p_in, p_out, seed = parse.parse('SBM_sizes_{}_p_in_{}_p_out_{}_seed_{}', name)
    return nx.random_partition_graph(
        [int(x) for x in sizes.split('_')],
        float(p_in),
        float(p_out),
        int(seed)
    ).graph['partition']


if __name__ == '__main__':
    print(generate_sbm_partition('SBM_sizes_100_100_100_p_in_0.1_p_out_0.01_seed_43'))
