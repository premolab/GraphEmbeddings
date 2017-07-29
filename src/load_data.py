import networkx as nx
from settings import PATH_TO_BLOG_CATALOG, PATH_TO_KARATE, PATH_TO_FOOTBALL, PATH_TO_STARS, PATH_TO_POLBOOKS, \
    PATH_TO_PROTEIN, PATH_TO_EMAIL, PATH_TO_AMAZON


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


if __name__ == '__main__':
    print(len(load_protein()[0]))
