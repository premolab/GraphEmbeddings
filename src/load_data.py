import networkx as nx
from src.settings import PATH_TO_BLOG_CATALOG


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


def load_blog_catalog():
    graph_filename = '{}/data/edges.csv'.format(PATH_TO_BLOG_CATALOG)
    graph = read_graph(graph_filename)
    return graph

    # comms_filename = '{}/data/group-edges.csv'.format(PATH_TO_BLOG_CATALOG)
    # comms = pd.DataFrame.from_dict(dict(list(map(int, line.split())) for line in open(comms_filename)), orient='index')
    # return graph, comms, "BlogCatalog"
