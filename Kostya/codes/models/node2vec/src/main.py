'''
Reference implementation of node2vec. 

Author: Aditya Grover

For more details, refer to the paper:
node2vec: Scalable Feature Learning for Networks
Aditya Grover and Jure Leskovec 
Knowledge Discovery and Data Mining (KDD), 2016
'''

import warnings

import networkx as nx

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim.models import Word2Vec

from . import node2vec
from .parse_args import parse_args


def read_graph(args):
    '''
    Reads the input network in networkx.
    '''
    # if args.weighted:
    #     G = nx.read_edgelist(args.input, nodetype=int, data=(('weight', float),), create_using=nx.DiGraph())
    # else:
    #     G = nx.read_edgelist(args.input, nodetype=int, create_using=nx.DiGraph())
    try:
        G = nx.read_edgelist(args.input, nodetype=int, create_using=nx.DiGraph())
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1
    except:
        G = nx.read_edgelist(args.input, nodetype=int, data=(('weight', float),), create_using=nx.DiGraph())

    if not args.directed:
        G = G.to_undirected()

    return G


def learn_embeddings(walks, args):
    '''
    Learn embeddings by optimizing the Skipgram objective using SGD.
    '''
    walks = [list(map(str, walk)) for walk in walks]
    model = Word2Vec(walks, size=args.dimensions, window=args.window_size, min_count=0, sg=1, workers=args.workers,
                     iter=args.iter)
    model.wv.save_word2vec_format(args.output)
    return model


def run(args, nx_G=None, name=''):
    '''
    Pipeline for representational learning for all nodes in a graph.
    '''
    if nx_G is None:
        nx_G = read_graph(args)
    G = node2vec.Graph(nx_G, name, args.directed, args.p, args.q)
    G.preprocess_transition_probs()
    walks = G.simulate_walks(args.num_walks, args.walk_length)
    model = learn_embeddings(walks, args)
    return walks, model


if __name__ == "__main__":
    args = parse_args()
    run(args)
