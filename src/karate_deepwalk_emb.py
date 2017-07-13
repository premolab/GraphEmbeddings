from lib import deepwalk

import networkx as nx

dimensions = [16, 32, 64]


for d in dimensions:
    args = deepwalk.parse_args(output='deepwalk_emb/karate_{}.csv'.format(d), representation_size=d)
    G = nx.read_edgelist('../data/graphs/karate.edgelist', nodetype=int)
    print(deepwalk.run(args, G))
