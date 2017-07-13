from lib import deepwalk

import networkx as nx

dimensions = [128, 256]


for d in dimensions:
    args = deepwalk.parse_args(output='deepwalk_emb/blog_dim{}.csv'.format(d), representation_size=d)
    G = nx.read_edgelist('../data/graphs/BlogCatalog/data/edges.csv', nodetype=int)
    print(deepwalk.run(args, G))
