from itertools import product

from load_data import *
from transformers import transformers

from settings import PATH_TO_DUMPS


def generate_embedding(graph, method, dimension, name):
    print("Starting embedding with method={} dim={}".format(method, str(dimension)))
    embedding_transformer = transformers[method](
        graph, name, dimension, seed=43, path_to_dumps=PATH_TO_DUMPS, dump_model=True
    )

    embedding_transformer.fit()


if __name__ == '__main__':
    # available methods:
    # 'stoc_hist', 'node2vec', 'stoc_opt', 'stocsk_opt',
    # 'bigclam', 'gamma', 'deepwalk', 'svd', 'nmf'
    methods = ['deepwalk']
    dimensions = [128]
    graphs = [load_blog_catalog(weighted=False)]

    p_outs = []
    sizes = []
    for p_out, size in product(p_outs, sizes):
        graphs += generate_sbm([size, size, size], 0.1, p_out, seed=43)

    for method, dimension, graph in product(methods, dimensions, graphs):
        generate_embedding(graph.graph, method, dimension, graph.name)
