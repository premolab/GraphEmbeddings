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
    methods = ['deepwalk', 'node2vec']
    dimensions = [2, 32, 64, 128]
    graph, name = load_amazon(weighted=True)
    for method, dimension in product(methods, dimensions):
        generate_embedding(graph, method, dimension, name)

    graph, name = load_dblp(weighted=True)
    for method, dimension in product(methods, dimensions):
        generate_embedding(graph, method, dimension, name)
