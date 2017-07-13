from itertools import product

from load_data import load_blog_catalog
from transformers import transformers

from settings import PATH_TO_DUMPS


def generate_embedding(graph, method, dimension):
    print("Starting embedding with method={} dim={}".format(method, str(dimension)))
    embedding_transformer = transformers[method](
        graph, 'BlogCatalog', dimension, seed=43, path_to_dumps=PATH_TO_DUMPS, dump_model=True
    )

    embedding_transformer.fit()


if __name__ == '__main__':
    # available methods:
    # 'stoc_hist', 'node2vec', 'stoc_opt', 'stocsk_opt',
    # 'bigclam', 'gamma', 'deepwalk', 'svd', 'nmf'
    methods = ['deepwalk', 'node2vec']
    dimensions = [32, 64, 128]
    graph = load_blog_catalog()
    for method, dimension in product(methods, dimensions):
        generate_embedding(graph, method, dimension)
