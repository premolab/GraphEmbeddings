from itertools import product

from src.load_data import load_blog_catalog
from src.transformers import transformers

from src.settings import PATH_TO_DUMPS


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
    methods = ['node2vec']
    dimensions = [32]
    graph = load_blog_catalog()
    for method, dimension in product(methods, dimensions):
        generate_embedding(graph, method, dimension)
