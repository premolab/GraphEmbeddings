from itertools import product

import traceback

from node_classification.Runner import run_blog_catalog
from settings import PATH_TO_DUMPS
from transformation.HistLossConfiguration import HistLossConfiguration
from transformation.RunConfiguration import RunConfiguration

if __name__ == '__main__':
    methods = []

    metrics = ['EMD']
    simmatrix_methods = ['ID', 'ADA']
    loss_methods = ['ASIM']
    calc_pos_methods = ['NORMAL', 'WEIGHTED']
    calc_neg_methods = ['NORMAL', 'WEIGHTED', 'IGNORE_NEG']
    calc_hist_methods = ['NORMAL']
    batch_sizes = [800]

    for (metric,
         simmatrix_method,
         loss_method,
         calc_pos_method,
         calc_neg_method,
         calc_hist_method,
         batch_size) in product(metrics,
                                simmatrix_methods,
                                loss_methods,
                                calc_pos_methods,
                                calc_neg_methods,
                                calc_hist_methods,
                                batch_sizes):
        if calc_neg_method != calc_pos_method:
            continue
        methods += ['hist_loss_' +
                    str(HistLossConfiguration(metric,
                                              simmatrix_method,
                                              loss_method,
                                              calc_pos_method,
                                              calc_neg_method,
                                              calc_hist_method,
                                              batch_size))]
    methods += ['deepwalk']
    dimensions = [4, 8, 16, 32]
    names = ['blog_catalog']

    for (method, name, dim) in product(methods, names, dimensions):
        print(method, name, dim)
        try:
            print(run_blog_catalog(RunConfiguration(method, name, dim), path_to_dumps=PATH_TO_DUMPS))
        except Exception:
            traceback.print_exc()

