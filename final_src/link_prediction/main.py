from itertools import product
from pathlib import Path

import traceback

from link_prediction.Runner import run
from transformation.HistLossConfiguration import HistLossConfiguration
from transformation.RunConfiguration import RunConfiguration

if __name__ == '__main__':
    methods = []

    metrics = ['EMD']
    simmatrix_methods = ['ID', 'ADA']
    loss_methods = ['ASIM']
    calc_pos_methods = ['NORMAL', 'WEIGHTED']
    calc_neg_methods = ['NORMAL', 'WEIGHTED', 'IGNORE-NEG']
    calc_hist_methods = ['NORMAL']
    batch_sizes = [400]

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
        if (calc_neg_method == 'WEIGHTED') ^ (calc_pos_method == 'WEIGHTED'):
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
    names = ['football', 'polbooks', 'facebook']

    for (method, name, dim) in product(methods, names, dimensions):
        print(method, name, dim)
        try:
            print(run(RunConfiguration(method, name, dim), path_to_dumps=Path('./dumps').absolute()))
        except Exception:
            traceback.print_exc()

