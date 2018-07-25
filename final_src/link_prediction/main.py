from itertools import product
from pathlib import Path

import traceback

from link_prediction.Runner import run
from transformation.HistLossConfiguration import HistLossConfiguration
from transformation.RunConfiguration import RunConfiguration


def main():
    methods = []

    methods += ['deepwalk', 'hope']
    metrics = ['EMD']
    simmatrix_methods = ['ID']
    loss_methods = ['ASIM']
    calc_pos_methods = ['NORMAL']
    calc_neg_methods = ['IGNORE-NEG', 'NORMAL']
    calc_hist_methods = ['TF-KDE']

    for (metric,
         simmatrix_method,
         loss_method,
         calc_pos_method,
         calc_neg_method,
         calc_hist_method) in product(metrics,
                                      simmatrix_methods,
                                      loss_methods,
                                      calc_pos_methods,
                                      calc_neg_methods,
                                      calc_hist_methods):
        if (calc_neg_method == 'WEIGHTED') ^ (calc_pos_method == 'WEIGHTED'):
            continue
        methods += ['hist_loss_' +
                    str(HistLossConfiguration(metric,
                                              simmatrix_method,
                                              loss_method,
                                              calc_pos_method,
                                              calc_neg_method,
                                              calc_hist_method,
                                              ))]
    dimensions = [4, 8, 16, 32]
    # names = ['sbm-01-001', 'sbm-01-003', 'sbm-008-003', 'football', 'polbooks', 'facebook']
    names = ['blog_catalog']

    res = {}

    for (method, name, dim) in product(methods, names, dimensions):
        print(method, name, dim)
        try:
            a = run(RunConfiguration(method, name, dim), path_to_dumps=Path('./dumps').absolute())
            print("'" + method + ' ' + name + ' ' + str(dim) + "': " + str(a) + ',')
            res[method + ' ' + name + ' ' + str(dim)] = a
        except Exception:
            traceback.print_exc()
    print(res)


if __name__ == '__main__':
    main()
