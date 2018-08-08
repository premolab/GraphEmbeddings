import os
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
    calc_neg_methods = ['IGNORE-NEG']
    calc_hist_methods = ['TF-KDE']
    linearities = ['linear', 'nonlinear2', 'nonlinear2-reduce']

    for (metric,
         simmatrix_method,
         loss_method,
         calc_pos_method,
         calc_neg_method,
         calc_hist_method,
         linearity) in product(metrics,
                                      simmatrix_methods,
                                      loss_methods,
                                      calc_pos_methods,
                                      calc_neg_methods,
                                      calc_hist_methods,
                               linearities):
        if (calc_neg_method == 'WEIGHTED') ^ (calc_pos_method == 'WEIGHTED'):
            continue
        methods += ['hist_loss_' +
                    str(HistLossConfiguration(metric,
                                              simmatrix_method,
                                              loss_method,
                                              calc_pos_method,
                                              calc_neg_method,
                                              calc_hist_method,
                                              linearity
                                              ))]
    dimensions = [4, 8, 16]
    names = []
    # names = [
    #     'sbm-01-0001',
    #     'sbm-01-0005',
    #     'sbm-01-001',
    #     'sbm-01-002',
    #     # 'sbm-01-004',
    #     # 'sbm-01-005',
    #     # 'sbm-01-006',
    #     # 'sbm-01-007',
    # ]
    names += ['football', 'polbooks', 'facebook']

    path_to_dumps = Path(os.path.dirname(os.path.abspath(__file__))) / 'dumps'
    print("Path to dumps: {}".format(path_to_dumps))

    res = {}
    for (method, name, dim) in product(methods, names, dimensions):
        print(method, name, dim)
        try:
            a = run(
                RunConfiguration(method, name, dim),
                path_to_dumps=path_to_dumps,
            )
            print("'" + method + ' ' + name + ' ' + str(dim) + "': " + str(a) + ',')
            res[method + ' ' + name + ' ' + str(dim)] = a
        except Exception:
            traceback.print_exc()
    print(res)


if __name__ == '__main__':
    main()
