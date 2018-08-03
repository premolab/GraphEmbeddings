from itertools import product

import traceback
import numpy as np

from node_classification.Runner import run_blog_catalog, run_sbm, run_cliques
from settings import PATH_TO_DUMPS
from transformation.HistLossConfiguration import HistLossConfiguration
from transformation.RunConfiguration import RunConfiguration


def main():
    methods = ['deepwalk', 'hope']
    methods += ['deepwalk']
    metrics = ['EMD']
    simmatrix_methods = ['ID']
    loss_methods = ['ASIM']
    calc_pos_methods = ['NORMAL']
    calc_neg_methods = ['IGNORE-NEG']
    calc_hist_methods = ['NORMAL']

    for (metric,
         simmatrix_method,
         loss_method,
         calc_pos_method,
         calc_neg_method,
         calc_hist_method,
         ) in product(metrics,
                                simmatrix_methods,
                                loss_methods,
                                calc_pos_methods,
                                calc_neg_methods,
                                calc_hist_methods,
                                ):
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

    f = open('out_clas.txt', 'w')
    res_dict = {}

    for (method, name, dim) in product(methods, ['cliques'], dimensions):
        print(method, name, dim)
        try:
            res = run_cliques(RunConfiguration(method, name, dim), path_to_dumps=PATH_TO_DUMPS)
            x = np.mean(res)
            print("'" + method + ' ' + name + ' ' + str(dim) + "': " + str(x) + ',')
            f.write("'" + method + ' ' + name + ' ' + str(dim) + "': " + str(x) + ',\n')
            res_dict[method + ' ' + name + ' ' + str(dim)] = x
        except Exception:
            traceback.print_exc()

    for (method, name, dim) in product(methods, [], dimensions):
        print(method, name, dim)
        try:
            res = run_sbm(RunConfiguration(method, name, dim), path_to_dumps=PATH_TO_DUMPS)
            x = np.mean(res)
            print("'" + method + ' ' + name + ' ' + str(dim) + "': " + str(x) + ',')
            f.write("'" + method + ' ' + name + ' ' + str(dim) + "': " + str(x) + ',\n')
            res_dict[method + ' ' + name + ' ' + str(dim)] = x
        except Exception:
            traceback.print_exc()

    for (method, name, dim) in product(methods, [], dimensions):
        print(method, name, dim)
        try:
            res = run_blog_catalog(RunConfiguration(method, name, dim), path_to_dumps=PATH_TO_DUMPS)
            x = np.mean(res)
            print("'" + method + ' ' + name + ' ' + str(dim) + "': " + str(x) + ',')
            f.write("'" + method + ' ' + name + ' ' + str(dim) + "': " + str(x) + ',\n')
            res_dict[method + ' ' + name + ' ' + str(dim)] = x
        except Exception:
            traceback.print_exc()

    print(res_dict)


if __name__ == '__main__':
    main()
