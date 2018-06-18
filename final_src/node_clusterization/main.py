from itertools import product

import traceback

from node_clusterization.Runner import run_sbm, run_football
from settings import PATH_TO_DUMPS
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
    calc_hist_methods = ['NORMAL']

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
                                              calc_hist_method))]

    dimensions = [4, 8, 16, 32]

    f = open('out.txt', 'w')
    res_dict = {}

    for (method, name, dim) in product(methods, ['sbm-01-001', 'sbm-01-003', 'sbm-008-003'], dimensions):
        print(method, name, dim)
        try:
            res = run_sbm(RunConfiguration(method, name, dim), path_to_dumps=PATH_TO_DUMPS)
            print("'" + method + ' ' + name + ' ' + str(dim) + "': " + str(res) + ',')
            f.write("'" + method + ' ' + name + ' ' + str(dim) + "': " + str(res) + ',')
            res_dict[method + ' ' + name + ' ' + str(dim)] = res
        except Exception:
            traceback.print_exc()

    print(res_dict)

    res_dict = {}
    for (method, name, dim) in product(methods, ['football'], dimensions):
        print(method, name, dim)
        try:
            res = run_football(RunConfiguration(method, name, dim), path_to_dumps=PATH_TO_DUMPS)
            print("'" + method + ' ' + name + ' ' + str(dim) + "': " + str(res) + ',')
            f.write("'" + method + ' ' + name + ' ' + str(dim) + "': " + str(res) + ',\n')
            res_dict[method + ' ' + name + ' ' + str(dim)] = res
        except Exception:
            traceback.print_exc()
    print(res_dict)


if __name__ == '__main__':
    main()
