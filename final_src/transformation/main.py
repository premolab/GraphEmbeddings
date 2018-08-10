import os
from itertools import product
from pathlib import Path

from transformation.RunConfiguration import RunConfiguration
from transformation.Runner import run


def main():
    methods = ['sdne']

    metrics = ['EMD']
    simmatrix_methods = ['ID']
    loss_methods = ['ASIM']
    calc_pos_methods = ['NORMAL']
    calc_neg_methods = ['IGNORE-NEG']
    calc_hist_methods = ['TF-KDE']
    embedding_preprocessing_methods = ['sigmoid', 'normalization']

    for (metric,
         simmatrix_method,
         loss_method,
         calc_pos_method,
         calc_neg_method,
         calc_hist_method,
         embedding_preprocessing_method) in product(metrics,
                                                    simmatrix_methods,
                                                    loss_methods,
                                                    calc_pos_methods,
                                                    calc_neg_methods,
                                                    calc_hist_methods,
                                                    embedding_preprocessing_methods
                                                    ):
        methods += ['hist_loss_' +
                    str(HistLossConfiguration(metric,
                                              simmatrix_method,
                                              loss_method,
                                              calc_pos_method,
                                              calc_neg_method,
                                              calc_hist_method,
                                              embedding_preprocessing_method
                                              ))]

    dimensions = [4]
    names = ['sbm-01-001']

    path_to_dumps = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / 'dumps'
    print("Path to dumps: {}".format(path_to_dumps))

    for (method, name, dim) in product(methods, names, dimensions):
        print(method, name, dim)
        try:
            run(RunConfiguration(method, name, dim), path_to_dumps=path_to_dumps.absolute())
        except Exception as e:
            print(e)


if __name__ == '__main__':
    main()
