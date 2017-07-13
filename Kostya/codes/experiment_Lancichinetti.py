import numpy as np
from codes.Experiment import run_experiment
import pickle
from codes.load_data import LoadLancichinettiBenchmark
from time import gmtime, strftime
from multiprocessing import Pool, freeze_support, set_start_method


def time():
    return strftime("%H:%M:%S ", gmtime())


def worker(args):
    (q, n, model, A, true_comm, d) = args
    return {'model': model,
            'dim': d,
            'mix': q,
            'result': run_experiment(A, true_comm, d, 'Lancichinetti_benchmark-{}-{}'.format(q, n), model,
                                     load_dumped_res=False)}


def run():
    freeze_support()
    data_params = {
        'N': 1000,
        'mut': 0.1,
        'maxk': 150,
        'k': 75,
        'om': 2,
        'muw': 0.1,
        'beta': 2,
        't1': 2,
        't2': 2,
        'on': 0,
    }

    data_params['N'] = 1000

    Q = [0, 0.1, 0.3, 0.5]
    d = 10

    models = ['stoc_hist', 'hist', 'node2vec', 'gamma', 'svd', 'nmf', 'bigclam']

    result = []
    seed = 11646
    #with Pool(processes=4) as pool:
    for i_mix, q in enumerate(Q):
        print('\n{} mix: {}'.format(time(), q))
        with open(r'..\external\Lancichinetti_benchmark\time_seed.dat', 'w') as f:
            f.write(str(seed))
        data_params['on'] = np.floor(data_params['N'] * q)
        A, true_comm, name = LoadLancichinettiBenchmark(**data_params)
        result.append([])

        argss = [(q, data_params["N"], model, A, true_comm, d) for model in models]

        #result.append(pool.map(worker, argss))
        result.append([])
        for args in argss:
            res = worker(args)
            result[-1].append(res)
        pickle.dump(result, open('./dumps/result_l-partition-temp_part', 'wb'))

    result = np.array(result)
    pickle.dump(result, open('./dumps/result_l-partition-temp', 'wb'))


if __name__ == '__main__':
    run()

# sns.set(style="white")
# plt.subplot(121)
# result = pickle.load(open('./dumps/result_l-partition-first', 'rb'))
# legend = []
# for test in [0, 1]:
#     for model in range(result.shape[1]):
#         plt.plot(q, result[:, model, test], 'rgb'[model] + '-' if test else '--')
#         legend.append('{}, {}'.format(models[model], "test" if test else 'train'))
# plt.legend(legend, loc=0)
# plt.xlabel('Вероятность появления ребра вне кластера')
# plt.ylabel('Точность')
#
# plt.subplot(122)
# result = pickle.load(open('./dumps/result_l-partition-temp', 'rb'))
# legend = []
# for test in [0, 1]:
#     for model in range(result.shape[1]):
#         plt.plot(q, result[:, model, test], 'rgb'[model] + '-' if test else '--')
#         legend.append('{}, {}'.format(models[model], "test" if test else 'train'))
# plt.legend(legend, loc=0)
# plt.xlabel('Вероятность появления ребра вне кластера')
# plt.ylabel('Точность')
# plt.tight_layout()
# plt.show()
# pass
