# coding: utf-8


from .Experiments import *
from algorithms.big_clam import *

def mean(lists):
    length = max(len(l) for l in lists)
    N = len(lists)
    res = np.zeros(shape=(length,))
    for i in range(length):
        temp = [l[i] for l in lists if len(l) > i]
        res[i] = 1.0 * sum(temp) / len(temp) #if len(temp) > 0 else float('nan')
    return res

if __name__ == "__main__":
    # power = 0.17
    # F_true = 0.4 * Fs3[0]
    # print F_true.shape
    # A =  gamma_model_test_data(F_true)
    #
    # P = 1 - 0.8 * np.exp(- F_true.T.dot(F_true))
    # np.random.rand(*A.shape)
    # bord = np.random.rand(*A.shape)
    #
    # mask = P <= (bord + bord.T) / 2
    #
    # B = A.copy()
    # B[mask] = 0
    # C = B.copy()
    # C[B != 0] = 1
    #
    # K = 4
    # import os
    #
    DATA_PATH = '../data/vk/egonets/'
    ego_paths = [f for f in os.listdir(DATA_PATH) if f.endswith(".egonet")]
    # # ego_paths = ego_paths[:2]
    # inits = ['cond', 'cond_new', 'rand', 'cond_randz', 'cond_new_randz', 'cond_randz_spr', "cond_new_randz_spr"]
    # Fss = []
    # initFss = []
    # itersLLHs = []
    # for indx, ego in progress(list(enumerate(ego_paths))):
    #     try:
    #         D = cPickle.load(file(os.path.join(DATA_PATH, ego)))
    #     except:
    #         D = {}
    #         for line in file(os.path.join(DATA_PATH, ego)):
    #             key, data = line.split(':')
    #             D[int(key)] = map(int, data.split())
    #     G = nx.Graph(D)
    #     A = np.array(nx.to_numpy_matrix(G))
    #     Fs = []
    #     initFs = []
    #     itersLLH = []
    #     for init in inits:
    #         bigClam = BigClam(A, K, initF=init, debug_output=False, LLH_output=False, eps=0, iter_output=40, processesNo=6, max_iter=4000)
    #         res = bigClam.fit()
    #         initFs.append(bigClam.initFmode)
    #         itersLLH.append(bigClam.LLH_output_vals)
    #         Fs.append(res[0])
    #     itersLLHs.append(itersLLH)
    #     initFss.append(initFs)
    #     Fss.append(Fs)
    #
    # cPickle.dump((itersLLHs, inits, initFss, Fss), file('../data/dumps/init_ego_dump', 'w'))

    (itersLLHs, inits, initFss, Fss) = cPickle.load(file('../data/dumps/init_ego_dump', 'r'))

    iter_output=40
    itersLLH = []
    for i, init in enumerate(inits):
        itersLLH.append(mean([t[i] for t in itersLLHs]))
    plt.figure(figsize=(14, 5))
    for llh, init in zip(itersLLH, inits):
        llh = llh[:100]
        X = list(range(0, iter_output * len(llh), iter_output))
        plt.subplot(121)
        plt.plot(X, -np.log(-np.array(llh)), label=init)
        plt.subplot(122)
        plt.plot(X[80:], -np.log(-np.array(llh[80:])), label=init)
    plt.subplot(121)
    plt.legend(loc=4)
    plt.show()
    pass