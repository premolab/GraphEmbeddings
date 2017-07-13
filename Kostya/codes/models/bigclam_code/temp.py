# coding: utf-8
# # Эксперименты на реальных данных, сравнение моделей

# In[1]:

from pickle import load
from time import gmtime, strftime

from sklearn.decomposition import NMF

from .Experiments import *
from .Extends import *

def time():
    return strftime("%H:%M:%S ", gmtime())

# In[2]:

def NMF_clust(A, K):
    model = NMF(n_components=K)
    res = model.fit_transform(A)
    #print res.shape
    return res


model_params = {
    'initF': 'cond_new_randz',
    'LLH_output': False,
    'iter_output': 20000,
    'processesNo': 1,
    'dump': 1000,
    'eps': 1e-2,
    "max_iter": 500000,
}

# In[47]:

models = {#'BigClam-Zeros': lambda A, K, name: BigClam(1.0 * (A != 0), K, dump_name=name, **model_params).fit()[0],
          #'BigClam-Zeros-simple': lambda A, K, name: BigClam(1.0 * (A != 0), K, dump_name=name, stepSizeMod='simple', **model_params).fit()[0],
          #'BigClam-Mean': lambda A, K, name: BigClam(1.0 * (A < np.mean(A)), K, dump_name=name, **model_params).fit()[0],
          'BigClamWeighted': lambda A, K, name: BigClam(A, K, dump_name=name, **model_params).fit()[0],
          #'SparseGamma': lambda A, K, name: BigClamGamma(A, K, dump_name=name, **model_params).fit()[0],
          'BigClam-orig-zeros': lambda A, K, name: bigclam_orig(1.0 * (A != 0), K),
          #'BigClamWeighted-sp10': lambda A, K, name: BigClam(A, K, dump_name=name, sparsity_coef=10,  **model_params).fit()[0],
          #'SparseGamma-sp10': lambda A, K, name: BigClamGamma(A, K, dump_name=name, sparsity_coef=10, **model_params).fit()[0],
          #'BigClam-orig-mean': lambda A, K, name: bigclam_orig(1.0 * (A < np.mean(A)), K),
          'COPRA': lambda A, K, name: copra(A,K),
          #'NMF': lambda A, K, name: NMF_clust(A,K),
          #'CPM': lambda A, K, name: [list(x) for x in get_percolated_cliques(nx.from_numpy_matrix(1.0 * (A != 0)), 5)]
        }

qual_fun = {'MixedModularity': MixedModularity,
            '1-MeanConductance': lambda F,A: 1-MeanConductance(GetComms(F, A), A) if not isinstance(F, list) else 1-MeanConductance(F, A),
            '1-MaxConductance': lambda F,A: 1-MaxConductance(GetComms(F, A), A) if not isinstance(F, list) else 1-MaxConductance(F, A),
            '1-NMI': lambda F,A, true_comm: 1-NMI(GetComms(F, A), A, true_comm) if not isinstance(F, list) else NMI(F, A, true_comm),
            #'NMI_new': lambda F,A, true_comm: NMI3(GetComms(F, A), A, true_comm) if not isinstance(F, list) else NMI(F, A, true_comm),
            }


(models_res, mixing_range, mix, data_params) = load(file('../data/dumps/models_res_full-dump'))


    # In[49]:

def mean(l):
    lt = [x for x in l if not( x > -1e-6 and x < 0)]
    return 1.0 * sum(lt) / len(lt) if len(lt) > 0 else float('nan')

plt.figure(figsize=(15, 10))
for indx, qual_name in enumerate(qual_fun):
    plt.subplot(2, len(qual_fun) / 2, indx + 1)
    plt.ylabel('{}, N={}'.format(qual_name, data_params['N']))
    plt.xlabel('mixing parameter')
    colors = plt.get_cmap('hsv')(np.linspace(0, 1.0, len(models) + 1))
    for i, name in enumerate(models):
        if 'sp10' in name:
            continue
        plt.plot(mixing_range, [mean(res[name][qual_name]) for res in models_res if len(res) != 0], label=name,
                 color=colors[i])
    if indx == 1:
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
print()
print()
for mix, res in zip(mixing_range, models_res):
    for key in res:
        print(mix, ': ', key, res[key])
plt.show()