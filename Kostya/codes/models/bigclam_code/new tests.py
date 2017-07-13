
# coding: utf-8

# # План
# 
# * ✔ Разобраться с итоговыми графиками, быть уверенным что там все правильно. (19-21)
# 
# 
# 1. ~~Проверить рассматриваемые функционалы на простых графах, где ее можно посчитать, проверить что смешанная модулярность равна обычной, если группы вершин не будут пересекаться.~~
# * ~~Проверить, что проводимость учитывает взвешенность.~~
# * ~~Проанализировать NMF --- почему выдает высокое качество, там где оно высокое и т.д. Закрыть вопросы по нему (Посмотреть на критерий выделения сообществ из матрицы и истинное количество сообществ, которое выдает метод (часть пустая --- низкая проводимость?).~~
# * ✔ Добаввить значения истинного разбиения на графики для большей наглядности.
# * ✔ Добавить CFinder для сравнения с ним.
# * ✔ Добавить State of the art метод для выделения непересекающихся сообществ
# * Провести эксперименты для больших графов.
# 
# 
# * ✔ Провести полноценные эксперименты с инициализацией на бенчмарке. (22)
# * Оформить все в текст, т.е. переделать и дополнить текст курсовой в статью. (22-25)

# In[4]:

from pickle import dump, load
from shutil import copyfile
from time import gmtime, strftime

from .Experiments import *
from .Extends import *
from .big_clam_gamma import BigClamGamma


def time():
    return strftime("%H:%M:%S ", gmtime())


# In[5]:

from sklearn.decomposition import NMF
def NMF_clust(A, K):
    model = NMF(n_components=K)
    res = model.fit_transform(A)
    #print res.shape
    return res


# In[6]:

def draw_graph(G, comms, size=12, bord=1.5):
    ax = plt.figure(figsize=(size, size))
    #nx.draw_networkx(G_test, pos=pos, node_size=25, alpha=0.3, linewidths=0, width=0.5, with_labels=False)
    max_w = max(e[2]['weight'] for e in G.edges(data=True))

    node_size = 2 * size
    drawn = []
    print('1')
    for col_i, i in enumerate(comms):
        print('.', end=' ')
        G_part = nx.subgraph(G, comms[i])
        width = [2 * e[2]['weight'] / max_w for e in G_part.edges(data=True)]
        edgelist = list(G_part.edges())
        nx.draw_networkx_edges(G, pos, edgelist=edgelist, width=width, alpha=0.15, edge_color=col[col_i if col_i < len(col) else -1])
        drawn.extend(edgelist)
    print('2')
    temp = list(set(G.edges())- set(drawn))
    print('2.5')
    nx.draw_networkx_edges(G, pos, edgelist=temp , width=width, alpha=0.15)
    print('3')
    nx.draw_networkx_nodes(G, pos, node_color='#FFFFFF', node_size=node_size, alpha=1, linewidths=0)
    if len(comms) > len(col):
        print('WARNING: too low colors count')
    print('4')
    for j in G:
        #print '.',
        node_cols = [(col_i if col_i < len(col) else col_i%len(col)) for col_i, i in enumerate(comms) if j in comms[i]]
        for k, col_i in enumerate(node_cols):
            nx.draw_networkx_nodes(G, pos, nodelist=[j], node_color=col[col_i], node_size=(len(node_cols)-k) * node_size,
                                   alpha=1, linewidths=0, width=0.3, with_labels=False)
            #nx.draw_networkx_labels(G,pos)
    
    plt.xlim([-bord, bord])
    plt.ylim([-bord, bord])
    plt.axis('off')
    pass


# In[7]:

save_all=False


# In[8]:

def enshure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

lanc_bech_files = ['outputlog', 'parameters.dat', 'community.dat', 'statistics.dat', 'time_seed.dat', 'network.dat']
nmi_files = ['outputlog-nmi', 'clu1', 'clu2']
model_files = {'COPRA': ['../external/COPRA/COPRA_output.log', '../external/COPRA/test.COPRA', 'clusters-test.COPRA'],
               'BigClam-orig-zeros': ['../external/BigClam/bigClam_output.log', '../external/BigClam/test.bigClam', '../external/BigClam/cmtyvv.txt' ]}


# In[9]:

def worker(iter, mix):
        res = {name: {key: 0 for key in qual_fun} for name in models}
        print(' {}:'.format(iter), end=' ')
        G, comms = LancichinettiBenchmark(**data_params)
        if save_all:
            cur_save_dir = "../data/dumps/all_exp/{:.3f}/".format(mix)
            enshure_dir(cur_save_dir)
            cur_save_dir = cur_save_dir + str(iter) + '/'
            enshure_dir(cur_save_dir)
            for filename in lanc_bech_files:
                copyfile('../external/Lancichinetti benchmark/' + filename, cur_save_dir + 'testgraph-' + filename)
        A = np.zeros(shape=(len(G), len(G)))
        for u, v, data in G.edges(data=True):
            A[int(u)-1][int(v)-1] = data['weight']
        #Fs = pool.map(worker_models, zip([A]*len(models), [comms]*len(models), models.keys()))
        Fs = [worker_models((A, comms, name)) for name in models]

        Fs = dict(Fs)
        for name in Fs:
            for key in qual_fun:
                try:
                    if key not in {"1-NMI", "NMI", 'NMI_new'}:
                        q = qual_fun[key](Fs[name], A)
                    else:
                        q = qual_fun[key](Fs[name], A, comms)
                        if save_all:
                            for filename in nmi_files:
                                copyfile('../external/Lancichinetti benchmark/' + filename, cur_save_dir + name + '-' + filename)
                except:
                    print('Some err in ' + name, end=' ')
                    q = float('nan')
                res[name][key] = q
            if save_all:
                with open(cur_save_dir + name + '-' 'quals', 'w') as f:
                    for key in res[name]:
                        f.write('{}: {}\n'.format(key, res[name][key]))
        return res


# In[10]:

def worker_models(args):
    A, comms, name = args
    print(name, end=' ')
    if name != 'groundtruth':
        F = models[name](A, len(comms), name)
    else:
        F = models[name](comms)
    print('.', end=' ')
    return name, F


# In[11]:

def mean(l):
    #lt = [x for x in l if 1 > x ]
    lt = l
    return 1.0 * sum(lt) / len(lt) if len(lt) > 0 else float('nan')


# In[12]:

model_params = {
    'initF': 'cond_randz',
    'LLH_output': False,
    'iter_output': 20000,
    'processesNo': 1,
    'dump': False,
    'eps': 1e-3,
    "max_iter": 500000,
}


# In[13]:

models = {#'BigClam-Zeros': lambda A, K, name: BigClam(1.0 * (A != 0), K, dump_name=name, **model_params).fit()[0],
          #'BigClam-Zeros-simple': lambda A, K, name: BigClam(1.0 * (A != 0), K, dump_name=name, stepSizeMod='simple', **model_params).fit()[0],
          #'BigClam-Mean': lambda A, K, name: BigClam(1.0 * (A < np.mean(A)), K, dump_name=name, **model_params).fit()[0],
          'BigClamWeighted': lambda A, K, name: BigClam(A, K, dump_name=name, **model_params).fit()[0],
          #'BigClamWeighted-simple': lambda A, K, name: BigClam(A, K, dump_name=name, stepSizeMod='simple', **model_params).fit()[0],
          #'SparseGamma-p1': lambda A, K, name: BigClamGamma(A, K, dump_name=name, pow=1, **model_params).fit()[0],
          #'SparseGamma-p0.05': lambda A, K, name: BigClamGamma(A, K, dump_name=name, pow=0.05, **model_params).fit()[0],
          'SparseGamma': lambda A, K, name: BigClamGamma(A, K, dump_name=name, **model_params).fit()[0],
          'BigClam-orig-zeros': lambda A, K, name: bigclam_orig(1.0 * (A != 0), K),
          #'BigClamWeighted-sp10': lambda A, K, name: BigClam(A, K, dump_name=name, sparsity_coef=10,  **model_params).fit()[0],
          #'SparseGamma-sp10': lambda A, K, name: BigClamGamma(A, K, dump_name=name, sparsity_coef=10, **model_params).fit()[0],
          #'BigClam-orig-mean': lambda A, K, name: bigclam_orig(1.0 * (A < np.mean(A)), K),
          'COPRA': lambda A, K, name: copra(A, K),
          'NMF': lambda A, K, name: NMF_clust(A, K),
          'groundtruth': lambda res: [list(map(int, res[key])) for key in res],
          #'CFinder': lambda A, K, name: CFinder(A, K),
          #'CPM': lambda A, K, name: [list(x) for x in get_percolated_cliques(nx.from_numpy_matrix(1.0 * (A != 0)), 5)]
          'walktrap': lambda A, K, name: walktrap(A, K),
        }


# In[14]:

from collections import OrderedDict

qual_fun = OrderedDict([
            ('1-MeanConductance', lambda F, A: 1-MeanConductance(GetComms(F, A), A) if not isinstance(F, list) else 1-MeanConductance(F, A)),
            ('1-MaxConductance', lambda F, A: 1-MaxConductance(GetComms(F, A), A) if not isinstance(F, list) else 1-MaxConductance(F, A)),
            ('NMI', lambda F,A, true_comm: NMI(GetComms(F, A), A, true_comm) if not isinstance(F, list) else NMI(F, A, true_comm)),
            #('NMI_new', lambda F,A, true_comm: NMI3(GetComms(F, A), A, true_comm) if not isinstance(F, list) else NMI(F, A, true_comm)),
            ('MixedModularity', MixedModularity),
            #('NMI_skl', lambda F,A, true_comm: normalized_mutual_info_score(GetComms(F, A)))
    ])


# In[15]:

def calc_res(data_params, save_path='./dumps/models_res_full-dump'):
    iter_count = 1
    if save_all:
        enshure_dir("../data/dumps/all_exp")
    mixing_range = np.linspace(0, 0.5, 6)
    #mixing_range = np.linspace(0, 0.5, 3)
    models_res = []
    for i_mix, mix in enumerate(mixing_range):
        print('{} mix: {}'.format(time(), mix))
        with open(r'..\external\Lancichinetti benchmark\time_seed.dat', 'w') as f:
            f.write(str(seed))
        data_params['on'] = np.floor(data_params['N'] * mix)
        one_graph_res = {name: OrderedDict([(key, []) for key in qual_fun]) for name in models}
        res = []
        for iter in range(iter_count):
            res.append(worker(iter, mix))
        #res = pool.map(worker, xrange(iter_count))
        for iter in range(iter_count):
            for name in one_graph_res:
                for key in one_graph_res[name]:
                    one_graph_res[name][key].append(res[iter][name][key])

        models_res.append(one_graph_res)
        dump((models_res, mixing_range, mix, data_params),
             open(save_path + '-part-{}'.format(i_mix), 'w'))

    dump((models_res, mixing_range, mix, data_params), open(save_path, 'w'))


# In[17]:

def draw_res(res_path='../data/dumps/models_res_full-dump'):
    (models_res, mixing_range, mix, data_params) = load(open(res_path))

    models_ = list(models_res[0].keys())
    qual_fun_ = list(models_res[0][models_[0]].keys())
    print(models_, qual_fun_)
    plt.figure(figsize=(15, 15))
    #plt.figure(figsize=(14, 40))
    for indx, qual_name in enumerate(qual_fun_):
        plt.subplot(2, len(qual_fun_) / 2, indx + 1)
        #plt.subplot(len(qual_fun_), 2, indx + 1)
        plt.ylabel('{}, N={}'.format(qual_name, data_params['N']))
        plt.xlabel('mixing parameter')
        colors = plt.get_cmap('hsv')(np.linspace(0, 1.0, len(models_) + 1))
        for i, name in enumerate(models_):
            #if name in ['SparseGamma-p1', 'BigClamWeighted-simple', 'BigClam-Zeros-simple', 'BigClam-Zeros', 'SparseGamma-p0.05']:
            #    continue
            plt.plot(mixing_range, [mean(res[name][qual_name]) for res in models_res if len(res) != 0], label=name,
                     color=colors[i] if name != 'groundtruth' else 'k', marker=('.', 'o', 'x', '+')[i%4])
        if indx == 1:
            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        #plt.ylim([-0.05, 1.05])
    plt.show()
    plt.savefig('../plots/{}_{}.eps'.format(data_params['N'], data_params['mut']))


# In[18]:

seed = 21113222
data_params = {
    'N': None,
     'mut': None,
     'maxk': 50,
     'k': 30,
     'om': 2,
     'muw': None,
     'beta': 2,
     't1': 2,
     't2': 2,
     'on': 0,
     }


# In[23]:

data_params['N'] = 1000
data_params['mut'] = 0.1
data_params['muw'] = 0.1
print('~~~~~~~~~~~~~~ {} {} {} ~~~~~~~~~~~~~~'.format(data_params['N'], data_params['mut'], data_params['muw']))
calc_res(data_params, '../data/dumps/models_res_full-dump-{}-{}'.format(data_params['N'], data_params['mut']))

data_params['mut'] = 0.3
data_params['muw'] = 0.3
print('~~~~~~~~~~~~~~ {} {} {} ~~~~~~~~~~~~~~'.format(data_params['N'], data_params['mut'], data_params['muw']))
calc_res(data_params, '../data/dumps/models_res_full-dump-{}-{}'.format(data_params['N'], data_params['mut']))

data_params['N'] = 5000
data_params['mut'] = 0.1
data_params['muw'] = 0.1
print('~~~~~~~~~~~~~~ {} {} {} ~~~~~~~~~~~~~~'.format(data_params['N'], data_params['mut'], data_params['muw']))
calc_res(data_params, '../data/dumps/models_res_full-dump-{}-{}'.format(data_params['N'], data_params['mut']))

data_params['mut'] = 0.3
data_params['muw'] = 0.3
print('~~~~~~~~~~~~~~ {} {} {} ~~~~~~~~~~~~~~'.format(data_params['N'], data_params['mut'], data_params['muw']))
calc_res(data_params, '../data/dumps/models_res_full-dump-{}-{}'.format(data_params['N'], data_params['mut']))


# In[ ]:




# In[ ]:

print()
print()
for mix, res in zip(mixing_range, models_res):
    for key in res:
        print(mix, ': ', key, res[key])


# In[ ]:




# In[ ]:



