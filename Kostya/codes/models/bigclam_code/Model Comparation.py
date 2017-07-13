
# coding: utf-8

# # Эксперименты на реальных данных, сравнение моделей

# In[1]:

from pickle import dump, load
from time import gmtime, strftime

import igraph as ig

from .Experiments import *
from .Extends import *
from .big_clam_gamma import BigClamGamma


def time():
    return strftime("%H:%M:%S ", gmtime())
get_ipython().magic('matplotlib inline')


# In[2]:

from sklearn.decomposition import NMF


# In[3]:

def NMF_clust(A, K):
    model = NMF(n_components=K)
    res = model.fit_transform(A)
    #print res.shape
    return res


# In[4]:

import networkx as nx
from collections import defaultdict

def get_percolated_cliques(G, k):
    perc_graph = nx.Graph()
    cliques = [frozenset(c) for c in nx.find_cliques(G) if len(c) >= k]
    perc_graph.add_nodes_from(cliques)

    # First index which nodes are in which cliques
    membership_dict = defaultdict(list)
    for clique in cliques:
        for node in clique:
            membership_dict[node].append(clique)

    # For each clique, see which adjacent cliques percolate
    for clique in cliques:
        for adj_clique in get_adjacent_cliques(clique, membership_dict):
            if len(clique.intersection(adj_clique)) >= (k - 1):
                perc_graph.add_edge(clique, adj_clique)

    # Connected components of clique graph with perc edges
    # are the percolated cliques
    for component in nx.connected_components(perc_graph):
        yield(frozenset.union(*component))

def get_adjacent_cliques(clique, membership_dict):
    adjacent_cliques = set()
    for n in clique:
        for adj_clique in membership_dict[n]:
            if clique != adj_clique:
                adjacent_cliques.add(adj_clique)
    return adjacent_cliques


# # Генерация модельных примеров

# In[5]:

seed = 21773222
# data_params = {
#     'N': 1000,
#      'mut': 0.1,
#      'maxk': 50,
#      'k': 30,
#      'om': 2,
#      'muw': 0.1,
#      'beta': 2,
#      't1': 2,
#      't2': 2,
#      'on': 0,
#      }

data_params = {
    'N': 3000,
     'mut': 0.2,
     'maxk': 200,
     'k': 100,
     'om': 2,
     'muw': 0.2,
     'beta': 2,
     't1': 2,
     't2': 1,
     'on': 0,
     }

iter_count = 1
mixing_range = np.linspace(0, 0.5, 4)

model_params = {
    'initF':'cond_new_randz', 
    'LLH_output':False, 
    'iter_output':20000, 
    'processesNo':4, 
    'dump':False,
    'eps':1e-2,
    "max_iter":500000,
    #'sparsity_coef': 5,
}


# In[23]:

models = {#'BigClam-Zeros': lambda A, K, name: BigClam(1.0 * (A != 0), K, dump_name=name, **model_params).fit()[0],
          'BigClam-Zeros-simple': lambda A, K, name: BigClam(1.0 * (A != 0), K, dump_name=name, stepSizeMod='simple', **model_params).fit()[0],
          #'BigClam-Mean': lambda A, K, name: BigClam(1.0 * (A < np.mean(A)), K, dump_name=name, **model_params).fit()[0],
          'BigClamWeighted': lambda A, K, name: BigClam(A, K, dump_name=name,**model_params).fit()[0],
          'SparseGamma': lambda A, K, name: BigClamGamma(A, K, dump_name=name, **model_params).fit()[0],
          'BigClam-orig-zeros': lambda A, K, name: bigclam_orig(1.0 * (A != 0), K),
          #'BigClam-orig-mean': lambda A, K, name: bigclam_orig(1.0 * (A < np.mean(A)), K),
          'COPRA': lambda A, K, name: copra(A,K),
          'NMF': lambda A, K, name: NMF_clust(A,K),
          #'CPM': lambda A, K, name: [list(x) for x in get_percolated_cliques(nx.from_numpy_matrix(1.0 * (A != 0)), 5)]
        }

qual_fun = {'MixedModularity': MixedModularity,
            '1-MeanConductance': lambda F,A: 1-MeanConductance(GetComms(F, A), A) if not isinstance(F, list) else 1-MeanConductance(F, A),
            '1-MaxConductance': lambda F,A: 1-MaxConductance(GetComms(F, A), A) if not isinstance(F, list) else 1-MaxConductance(F, A),
            'NMI': lambda F,A, true_comm: NMI(GetComms(F, A), A, true_comm) if not isinstance(F, list) else NMI(F, A, true_comm),
            #'NMI_new': lambda F,A, true_comm: NMI3(GetComms(F, A), A, true_comm) if not isinstance(F, list) else NMI(F, A, true_comm),
            }


# In[24]:

models_res = []

for i_mix, mix in enumerate(mixing_range):
    print('{} mix: {}'.format(time(), mix))
    with file(r'..\external\Lancichinetti benchmark\time_seed.dat', 'w') as f:
        f.write(str(seed))
    data_params['on'] = np.floor(data_params['N'] * mix)
    one_graph_res = {name: {key: [] for key in qual_fun} for name in models}
    for iter in range(iter_count):
        print(' {}:'.format(iter), end=' ')
        G, comms = LancichinettiBenchmark(**data_params)
        A = np.array(nx.to_numpy_matrix(G))
        for name in models:
            print(name, end=' ') 
            F = models[name](A, len(comms), name)
            for key in qual_fun:
                if key not in  {"NMI", 'NMI_new'}:
                    res = qual_fun[key](F, A)
                else: 
                    res = qual_fun[key](F, A, comms)
                one_graph_res[name][key].append(res)
        if iter != iter_count-1:
            print('\r' + ' '*100 + '\r', end=' ')
        else:
            print() 
    models_res.append(one_graph_res)
    dump((models_res, mixing_range, mix, data_params), file('../data/dumps/models_res_temp-{}-dump'.format(i_mix), 'w'))
dump((models_res, mixing_range, mix, data_params), file('../data/dumps/models_res_full-dump', 'w'))


# In[17]:

(models_res, mixing_range, mix, data_params) = load(file('../data/dumps/models_res_full-dump'))


# In[25]:

def mean(l):
    return 1.0 * sum(l) / len(l)

plt.figure(figsize=(15,10))
for indx, qual_name in enumerate(qual_fun):
    plt.subplot(2,len(qual_fun)/2, indx+1)
    plt.ylabel('{}, N={}'.format(qual_name, data_params['N']))
    plt.xlabel('mixing parameter')
    colors = plt.get_cmap('hsv')(np.linspace(0, 1.0, len(models)+1))
    for i, name in enumerate(models):
        plt.plot(mixing_range, [res[name][qual_name][0] for res in models_res if len(res) != 0], label=name, color=colors[i])
    if indx == 1:
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


# In[43]:

models_res


# In[8]:

list(models_res[-2].keys())


# In[28]:

res[name][qual_name]


# In[29]:

res


# In[43]:

draw_matrix(A)


# # REAL

# In[47]:

s = 0
import os
DATA_PATH = '../data/weighted/'
gml_paths = [f for f in os.listdir(DATA_PATH) if f.endswith(".gml")]
Fss = []
LLHs = []

res = {'LLH':None, 'F': None, 'Model':None}
K = {'lesmis.gml': 8, 'netscience.gml': 300}
params = {
    'initF':'cond_new_randz_spr', 
    'LLH_output':True, 
    'iter_output':20000, 
    'processesNo':4, 
    'dump':10,
    'eps':1e-4,
    "max_iter":500000
}

models = {'BigClam-Zeros': lambda A, K, name: BigClam(1.0 * (A != 0), K, dump_name=name, **params),
          'BigClam-Zeros-s': lambda A, K, name: BigClam(1.0 * (A != 0), K, dump_name=name, stepSizeMod='simple', **params),
          #'BigClam-Mean': lambda A: BigClam(1.0 * (A < np.mean(A))),
          #'BigClamWeighted': lambda A: BigClam(A),
          'SparseGamma': lambda A, K, name: BigClamGamma(A, K, dump_name=name, **params),
          'SparseGamma-005': lambda A, K, name: BigClamGamma(A, K, dump_name=name, pow=0.05, **params),
          'SparseGamma-0005': lambda A, K, name: BigClamGamma(A, K, dump_name=name, pow=0.005, **params),
          'SparseGamma-1': lambda A, K, name: BigClamGamma(A, K, dump_name=name, pow=1, **params),
         }

models_res_all = []

for ego in gml_paths:
    G = nx.read_gml(file(DATA_PATH + '{}'.format(ego)))
    A = np.array(nx.to_numpy_matrix(G))
    models_res = {}
    print('{} data, {} nodes, {} edges'.format(ego, len(G), nx.number_of_edges(G)))
    for key in models:
        print("    " + key + ' model...')
        model = models[key](A, K[ego], key+ '_' + ego + '.dump')
        res = model.fit()
        models_res[key] = {'LLH':model.LLH_output, 'F': res[0], 'Model':model, "A":A}
    models_res_all.append(models_res)


# In[5]:

os.getcwd()


# In[44]:

def nx2ig(G):
    Gnew = ig.Graph()
    Gnew.add_vertices(G.nodes())
    Gnew.add_edges(G.edges())
    return Gnew

def NMI(F, A):
    C = F > np.sum(A) / (np.mean(A[A!=0]) * A.shape[0] * (A.shape[0] - 1))
    G = igraph.Weighted_Adjacency(A)


# In[11]:




# In[52]:

indx = 0

G = nx.read_gml(file(DATA_PATH + '{}'.format(gml_paths[indx])))
A = np.array(nx.to_numpy_matrix(G))

print(MixedModularity(models_res_all[indx]['SparseGamma']['F'], A))
print(MixedModularity(models_res_all[indx]['SparseGamma-1']['F'], A))
print(MixedModularity(models_res_all[indx]['SparseGamma-005']['F'], A))
print(MixedModularity(models_res_all[indx]['SparseGamma-0005']['F'], A))
print(MixedModularity(models_res_all[indx]['BigClam-Zeros']['F'], A))
print(MixedModularity(models_res_all[indx]['BigClam-Zeros-s']['F'], A))


# In[51]:

2 * np.sum(A) / (A.shape[0]*(A.shape[0]-1) )


# In[16]:

print(list(models_res_all[0].keys()))
plt.figure(figsize=(15,10))
plt.subplot(121)
draw_matrix(models_res_all[1]['SparseGamma']['F'])
plt.subplot(122)
draw_matrix(models_res_all[1]['BigClam-Zeros']['F'])


# In[ ]:




# In[125]:

F = np.array([[1.,1.,1.,1.,0.,0.,0.], [0.,0.,0.,1.,1.,1.,1.]]).T
At = test_example()
draw_matrix(At)
print(MixedModularity(F, At))


# # Модельные данные. Обзор

# In[35]:

draw_matrix(np.log(nx.to_numpy_matrix(G)))


# In[36]:

plt.hist([e[2]['weight'] for e in G.edges(data=True)], 100)
pass


# In[33]:

len(comms)


# In[6]:

seed = 21113222
data_params = {
    'N': 500,
     'mut': 0.05,
     'maxk': 50,
     'k': 20,
     'om': 1.5,
     'muw': 0.2,
     'beta': 1.5,
     't1': 1.5,
     't2': 1.5,
     #'minc': 100,
     'on': 0.1,
     }

# data_params = {
#     'N': 400,
#      'mut': 0.1,
#      'maxk': 50,
#      'k': 30,
#      'om': 2,
#      'muw': 0.1,
#      'beta': 2,
#      't1': 2,
#      't2': 2,
#      'minc': 20,
#      'on': 0.2,
#      }
G, comms = LancichinettiBenchmark(**data_params)


# In[31]:

G, comms = LancichinettiBenchmark(N=700, on=70, om=2, mut=0.04, muw=0.04, t1=2, t2=1, k= 60)


# In[32]:

pos=nx.spring_layout(G)
print('Pos!')


# In[34]:

size = 18
ax = plt.figure(figsize=(size, size))
#nx.draw_networkx(G_test, pos=pos, node_size=25, alpha=0.3, linewidths=0, width=0.5, with_labels=False)
max_w = max(e[2]['weight'] for e in G.edges(data=True))

node_size = 2 * size
drawn = []
print('1')
for col_i, i in enumerate(comms):
    print('.', end=' ')
    G_part = nx.subgraph(G, comms[i])
    width = [0.2 + 1.5 * e[2]['weight'] / max_w for e in G_part.edges(data=True)]
    edgelist = list(G_part.edges())
    nx.draw_networkx_edges(G, pos, edgelist=edgelist, width=width, alpha=0.05, edge_color=col[col_i if col_i < len(col) else -1])
    drawn.extend(edgelist)
print('2')
temp = list(set(G.edges())- set(drawn))
print('2.5')
nx.draw_networkx_edges(G, pos, edgelist=temp , width=width, alpha=0.01)
print('3')
nx.draw_networkx_nodes(G, pos, node_color='#FFFFFF', node_size=node_size, alpha=1, linewidths=0)
if len(comms) > len(col):
    print('WARNING: too low colors count')
print('4')
for j in G:
    #print '.',
    node_cols = [(col_i if col_i < len(col) else col_i%len(col)) for col_i, i in enumerate(comms) if j in comms[i]]
    for k, col_i in enumerate(node_cols):
        nx.draw_networkx_nodes(G, pos, nodelist=[j], node_color=col[col_i], node_size=(len(node_cols)-1 * k) * node_size,
                               alpha=1, linewidths=0, width=0.3, with_labels=False)

bord = 1
plt.xlim([-bord, bord])
plt.ylim([-bord, bord])
plt.axis('off')
plt.savefig('background.png', dpi=600, format='png')
pass



# In[11]:

plt.savefig('background.png', dpi=300, format='png')


# In[ ]:



