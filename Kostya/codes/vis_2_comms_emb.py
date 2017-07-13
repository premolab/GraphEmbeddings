import numpy as np
from codes.Experiment import run_experiment
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
import pickle
from codes.load_data import generate_graphs
from gensim.models import KeyedVectors

sns.set(style="white")

n = 200
frac = 0.5
p = 0.2
# q = [0.01, 0.02, 0.04, 0.08, 0.1, 0.15, 0.2]
Q = [0.01]  # , 0.05, 0.1, 0.2, 0.25]
d = 2

models = ['opt', 'hist', 'node2vec']

result = []
for A, true_comm, q in generate_graphs(n, frac, p, Q):
    result.append([])
    for model in models:
        result[-1].append(run_experiment(A, true_comm, d, 'l-partition-{}-{}'.format(q, p), model, load_dumped_res=False).mean(axis=2)[:, 2])

result = np.array(result)

pickle.dump(result, open('./dumps/result_l-partition-temp', 'wb'))

# plt.subplot(121)

result_hist = pickle.load(open('./dumps/results/result_l-partition-0.01-0.2_hist_d2.dump', 'rb'))
result_node2vec = pickle.load(open('./dumps/results/result_l-partition-0.01-0.2_node2vec_d2.dump', 'rb'))
result_opt = pickle.load(open('./dumps/results/result_l-partition-0.01-0.2_opt_d2.dump', 'rb'))
pass

emb_hist = pickle.load(open('./dumps/models/hist2v_l-partition-0.01-0.2_d2.dump', 'rb'))[1]
emb_node2vec = KeyedVectors.load_word2vec_format('./dumps/models/n2v_l-partition-0.01-0.2_p0.25_q0.25_d2.dump')
emb_node2vec = emb_node2vec[[str(x) for x in range(n)]]
emb_opt = pickle.load(open('./dumps/models/opt2v_l-partition-0.01-0.2_d2.dump', 'rb'))[1]
pass
node_size = 50
node_color = ['#56A0D3']*100 + ['r']*100
edge_color = '#222222'
alpha=0.8
width=0.1
for A, true_comm, q in generate_graphs(n, frac, p, Q):
    plt.figure()
    plt.subplot(131)
    nx.draw(A, pos=emb_hist, node_size=node_size, node_color=node_color, edge_color=edge_color, alpha=alpha, width=width)
    plt.title("{}, {}, {}".format('hist', result_hist['best_params'], result_hist['results'][1,2, :].mean()))
    plt.subplot(132)
    nx.draw(A, pos=emb_node2vec, node_size=node_size, node_color=node_color, edge_color=edge_color, alpha=alpha, width=width)
    plt.title("{}, {}, {}".format('node2vec', result_node2vec['best_params'], result_node2vec['results'][1, 2, :].mean()))
    plt.subplot(133)
    nx.draw(A, pos=emb_opt, node_size=node_size, node_color=node_color, edge_color=edge_color, alpha=alpha, width=width)
    plt.title("{}, {}, {}".format('opt', result_opt['best_params'], result_opt['results'][1,2, :].mean()))
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
plt.show()
pass
