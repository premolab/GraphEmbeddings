import numpy as np
from codes.Experiment import run_experiment
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
import pickle
from codes.load_data import generate_graphs

sns.set(style="white")

n = 100
frac = 0.5
p = 0.2
Q = [0.01, 0.02, 0.05, 0.1, 0.15]
D = [2, 4, 8, 16]
models = ['stoc_hist', 'hist', 'node2vec', 'svd', 'nmf', 'bigclam', 'gamma']  # 'opt',

result = []
for seed in [12124, 1241, 100, 35, 15]:
    for d in D:
        for q, args in zip(Q, generate_graphs(n, frac, p, Q, seed)):
            A, true_comm, name = args
            for model in models:
                result.append(run_experiment(A, true_comm, d,
                                             name, model, load_dumped_res=True)['results'])
                result[-1]['out cluster edge prob'] = q
                result[-1]['seed'] = seed

        temp_result = pd.concat(result, ignore_index=True)
        pickle.dump(temp_result, open('./dumps/result_l-partition-temp', 'wb'))


result = pickle.load(open('./dumps/result_l-partition-temp', 'rb'))
for d in D:
    plt.figure(figsize=(14, 7))
    plt.subplot(121)
    ax = sns.pointplot(x='out cluster edge prob', y="value", hue="embedding model",
                       data=result[(result['train|test'] == 'test') & (result['dim'] == d) ])
    plt.ylim([0, 1])
    plt.title('test d={}'.format(d))
    plt.grid()
    plt.subplot(122)
    ax.legend_.remove()
    ax = sns.pointplot(x="out cluster edge prob", y="value", hue="embedding model",
                       data=result[(result['train|test'] == 'train') &  (result['dim'] == d)])
    plt.ylim([0, 1])
    plt.title('train d={}'.format(d))
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.grid()
    plt.savefig('{}.png'.format(d))
plt.show()
pass
