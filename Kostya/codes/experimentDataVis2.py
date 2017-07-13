import pickle
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 1})
tips = sns.load_dataset("tips")


def visResult(filename, title=''):
    data = pickle.load(open(filename, 'rb'))
    results = data['results']

    for j in range(results.shape[1]):
        plt.figure()
        plt.title(['f1-macro', 'f1-micro', 'accuracy'][j])
        plt.ylim([0, 1])
        plt.scatter(range(results.shape[2]), results[0, j, :], c='r')
        plt.scatter(range(results.shape[2]), results[1, j, :], c='g')
    plt.title(title)

prefix = './dumps/results/'
datanames = ['PPI',
             #'l_partition-n200-m2-p0.1-q0.01-s100',
             #'l_partition-n200-m2-p0.2-q0.01-s100',
             #'l_partition-n200-m2-p0.2-q0.1-s100'
             ]


#D = [2, 4, 8, 16, 24, 32, 64, 96]
D = [32, 64, 128, 192, 256]
template = '{}{}_{}_d{}.dump'
#['stoc_hist', 'node2vec', 'stoc_opt', 'stocsk_opt', 'bigclam', 'gamma', 'deepwalk', 'svd', 'nmf']
methods = ['stoc_hist', 'node2vec', 'stoc_opt', 'stocsk_opt', 'bigclam', 'gamma', 'deepwalk', 'svd', 'nmf']#['bigclam', 'stoc_opt',  'stoc_hist', 'node2vec', 'deepwalk']

for dataname in datanames:
    pd_ress = []

    for d in D:
        for method in methods:
            filename = template.format(prefix, dataname, method, d)
            try:
                data = pickle.load(open(filename, 'rb'))
                results = data['results']
                results['embedding model'] = method
                pd_ress.append(results)
            except:
                pass

    pd_res = pd.concat(pd_ress, ignore_index=True)

    plt.subplot(121)

    ax = sns.pointplot(x="dim", y="value", hue="embedding model", data=pd_res.loc[(pd_res['train|test'] == 'test') & (pd_res['scorer'] == 'f1-macro')])
    plt.ylim([0, 1])
    plt.title('{} dataset test'.format(dataname))
    plt.subplot(122)
    ax.legend_.remove()
    ax = sns.pointplot(x="dim", y="value", hue="embedding model", data=pd_res.loc[(pd_res['train|test'] == 'train') & (pd_res['scorer'] == 'f1-macro')])
    plt.ylim([0, 1])
    plt.title('{} dataset train'.format(dataname))
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()
    pass