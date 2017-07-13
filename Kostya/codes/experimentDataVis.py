import pickle
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from codes.utils import exp_file2df

sns.set_style("whitegrid")

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
dataname = 'football'
D = [2, 3, 5, 11, 15, 20]
d = 3


methods = ['hist', 'opt', 'node2vec', 'nmf', 'svd', 'gamma', 'bigclam']

for d in D:
    plt.figure()
    template = '{}result_{}_{}_d{}.dump'.format(prefix, dataname, '{}', d)

    pd_res2 = exp_file2df(methods, [template.format(x) for x in methods])

    plt.subplot(121)
    ax = sns.barplot(x="scorer", y="value", hue="method", data=pd_res2[pd_res2['type'] == 'test'])
    plt.legend(loc='upper right')
    plt.ylim([0, 1.2])
    plt.title('test d={}'.format(d))
    plt.subplot(122)
    ax = sns.barplot(x="scorer", y="value", hue="method", data=pd_res2[pd_res2['type'] == 'train'])
    ax.legend_.remove()
    plt.ylim([0, 1.2])
    plt.title('train d={}'.format(d))
    plt.tight_layout()
plt.show()
pass