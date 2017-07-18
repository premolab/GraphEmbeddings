import downhill
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import numpy as np
import pandas as pd
import networkx as nx
from settings import PATH_TO_DUMPS

from load_data import load_blog_catalog

N = 10312
dim = 32
srng = RandomStreams(seed=234)
neg_sampling = True
bin_num = 64

A = T.dmatrix('A')
E = T.dmatrix('E')

w = theano.shared(np.random.normal(size=(N, dim)))
b = theano.shared(np.zeros((N, )))

E_norm = E/E.norm(2, axis=1).reshape((E.shape[0], 1))

E_corr = T.dot(E_norm, E_norm.T)

pos_mask = A
neg_mask = 1 - pos_mask - T.eye(pos_mask.shape[0])

pos_samples = E_corr[pos_mask.nonzero()]
neg_samples = E_corr[neg_mask.nonzero()]

if neg_sampling:
    neg_samples = neg_samples[srng.permutation(n=neg_samples.shape[0], size=(1, ))[0, : 2 * pos_samples.shape[0]]]
#    neg_samples = neg_samples[srng.choice(size=(2*pos_samples.shape[0], ), a=neg_samples.shape[0])]

f2 = theano.function(
    [A, E], [pos_samples, neg_samples]
)

def calc_hist_vec(samples, bin_num=bin_num):
    delta = 2/(bin_num - 1)
    # ts =  -1 + delta * T.arange(bin_num)
    rs = T.floor((samples + 1) / delta) + 1  # r is natural index
    ts = -1 + delta * (rs - 1)  # t_r is value between -1 and 1
    samples_sub_t_prev = samples - ts
    t_next_sub_samples = delta - samples_sub_t_prev

    H, _ = theano.map(
        lambda r: (
            T.sum(samples_sub_t_prev[T.eq(rs, r).nonzero()]) +
            T.sum(t_next_sub_samples[T.eq(rs+1, r+1).nonzero()])
        ),
        np.arange(1, bin_num + 1)
    )
    return H / delta

pos_hist = calc_hist_vec(pos_samples)
neg_hist = calc_hist_vec(neg_samples)

agg_pos = T.extra_ops.cumsum(pos_hist)
loss = T.sum(T.dot(agg_pos, neg_hist))

pass

# testing
import time

print('Compiling')
t = time.time()
f = theano.function(
    [A, E], [E_norm, E_corr, pos_hist, neg_hist, loss, pos_mask, neg_mask]
)
print(time.time() - t)

# t = time.time()
# res = f([[0, 1, 0], [1, 0, 1], [0, 1, 0]], [[1, 2, 3], [2, 3, 4], [-1, -2, -3]])
# print(time.time() - t)

print('Reading embedding')
t = time.time()
X = pd.read_csv(
    '{}/models/deepwalk_BlogCatalog_d32.csv'.format(PATH_TO_DUMPS),
    delim_whitespace=True, header=None,
    skiprows=1,
    index_col=0
).sort_index()
E_input = X.values
print(time.time() - t)

print('Reading graph')
t = time.time()
graph = load_blog_catalog()
nodes = graph.nodes()
adjacency_matrix = nx.adjacency_matrix(graph, nodes).todense().astype("float32")
print(time.time() - t)

print('Solving')
t = time.time()
res = f(adjacency_matrix, E_input)
print(time.time() - t)

print(res[0])
print(res[1][:2])
print(res[2])
print(res[3])
print(res[4])
print(res[5][:2])
print(res[6][:2])
