import downhill
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import numpy as np
import pandas as pd
import networkx as nx

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

E_corr, _ = theano.map(
    lambda e_i: theano.map(
        lambda e_j: T.dot(e_i, e_j),
        E_norm
    ),
    E_norm
)

pos_mask = A
neg_mask = 1 - pos_mask - T.eye(pos_mask.shape[0])

pos_samples = E_corr[pos_mask.nonzero()]
neg_samples = E_corr[neg_mask.nonzero()]

if neg_sampling:
    neg_samples = srng.permutation(n=neg_samples.shape[0], size=(0, 1))[0 : 2 * pos_samples.shape[0]]


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

t = time.time()
f = theano.function(
    [A, E], [pos_hist, neg_hist, loss]
)
print(time.time() - t)


t = time.time()
X = pd.read_csv(
    '/home/stas/PycharmProjects/GraphEmbeddings/dumps/models/deepwalk_BlogCatalog_d32.csv',
    delim_whitespace=True, header=None,
    skiprows=1,
    index_col=0
).sort_index()
E_input = X.values
print(time.time() - t)

t = time.time()
graph = load_blog_catalog()
nodes = graph.nodes()
adjacency_matrix = nx.adjacency_matrix(graph, nodes).todense().astype("float32")
print(time.time() - t)

t = time.time()
res = f(adjacency_matrix, E_input)
print(time.time() - t)

print(res[0])
print(res[1])
print(res[2])
print(res[3])
print(res[4])
