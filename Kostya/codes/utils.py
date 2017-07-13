import os
import pickle

import imageio
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from codes.log_progress import log_progress
from codes.models.HistLossGraph2vec import HistLossGraph2vec

# sns.set_style('white')

bin_num = 128
min_cov = 1e-6

def distance_histogram_png_from_file(embeding_filename, nx_G, savepath):
    nodes, embedding = HistLossGraph2vec.load(embeding_filename)
    distance_histogram_png(embedding, nx_G, savepath)

def get_sims(embedding, nx_G):
    nodes = nx_G.nodes()
    pos_mask = nx.adjacency_matrix(nx_G).todense()
    neg_mask = 1 - pos_mask - np.eye(*pos_mask.shape).astype(int)

    e = np.array(embedding[nodes])
    sim = e.T.dot(e)

    sim_pos = sim[pos_mask.nonzero()]
    sim_neg = sim[neg_mask.nonzero()]

    return sim_pos, sim_neg

def make_hist(sim):
    hist = calc_hist_vals_vector(sim, -1.0, 1.0)
    hist /= hist.sum()
    return hist

def loss(sim_pos, sim_neg):
    hist_pos = make_hist(sim_pos)
    hist_neg = make_hist(sim_neg)
    return hist_loss(hist_neg, hist_pos)

def hist_loss(hist_neg, hist_pos):
    agg_pos = np.zeros(shape=(bin_num,))
    agg_pos2 = np.cumsum(hist_pos)
    for ind in np.arange(bin_num):
        agg_pos[ind] = np.sum(hist_pos[0:ind+1])

    # agg_pos, _ = theano.scan(fn=lambda ind, A: T.sum(A[0:ind + 1]),
    #                          outputs_info=None,
    #                          sequences=T.arange(self.bin_num),
    #                          non_sequences=hist_pos)

    return np.sum(np.dot(agg_pos, hist_neg))

def calc_hist_vals_vector(sim, hist_min, hist_max):
    sim_mat = np.tile(sim[:, None], (1, bin_num))
    w = max((hist_max - hist_min) / bin_num, min_cov)
    grid_vals = np.arange(0, bin_num) * (hist_max - hist_min) / bin_num + hist_min + w / 2.0
    grid = np.tile(grid_vals, (sim.shape[0], 1))
    w_triang = 4.0 * w
    D = np.abs(grid - sim_mat)
    mask = (D <= w_triang / 2)
    D_fin = w_triang * (D * (-2.0 / w_triang ** 2) + 1.0 / w_triang) * mask
    hist_corr = np.sum(D_fin, 0) + min_cov
    return hist_corr

def distance_histogram_png(embedding, nx_G, savepath):
    sim_pos, sim_neg = get_sims(embedding, nx_G)
    loss(sim_pos, sim_neg)
    sns.distplot(sim_pos, 32, label='positive')
    sns.distplot(sim_neg, 32, label='negative')
    plt.legend()
    plt.axis('on')
    plt.grid('on')
    plt.tight_layout()
    plt.savefig(savepath)
    plt.close()


def vis_embeddings(graph, embedding, savename, labels=None, i=None, ):
    dir_name = '../histplot/{}/'.format(savename)
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    filesave_temp = dir_name + '{}-{}.png'.format(savename, str(i).zfill(6))

    cols = 'b' if labels is None else labels
    node_size = 50
    edge_color = '#222222'
    alpha = 0.8
    width = 0.1
    plt.figure(1, figsize=(10, 5))
    plt.subplot(121)
    nx.draw_networkx(graph,
                     pos=embedding,
                     node_size=node_size,
                     node_color=cols,
                     edge_color=edge_color,
                     alpha=alpha,
                     width=width,
                     cmap=plt.get_cmap('plasma'),
                     with_labels=False, )
    plt.title('{} dataset, {} iter'.format(savename, str(i).zfill(6)))
    plt.subplot(122)
    distance_histogram_png(embedding, graph, filesave_temp)


def exp_file2df(methods, filenames):
    pd_res = []
    for filename, method in zip(filenames, methods):
        data = pickle.load(open(filename, 'rb'))
        results = data['results']
        for j in range(results.shape[1]):
            for k in range(results.shape[2]):
                pd_res.append([method, ['f1-macro', 'f1-micro', 'accuracy'][j], 'train', results[0, j, k]])
                pd_res.append([method, ['f1-macro', 'f1-micro', 'accuracy'][j], 'test', results[1, j, k]])

    pd_res2 = pd.DataFrame(pd_res, columns=['method', 'scorer', 'type', 'value'])
    return pd_res2


def make_gif(folder):
    with imageio.get_writer(folder + 'animation.gif', mode='I', duration=0.1) as writer:
        files = os.listdir(folder)
        files.sort(key=lambda x: os.stat(os.path.join(folder, x)).st_mtime)
        for filename in log_progress(files):
            image = imageio.imread(folder + filename)
            writer.append_data(image)
