import pickle
import os
import subprocess

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .Extends import toBigClamFormat, fromBigClamFormat
from algorithms.big_clam import BigClam
from .big_clam_gamma import BigClamGamma


def bigclam_test():
    cur_path = os.getcwd()
    data_dir = '../data/'
    data_file_name = 'test'
    data_file_path = os.path.join(data_dir, data_file_name)
    bigClam_data_file_path = os.path.join(data_dir, data_file_name) + '.bigClam'
    bigClam_res_prefix = data_dir
    exe_path = '../external/snap-master/examples/bigclam/Debug/bigclam.exe'

    K = 2
    G = fromBigClamFormat(bigClam_data_file_path)
    G = max(nx.connected_component_subgraphs(G), key=len)
    A = nx.to_scipy_sparse_matrix(G)

    bigClam = BigClam(A, K)
    F = bigClam.fit()

    args = '-i:\"{}\" -o:\"{}\" -c:{} -nt:1'.format(os.path.join(cur_path, bigClam_data_file_path),
                                               os.path.join(cur_path, bigClam_res_prefix),  K)

    with open('bigClam_output.log', 'w') as output_f:
        res = subprocess.call('\"{}\" {}'.format(exe_path, args), stdout=output_f, stderr=output_f)


def ego_test():
    cur_path = os.getcwd()
    data_dir = '../data/vk/'
    data_file_name = '4870053.ego'
    data_file_path = os.path.join(data_dir, data_file_name)
    bigClam_data_file_path = data_file_path + '.bigClam'
    bigClam_res_prefix = data_dir
    exe_path = '../external/snap-master/examples/bigclam/Debug/bigclam.exe'

    K = 2
    D = pickle.load(file(data_file_path))
    G = nx.Graph(D)
    G = max(nx.connected_component_subgraphs(G), key=len)
    A = np.array(nx.to_numpy_matrix(G))

    toBigClamFormat(G, bigClam_data_file_path)

    bigClam = BigClam(A, K)
    F = bigClam.fit()

    args = '-i:\"{}\" -o:\"{}\" -c:{} -nt:1'.format(os.path.join(cur_path, bigClam_data_file_path),
                                               os.path.join(cur_path, bigClam_res_prefix),  K)
    with open('bigClam_output.log', 'w') as output_f:
        res = subprocess.call('\"{}\" {}'.format(exe_path, args), stdout=output_f, stderr=output_f)

def test3():
    cur_path = os.getcwd()
    data_dir = '../data/vk/'
    data_file_name = '4870053.ego'
    data_file_path = os.path.join(data_dir, data_file_name)
    bigClam_data_file_path = data_file_path + '.bigClam'
    bigClam_res_prefix = data_dir
    exe_path = '../external/snap-master/examples/bigclam/Debug/bigclam.exe'

    F_true = Fs3[0]
    B = gamma_model_test_data(F_true)
    x = np.mean(B) * 0.8
    A = B.copy()
    A[B < x] = 0
    A[B >= x] = 1
    K = 3
    G = nx.Graph(A)

    toBigClamFormat(G, bigClam_data_file_path)

    bigClam = BigClam(A, K)
    F = bigClam.fit()

    args = '-i:\"{}\" -o:\"{}\" -c:{} -nt:1'.format(os.path.join(cur_path, bigClam_data_file_path),
                                               os.path.join(cur_path, bigClam_res_prefix),  K)
    with open('bigClam_output.log', 'w') as output_f:
        res = subprocess.call('\"{}\" {}'.format(exe_path, args), stdout=output_f, stderr=output_f)
    pass


Fs2 = [2 * np.array([[1, 1, 1, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 1, 1, 1]]),
       2 * np.array([[1, 1, 1, 1, 1, 0, 0, 0], [0, 0, 0, 1, 1, 1, 1, 1]]),
       2 * np.array([[1, 1, 1, 1, 1, 1, 0, 0], [0, 0, 1, 1, 1, 1, 1, 1]]),
       2 * np.array([[1, 1, 1, 5, 5, 5, 0, 0], [0, 0, 5, 5, 5, 1, 1, 1]]),
       1 * np.array([[4, 4, 4, 4, 4, 1, 1, 1], [1, 1, 1, 4, 4, 4, 4, 4]]),
       1 * np.array([[3, 3, 3, 5, 5, 5, 1, 1], [1, 1, 5, 5, 5, 3, 3, 3]]),
       2 * np.array([[1] * 100 + [0] * 100, [0] * 100 + [1] * 100]),
       2 * np.array([[1] * 8 + [0] * 6, [0] * 6 + [1] * 8]), ]

Fs3 = [2 * np.array([[1] * 80 + [0] * 60,
                     [0] * 20 + [1] * 20 + [0] * 20 + [1] * 20 + [0] * 20 + [1] * 20 + [1] * 20,
                     [0] * 40 + [1] * 40 + [1] * 20 + [1] * 20 + [0] * 20]), ]

def gamma_model_test_data(F = None):
    np.random.seed(1122)
    if F is None:
        F = 2 * np.array([[1, 1, 1, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 1, 1, 1]])
    theta = F.T.dot(F)
    A = np.zeros(theta.shape)
    for i in range(theta.shape[0]):
        for j in range(theta.shape[1]):
            A[i][j] = np.random.gamma(1 + theta[i][j], 1)
    np.fill_diagonal(A, 0)
    return A

def draw_matrix(photo_l, title="", hide_ticks=True):
    if photo_l.shape[0] / photo_l.shape[1] > 5 or photo_l.shape[0] / photo_l.shape[1] < 0.2:
        k = np.floor(np.max(photo_l.shape) / np.min(photo_l.shape) / 2)
        if photo_l.shape[1] < photo_l.shape[0]:
            photo_l = np.reshape(np.tile(photo_l.copy(), (k, 1)).T, (photo_l.shape[1]*k, photo_l.shape[0])).T
        else:
            photo_l = np.reshape(np.tile(photo_l.copy(), (1, k)), (photo_l.shape[0] * k, photo_l.shape[1]))
    ax = plt.gca()
    im = plt.imshow(photo_l, interpolation='none')
    if hide_ticks:
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.title(title, y=1.02, x = 0.6)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="7%", pad=0.07)
    plt.colorbar(im, cax=cax, ticks=np.linspace(np.min(photo_l), np.max(photo_l), 5))

def test_gamma_model():

    for F_true in Fs2:
        A = gamma_model_test_data(F_true)
        gamma_model = BigClamGamma(A, 2, debug_output=False, LLH_output=False)
        F_model = gamma_model.fit()
        output_str = "True: \n{}\nModel estimation:\n{}\nDiff:\n{}\nRMSE/MeanVal:{}\n"
        print(output_str.format(F_true, np.around(F_model.T, 2), np.around(np.abs(F_true - F_model.T), 4),
                                np.sqrt(np.mean((F_true - F_model.T) ** 2)) / np.mean(F_true)))
        plt.figure()
        plt.subplot(131)
        plt.imshow(A, interpolation='none')
        plt.subplot(132)
        plt.imshow(F_true, interpolation='none')
        plt.subplot(133)
        plt.imshow(F_model.T, interpolation='none')
        plt.show()

    for F_true in Fs3:
        A = gamma_model_test_data(F_true)
        gamma_model = BigClamGamma(A, 3, debug_output=False, LLH_output=False)
        F_model = gamma_model.fit()
        output_str = "True: \n{}\nModel estimation:\n{}\nDiff:\n{}\nRMSE/MeanVal:{}\n"
        print(output_str.format(F_true, np.around(F_model.T, 2), np.around(np.abs(F_true - F_model.T), 4),
                                np.sqrt(np.mean((F_true - F_model.T) ** 2)) / np.mean(F_true)))
        plt.figure()
        plt.subplot(131)
        plt.imshow(A, interpolation='none')
        plt.subplot(132)
        plt.imshow(F_true.T.dot(F_true), interpolation='none')
        plt.subplot(133)
        plt.imshow(F_model.T.dot(F_model), interpolation='none')
        plt.show()

def draw_res(B, F_true, F_model, F_model7=None):
    Xs, Ys = 2, 5
    f = plt.figure(figsize=(18, 6))

    plt.subplot(Xs, Ys, 1)
    draw_matrix(B, 'Agency matix sample (B)')
    plt.subplot(Xs, Ys, 2)
    draw_matrix(F_model.dot(F_model.T), 'Reconstructed matix (3 comm)')
    plt.subplot(Xs, Ys, 5)
    draw_matrix(np.abs(F_model.dot(F_model.T)-F_true.T.dot(F_true)), 'diff (3 comm)')

    plt.subplot(Xs, Ys, 6)
    draw_matrix(F_true, "true F value")
    plt.subplot(Xs, Ys, 7)
    draw_matrix(F_model.T, "Reconstructed F (3 comm)")

    if F_model7 is not None:
        plt.subplot(Xs, Ys, 3)
        draw_matrix(F_model7.dot(F_model7.T), 'Reconstructed matix (7 comm)')
        plt.subplot(Xs, Ys, 4)
        draw_matrix(np.abs(F_model7.dot(F_model7.T)-F_true.T.dot(F_true)), 'diff (7 comm)')
        plt.subplot(Xs, Ys, 8)
        draw_matrix(F_model7.T, "Reconstructed F (7 comm)")

    #plt.subplot(Xs, Ys, 9)
    #draw_matrix(np.abs(F_model.T - F_true), "diff (3 comm)")

if __name__ == "__main__":
    F_true = Fs3[0]
    #F_true = Fs2[-1]
    plt.figure(figsize=(18,6))
    plt.subplot(1, 5, 1)
    draw_matrix(F_true, "True F value")
    plt.subplot(1, 5, 2)
    draw_matrix(F_true.T.dot(F_true), "A generation model")
    A = gamma_model_test_data(F_true)
    plt.subplot(1, 5, 3)
    draw_matrix(A, "Agency matix sample (A)")
    x = np.mean(A) * 0.8
    print(x, np.mean(A))
    B = A.copy()
    B[B < x] = 0
    C = B.copy()
    C[C >=x] = 1
    plt.subplot(1, 5, 4)
    draw_matrix(B, "Zeros if A < {:.2f} (B)".format(x))
    plt.subplot(1, 5, 5)
    draw_matrix(C, "A < {:.2f} (C matrix)".format(x))

    w_model2 = BigClam(C, 3, debug_output=False, LLH_output=True, initF='rand', iter_output=1)
    F_model2, LLH2 = w_model2.fit()

    print(len(w_model2.LLH_output_vals))
    plt.figure()
    plt.plot(-np.log(-np.array(w_model2.LLH_output_vals[:3000])))
    draw_res(C, F_true, F_model2)

    plt.show()
