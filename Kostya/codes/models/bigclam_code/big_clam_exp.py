import pickle

import networkx as nx
import numpy as np

from .Extends import draw_groups
from algorithms.big_clam import BigClam


class BigClamExp(BigClam):
    def __init__(self, A=None, K=None, theta=1):
        super(BigClamExp, self).__init__(A, K)
        self.theta = theta
        #self.A[self.A == 0] = 0.01

    def init_sumF(self, F):
        pass

    def update_sumF(self, newFu, oldFu):
        pass

    def loglikelihood(self, F, u=None, newFu=None):
        if newFu is not None:
            Fu = newFu
        else:
            Fu = F[u]

        if u is None:
            FF = F.dot(F.T)
            P = np.log(FF + self.epsCommForce)
            return -np.sum(P * self.A) + np.sum(P)
        else:
            neig_indx = np.where(self.A[u, :])
            neigF = F[neig_indx]
            w = self.A[u, neig_indx]
            #f = np.ma.array(F, mask=False)
            #f.mask[u] = True
            # print np.all(F >= 0)
            return -np.sum(Fu.dot(neigF.T) * w) + np.sum(np.log(Fu.dot(F.T) + self.epsCommForce))

    def gradient(self, F, u=None):
        if u is None:
            raise NotImplemented
        else:
            neig_indx = np.where(self.A[u, :])
            neigF = F[neig_indx]
            w = self.A[u, neig_indx]
            #f = np.ma.array(F, mask=False)
            #f.mask[u] = True
            PP = 1 / (F[u].dot(F.T) + self.epsCommForce)
            S1 = np.sum(PP[:, None] * F, axis=0)
            S2 = -np.sum(w.T * neigF, axis=0)
            grad = S1 + S2

            grad[grad > 10] = 10
            grad[grad < -10] = -10
            # is C++ code grad[grad > 10] = 10
            # is C++ code grad[grad < -10] = -10

        return grad

    def initNeighborComF(self, A=None):
        if A is None:
            A = self.A.copy()

        zero = A < np.mean(A)
        A[zero] = 0
        A[np.logical_not(zero)] = 1
        res = super(BigClamExp, self).initNeighborComF(A)
        #res[res==0] = 0.001

        return res


if __name__ == "__main__":
    K = 2
    D = pickle.load(file('../data/vk/8357895.ego'))
    G = nx.Graph(D)
    A = np.array(nx.to_numpy_matrix(G))
    N = A.shape[0]
    ids = list(G.node.keys())
    names = pickle.load(file('../data/vk/8357895.frName'))
    names = dict([(x['id'], x['first_name'][0] + '.' + x['last_name']) for x in names])

    bigClamExp = BigClamExp(A, K)
    bigClam = BigClam(A, K)

    Fexp = bigClamExp.fit(A, K)
    F = bigClam.fit(A, K)

    mse = np.linalg.norm((A - F.dot(F.T)))
    FF_exp = Fexp.dot(Fexp.T)
    mse_exp = np.linalg.norm((A - FF_exp * np.exp(-FF_exp)))
    print('MSE> exp: {}, basic: {}, win - {}'.format(mse_exp, mse, 'basic' if mse < mse_exp else '!!! Exp !!!'))

    #plt.matshow(F[:100,:])
    #plt.show()

    draw_groups(A, F, ids, names, 'ExpBigClamK', dpi=1000)

    # N, K = F.shape
    # F = 1 / F
    # plt.matshow(F)
    # plt.show()
    # C = F > np.sum(A) / (A.shape[0] * (A.shape[0] - 1))
    # plt.matshow(C)
    # plt.show()
    # C = F > np.mean(F)
    # plt.matshow(C)
    # plt.show()
    # indx = np.argmax(F, axis=1)
    # for i in xrange(N):
    #     C[i, indx[i]] = True
    # print F
    # print C

    # plt.plot(*bigClam.LLH)
    # plt.xlabel('iteration')
    # plt.ylabel('loglikelihood')
    # plt.grid(True)
    #
    # plt.show()

