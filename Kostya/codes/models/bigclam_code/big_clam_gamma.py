import numpy as np
from matplotlib import pyplot as plt
from scipy.special import digamma, gammaln

from .big_clam import BigClam


class BigClamGamma(BigClam):
    def __init__(self, A=None, K=None, theta=1, debug_output=False, LLH_output=True, sparsity_coef = 0, initF='cond', eps=1e-6,
                 iter_output=None, alpha=None, rand_init_coef=0.1, stepSizeMod="simple", processesNo=None, save_hist=False, pow=0.2, max_iter=1000000, dump=False, dump_name = None):
        self.weights = A.copy()
        A = 1.0 * (self.weights != 0)

        super(BigClamGamma, self).__init__(A, K, debug_output, LLH_output, sparsity_coef, initF, eps,
                 iter_output, alpha, rand_init_coef, stepSizeMod, processesNo, save_hist, max_iter, dump, dump_name)
        self.theta = theta
        self.Fbord = 1
        self.logA = np.log(self.weights + self.epsCommForce)
        self.sqrt_pow = np.sqrt(pow)

    def initRandF(self):
        F = 0 + np.sqrt(np.max(self.weights))*np.random.rand(self.N, self.K)
        return F

    def loglikelihood_u(self, F, u=None, newFu=None):
        llh_u = super(BigClamGamma, self).loglikelihood_u(self.sqrt_pow * F, u, newFu)

        if newFu is not None:
            Fu = newFu
        else:
            Fu = F[u]

        indx = self.A[u, :] != 0
        neigF = F[indx, :]
        FF = Fu.dot(neigF.T)
        S1 = np.sum(-gammaln(FF + self.Fbord) - (FF + self.Fbord) * np.log(self.theta))
        S2 = np.sum(FF * self.logA[indx, u] - self.weights[indx, u] / self.theta)

        return llh_u + S1 + S2

    def loglikelihood_w(self, F):
        FF = F.dot(F.T)
        P = -gammaln(FF + self.Fbord) - (FF + self.Fbord) * np.log(self.theta)
        S1 = np.sum(P[self.A == 1])
        P2 = (FF * self.logA - self.weights / self.theta)
        S2 = np.sum(P2[self.A == 1])
        return S1 + S2

    def loglikelihood(self, F):
        llh = super(BigClamGamma, self).loglikelihood(self.sqrt_pow * F)
        llh_w = self.loglikelihood_w(F)
        return llh + llh_w

    def gradient(self, F, u=None):
        grad = super(BigClamGamma, self).gradient(self.sqrt_pow * F, u)
        grad_w = self.gradient_w(F, u)
        res = grad + grad_w
        m = max(np.abs(res))
        if m > 100:
            res = res * 100.0 / m
        return res

    def gradient_w(self, F, u=None):
        if u is None:
            raise NotImplemented
        else:
            FF = F[u].dot(F.T)
            DD = digamma(FF + self.Fbord)
            S1 = DD.T[self.A[:, u] == 1, None]
            S2 = (np.log(self.theta) - self.logA[self.A[:, u] == 1, u])[:, None]
            f = -F[self.A[:, u] == 1, :]
            grad = np.sum(f * (S1 + S2), axis=0)

            return grad
