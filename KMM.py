import numpy as np
import sklearn.metrics
from cvxopt import matrix, solvers


#%% Kernel
def kernel(ker, X1, X2, gamma):
    K = None
    if ker == 'linear':
        if X2 is not None:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1), np.asarray(X2))
        else:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1))
    elif ker == 'rbf':
        if X2 is not None:
            K = sklearn.metrics.pairwise.rbf_kernel(np.asarray(X1), np.asarray(X2), gamma)
        else:
            K = sklearn.metrics.pairwise.rbf_kernel(np.asarray(X1), None, gamma)
    return K


#%% Kernel Mean Matching (KMM)
class KMM:
    def __init__(self, kernel_type='linear', gamma=1.0, B=1.0, eps=None):
        '''
        Initialization function
        :param kernel_type: 'linear' | 'rbf'
        :param gamma: kernel bandwidth for rbf kernel
        :param B: bound for beta
        :param eps: bound for sigma_beta
        '''
        self.kernel_type = kernel_type
        self.gamma = gamma
        self.B = B
        self.eps = eps

    def fit(self, Xs, Xt):
        '''
        Fit source and target using KMM (compute the coefficients)
        :param Xs: ns * dim
        :param Xt: nt * dim
        :return: Coefficients (Pt / Ps) value vector (Beta in the paper)
        '''
        ns = Xs.shape[0]
        nt = Xt.shape[0]
        if self.eps == None:
            self.eps = self.B / np.sqrt(ns)
        K = np.float64(kernel(self.kernel_type, Xs, None, self.gamma))
        kappa = np.float64(np.sum(kernel(self.kernel_type, Xs, Xt, self.gamma) * float(ns) / float(nt), axis=1))

        K = matrix(K)
        kappa = matrix(kappa)
        G = matrix(np.r_[np.ones((1, ns)), -np.ones((1, ns)), np.eye(ns), -np.eye(ns)])
        h = matrix(np.r_[ns * (1 + self.eps), ns * (self.eps - 1), self.B * np.ones((ns,)), np.zeros((ns,))])

        sol = solvers.qp(K, -kappa, G, h)
        beta = np.array(sol['x'])

        self.evaluate_beta(Xs, Xt, beta)
        return beta

    def fit_in_batches(self, Xs, Xt, batch_size):
        beta = []
        ns = Xs.shape[0]
        for start in range(0, ns, batch_size):
            end = start + batch_size if start + batch_size < ns else ns
            beta.append(self.fit(Xs[start:end], Xt))

        beta = np.concatenate(beta, axis=0)
        return beta

    def evaluate_beta(self, Xs, Xt, beta):
        Xs_hate = beta * Xs
        print('Source mean: ', Xs.mean(axis=0))
        print('Weighted source mean: ', Xs_hate.mean(axis=0))
        print('Target mean: ', Xt.mean(axis=0))
