import numpy as np
import torch
import copy as cp


EPS = 1e-9


class NMF:
    def __init__(self, n_components, tol, max_iter, verbose=False):
        self.n_components = n_components
        self.tol = tol
        self.max_iter = max_iter
        self.verbose = verbose
        self.H = []

    def fit(self, V):
        V_cuda = torch.tensor(V, dtype=torch.float32).cuda()
        samples, self.dim = V_cuda.shape
        avg = torch.sqrt(torch.mean(V_cuda) / self.n_components)
        self.W_cuda = torch.abs(avg * torch.randn(samples, self.n_components, dtype=torch.float32).cuda())
        self.H_cuda = torch.abs(avg * torch.randn(self.n_components, self.dim, dtype=torch.float32).cuda())
        _transform(V=V_cuda, W=self.W_cuda, H=self.H_cuda, max_iter=self.max_iter, verbose=self.verbose, tol=self.tol)

    def transform(self, V):
        V_cuda = torch.tensor(V, dtype=torch.float32).cuda()
        samples, _ = V_cuda.shape
        W_cuda = torch.rand(samples, self.n_components, dtype=torch.float32).cuda()
        _transform(V=V, W=W_cuda, H=cp.copy(self.H_cuda), max_iter=self.max_iter, verbose=self.verbose, tol=self.tol)


def _transform(V, W, H, max_iter, verbose, tol):
    violation_init = torch.sum(torch.sqrt(torch.pow(V - torch.matmul(W, H), 2)))
    for i in range(max_iter):
        H = H * torch.matmul(W.T, V) / (torch.matmul(torch.matmul(W.T, W), H) + EPS)
        W = W * torch.matmul(V, H.T) / (torch.matmul(torch.matmul(W, H), H.T) + EPS)
        norms = torch.sqrt(torch.sum(torch.pow(H.T, 2)))
        H /= norms
        W *= norms

        violation = torch.sum(torch.sqrt(torch.pow(V - torch.matmul(W, H), 2)))
        Violation = violation / violation_init
        if verbose:
            print("violation: {}".format(violation))
        if Violation < tol:
            break
    return



