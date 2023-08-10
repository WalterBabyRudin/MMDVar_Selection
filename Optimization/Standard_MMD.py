import numpy as np
import cvxpy as cp
from numpy.lib import utils
import torch
import scipy.spatial
import time
from scipy.stats import wishart
from sklearn import metrics

from scipy.sparse.linalg import eigs
from scipy import linalg

dtype = torch.float
device = torch.device("cpu")

def kernelwidthPair(x1, x2):
    '''Implementation of the median heuristic. See Gretton 2012
       Pick sigma such that the exponent of exp(- ||x-y|| / (2*sigma2)),
       in other words ||x-y|| / (2*sigma2),  equals 1 for the median distance x
       and y of all distances between points from both data sets X and Y.
    '''
    n, nfeatures = x1.shape
    m, mfeatures = x2.shape
    
    k1 = np.sum((x1*x1), 1)
    q = np.tile(k1, (m, 1)).transpose()
    del k1
    
    k2 = np.sum((x2*x2), 1)
    r = np.tile(k2, (n, 1))
    del k2
    
    h= q + r
    del q,r
    
    # The norm
    h = h - 2*np.dot(x1,x2.transpose())
    h = np.array(h, dtype=float)
    
    mdist = np.median([i for i in h.flat if i])

    return mdist

def Pdist2(x, y):
    """compute the paired distance between x and y."""
    x_norm = (x ** 2).sum(1).view(-1, 1)
    if y is not None:
        y_norm = (y ** 2).sum(1).view(1, -1)
    else:
        y = x
        y_norm = x_norm.view(1, -1)
    Pdist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
    Pdist[Pdist<0]=0
    return Pdist

def MatConvert(x, device, dtype):
    """convert the numpy to a torch tensor."""
    x = torch.from_numpy(x).to(device, dtype)
    return x

def naive_quadratic_MMD_testing(X_Te, Y_Te, c, num_perm = 500):
    #  perform permutation testing on testing samples
    #    Input:
    #     X_Te: data from mu, dim: n*D
    #     Y_Te: data from nu, dim: n*D
    #        c: kernel bandwidth
    #Output:
    # decision: reject H0 or not
    nX_Te, D = np.shape(X_Te)
    nY_Te, _ = np.shape(Y_Te)

    Kx =  MatConvert(metrics.pairwise.polynomial_kernel(X_Te, X_Te, 2, 1, c), device, dtype)
    Ky =  MatConvert(metrics.pairwise.polynomial_kernel(Y_Te, Y_Te, 2, 1, c), device, dtype)
    Kxy = MatConvert(metrics.pairwise.polynomial_kernel(X_Te, Y_Te, 2, 1, c), device, dtype)

    Kxxy = torch.cat((Kx,Kxy),1)
    Kyxy = torch.cat((Kxy.transpose(0,1),Ky),1)
    Kxyxy = torch.cat((Kxxy,Kyxy),0)

    _, p_val, _ = mmd2_permutations(Kxyxy, nX_Te, permutations=num_perm)

    if p_val <= 0.05:
        decision = 1
    else:
        decision = 0
    return p_val, decision

def mmd2_permutations(K, n_X, permutations=500):
    """
        Fast implementation of permutations using kernel matrix.
    """
    K = torch.as_tensor(K)
    n = K.shape[0]
    assert K.shape[0] == K.shape[1]
    n_Y = n_X
    assert n == n_X + n_Y
    w_X = 1
    w_Y = -1
    ws = torch.full((permutations + 1, n), w_Y, dtype=K.dtype, device=K.device)
    ws[-1, :n_X] = w_X
    for i in range(permutations):
        ws[i, torch.randperm(n)[:n_X].numpy()] = w_X
    biased_ests = torch.einsum("pi,ij,pj->p", ws, K, ws)
    if True:  # u-stat estimator
        # need to subtract \sum_i k(X_i, X_i) + k(Y_i, Y_i) + 2 k(X_i, Y_i)
        # first two are just trace, but last is harder:
        is_X = ws > 0
        X_inds = is_X.nonzero()[:, 1].view(permutations + 1, n_X)
        Y_inds = (~is_X).nonzero()[:, 1].view(permutations + 1, n_Y)
        del is_X, ws
        cross_terms = K.take(Y_inds * n + X_inds).sum(1)
        del X_inds, Y_inds
        ests = (biased_ests - K.trace() + 2 * cross_terms) / (n_X * (n_X - 1))
    est = ests[-1]
    rest = ests[:-1]
    p_val = (rest > est).float().mean()

    #print(est.item())
    print(rest)
    return est.item(), p_val.item(), rest

def naive_linear_MMD_testing(X_Te, Y_Te, num_perm = 500):
    #  perform permutation testing on testing samples
    #    Input:
    #     X_Te: data from mu, dim: n*D
    #     Y_Te: data from nu, dim: n*D
    #Output:
    # decision: reject H0 or not
    nX_Te, D = np.shape(X_Te)
    nY_Te, _ = np.shape(Y_Te)
    n_Te = nX_Te + nY_Te

    Kx =  MatConvert(metrics.pairwise.linear_kernel(X_Te, X_Te), device, dtype)
    Ky =  MatConvert(metrics.pairwise.linear_kernel(Y_Te, Y_Te), device, dtype)
    Kxy = MatConvert(metrics.pairwise.linear_kernel(X_Te, Y_Te), device, dtype)

    Kxxy = torch.cat((Kx,Kxy),1)
    Kyxy = torch.cat((Kxy.transpose(0,1),Ky),1)
    Kxyxy = torch.cat((Kxxy,Kyxy),0)

    _, p_val, _ = mmd2_permutations(Kxyxy, nX_Te, permutations=num_perm)

    if p_val <= 0.05:
        decision = 1
    else:
        decision = 0
    return p_val, decision

def naive_Gaussian_MMD_testing(X_Te, Y_Te, sigma, num_perm = 500):
    #  perform permutation testing on testing samples
    #    Input:
    #     X_Te: data from mu, dim: n*D
    #     Y_Te: data from nu, dim: n*D
    #Output:
    # decision: reject H0 or not
    nX_Te, D = np.shape(X_Te)
    nY_Te, _ = np.shape(Y_Te)

    X_Te_torch = torch.tensor(X_Te)
    Y_Te_torch = torch.tensor(Y_Te)

    Dxx = Pdist2(X_Te_torch,X_Te_torch)
    Dyy = Pdist2(Y_Te_torch,Y_Te_torch)
    Dxy = Pdist2(X_Te_torch,Y_Te_torch)

    #sigma = torch.median(Dxy)
    Kx = torch.exp(-Dxx / (2*sigma))
    Ky = torch.exp(-Dyy / (2*sigma))
    Kxy = torch.exp(-Dxy / (2*sigma))
    Kxxy = torch.cat((Kx,Kxy),1)
    Kyxy = torch.cat((Kxy.transpose(0,1),Ky),1)
    Kxyxy = torch.cat((Kxxy,Kyxy),0)

    _, p_val, _ = mmd2_permutations(Kxyxy, nX_Te, permutations=num_perm)

    if p_val <= 0.05:
        decision = 1
    else:
        decision = 0
    return p_val, decision