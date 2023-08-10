import numpy as np
import cvxpy as cp
from numpy.lib import utils
import torch
import scipy.spatial
import time
from scipy.stats import wishart
from sklearn import metrics
from numpy import linalg as LA
from scipy.sparse.linalg import eigs

dtype = torch.float
device = torch.device("cpu")

def is_psd(mat):
    return bool((mat == mat.T).all() and (torch.linalg.eigvals(mat).real>=0).all())

def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)

def MatConvert(x, device, dtype):
    """convert the numpy to a torch tensor."""
    x = torch.from_numpy(x).to(device, dtype)
    return x

def Linear_MMD_coeff_revision(X, Y, Lambda, sigma=1):
    # generate parameters for input data points
    # Input:
    #     X: data from mu, dim: n*D
    #     Y: data from nu, dim: n*D
    #Lambda: variance regularization parameter
    #Output:
    #     A: matrix, dim: D*D
    #     t: vector, dim: D*1
    nX, D = np.shape(X)

    hh_hist = []

    t_hist = []
    for i in range(D):
        X_i = X[:,i]
        Y_i = Y[:,i]

        Data_xx = (X_i.reshape([-1,1]) - X_i.reshape([1,-1]))**2
        Data_yy = (Y_i.reshape([-1,1]) - Y_i.reshape([1,-1]))**2
        Data_xy = (X_i.reshape([-1,1]) - Y_i.reshape([1,-1]))**2
        tau = np.median(Data_xy) * sigma

        Kx = MatConvert(np.exp(-Data_xx/(2*tau)), device, dtype)
        Ky = MatConvert(np.exp(-Data_yy/(2*tau)), device, dtype)
        Kxy = MatConvert(np.exp(-Data_xy/(2*tau)), device, dtype)

        Kxxy = torch.cat((Kx,Kxy),1)
        Kyxy = torch.cat((Kxy.transpose(0,1),Ky),1)
        Kxyxy = torch.cat((Kxxy,Kyxy),0)
        
        hh_i = Kx+Ky-Kxy-Kxy.transpose(0,1)
        hh_hist.append(hh_i)

        t_i = torch.div((torch.sum(hh_i) - torch.sum(torch.diag(hh_i))), (nX * (nX - 1)))
        t_hist.append(t_i)


    hh_hist = torch.stack(hh_hist, dim=2)
    #print(hh_hist.shape)
    
    # formulating vector t
    t = torch.stack(t_hist, dim=0).reshape([-1,1])
    
    # formulating matrix h_i
    hh_hist_sum_i = torch.sum(hh_hist,dim=1)
    hh_hist_sum_total = torch.sum(hh_hist_sum_i,dim=0).reshape([-1,1])/(nX**2)
    
    # formulating matrix A
    A = -4*Lambda * hh_hist_sum_total@hh_hist_sum_total.T + 4*Lambda * (hh_hist_sum_i.T@hh_hist_sum_i)/(nX**3)
    
    A_np = A.numpy()
    A_np = (A_np + A_np.T)/2
    #print(A)
    w, v = LA.eig(A_np)
    Lambda_min = np.min(w)
    if Lambda_min <= 0:
        A_np = A_np - Lambda_min * np.eye(D)
    

    return A_np, t.numpy()

def Linear_MMD_testing_revision(X_Te, Y_Te, z, num_perm = 100, sigma=1):
    #  perform permutation testing on testing samples
    #    Input:
    #     X_Te: data from mu, dim: n*D
    #     Y_Te: data from nu, dim: n*D
    #        z: projection vector
    #Output:
    # decision: reject H0 or not
    #      eta: testing statistics
    #  t_alpha: critical statistics

    nX_Te, D = np.shape(X_Te)
    nY_Te, _ = np.shape(Y_Te)
    n_Te = nX_Te + nY_Te

    Kxyxy_hist = []

    for i in range(D):
        X_i = X_Te[:,i]
        Y_i = Y_Te[:,i]

        Data_xx = (X_i.reshape([-1,1]) - X_i.reshape([1,-1]))**2
        Data_yy = (Y_i.reshape([-1,1]) - Y_i.reshape([1,-1]))**2
        Data_xy = (X_i.reshape([-1,1]) - Y_i.reshape([1,-1]))**2
        tau = np.median(Data_xy)*sigma
        #tau = 1

        Kx = MatConvert(np.exp(-Data_xx/(2*tau)), device, dtype)
        Ky = MatConvert(np.exp(-Data_yy/(2*tau)), device, dtype)
        Kxy = MatConvert(np.exp(-Data_xy/(2*tau)), device, dtype)

        Kxxy = torch.cat((Kx,Kxy),1)
        Kyxy = torch.cat((Kxy.transpose(0,1),Ky),1)
        Kxyxy = torch.cat((Kxxy,Kyxy),0)
        Kxyxy_hist.append(Kxyxy)
    
    Kxyxy_hist = torch.stack(Kxyxy_hist, dim=2)
    Kxyxy_z = torch.sum(Kxyxy_hist*z.reshape([1,1,-1]), dim=2)

    _, p_val, _ = mmd2_permutations(Kxyxy_z, nX_Te, permutations=num_perm)

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
    return est.item(), p_val.item(), rest


