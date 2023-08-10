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
#Test:
#is_psd(torch.randn(2,2))
def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)


def MatConvert(x, device, dtype):
    """convert the numpy to a torch tensor."""
    x = torch.from_numpy(x).to(device, dtype)
    return x

def Linear_MMD_coeff(X, Y):
    # generate parameters for input data points
    # Input:
    #     X: data from mu, dim: n*D
    #     Y: data from nu, dim: n*D
    #Output:
    #     a: vector, dim: D

    n, D = np.shape(X)
    a = np.zeros(D)

    for k in range(D):
        X_k = X[:, k]
        Y_k = Y[:, k]

        a_k = (np.mean(X_k) - np.mean(Y_k))**2
        a[k] = a_k
    return a

def Linear_MMD_coeff_revision(X, Y, Lambda):
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
        tau = np.median(Data_xy)

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
    print(hh_hist.shape)
    
    # formulating vector t
    t = torch.stack(t_hist, dim=0).reshape([-1,1])
    
    # formulating matrix h_i
    hh_hist_sum_i = torch.sum(hh_hist,dim=1)
    hh_hist_sum_total = torch.sum(hh_hist_sum_i,dim=0).reshape([-1,1])/(nX**2)
    
    # formulating matrix A
    A = -4*Lambda * hh_hist_sum_total@hh_hist_sum_total.T + 4*Lambda * (hh_hist_sum_i.T@hh_hist_sum_i)/(nX**3)
    
    A_np = A.numpy()
    A_np = (A_np + A_np.T)/2
    w, v = LA.eig(A_np)
    Lambda_min = np.min(w)
    if Lambda_min <= 0:
        A_np = A_np - Lambda_min * np.eye(D)
    

    return A_np, t.numpy()
    
def trust_region_solver(A,t):
    #   Solve the formulation
    #   Maximize    zT*A*z + zT*t
    #   Subject to  norm(z,2)<=1, norm(z,0)<= d
    #   Input:
    #       A: data matrix, dim: D*D
    #       t: vector,      dim: D*1
    #  Output:
    #       z: estimated optimal solution
    
    z0 = np.linalg.solve(A, -0.5*t)
    D = len(t)
    

    M0 = np.block([
        [-np.eye(D),   -2*A],
        [-2*A,            -t @ t.T]
        ])
    M1 = np.block([
    [np.zeros([D,D]),   np.eye(D)],
    [np.eye(D),         np.zeros([D,D])]
    ])
    w, v = eigs(M0, M=-M1, k=2*D, which="LM")
    Lambda = np.real(w[0]).reshape([-1,])                # optimal dual variable
    x = np.real(v[:D,0]).reshape([-1,1])                 # optimal primal variable
    sign_x = np.float(np.sum(t[:,0] * v[D:,0]) > 0) * 2 -1
    x = x / np.linalg.norm(x)
    z1 = x * sign_x

    obj0 = t.T@z0 + z0.T @ (A@z0)
    obj1 = t.T@z1 + z1.T @ (A@z1)
    if obj1 >= obj0:
        return z1
    else:
        return z0
    
def MIQP_app_solver(A,t,d):
    #   Solve the formulation
    #   Maximize    zT*A*z + zT*t
    #   Subject to  norm(z,2)=1, norm(z,0)<= d
    #   Input:
    #       A: data matrix, dim: D*D
    #       t: vector,      dim: D*1
    #       d: sparsity budget
    #  Output:
    #       z: estimated optimal solution
    D = len(t)
    obj_hist = []
    sol_hist = []

    for i in range(D):
        A_i = A[:,i]
        A_i_idx = np.argsort(-np.abs(A_i))

        v_i = np.zeros(D)
        for j in range(d):
            v_i[A_i_idx[j]] = A_i[A_i_idx[j]]
        v_i = v_i / np.linalg.norm(v_i)
        v_i = v_i.reshape([-1,1])
        obj_v_i = t.T@v_i + v_i.T @ (A@v_i)

        obj_hist.append(obj_v_i[0,0])
        sol_hist.append(v_i)
    
        e_i = np.zeros(D).reshape([-1,1])
        e_i[i,0] = 1.0
        obj_e_i = t.T@e_i + e_i.T @ (A@e_i)

        obj_hist.append(obj_e_i[0,0])
        sol_hist.append(e_i)
    
    z_trust = trust_region_solver(A,t)
    z_trust_idx = np.argsort(-np.abs(z_trust))
    z_trust_new = np.zeros(D)
    for j in range(d):
        z_trust_new[z_trust_idx[j]] = z_trust[z_trust_idx[j]]
    z_trust_new = z_trust_new / np.linalg.norm(z_trust_new)
    z_trust_new = z_trust_new.reshape([-1,1])
    obj_z_trust_new = t.T@z_trust_new + z_trust_new.T @ (A@z_trust_new)
    obj_hist.append(obj_z_trust_new[0,0])
    sol_hist.append(z_trust_new)

    obj_hist = np.array(obj_hist)
    idx_opt = np.argmax(obj_hist)

    return sol_hist[idx_opt], obj_hist[idx_opt]
    

        



    

    


    # residual_1 = np.abs(np.linalg.norm(x)-1)
    # residual_2 = np.linalg.norm(- 2*(A @ x) + Lambda * x - t)
    # residual_3 = is_pos_def(-A + Lambda * np.eye(D))
    # print([residual_1, residual_2, residual_3])


def Linear_MMD_training(a, d):
    # obtain optimal projection vector from linear MMD
    #   Input:
    #       a: input coefficient, dim: D
    #       d: proejcted dimension
    #  Output:
    #       z: proejction vector s.t. ||z||_2 = 1, ||z||_0 = d, dim: D
    index_a = np.argsort(-a)
    D = len(a)

    z = np.zeros(D)
    z[index_a <= d-1] = a[index_a <= d-1]
    z = z / np.linalg.norm(z)
    
    return z

def Linear_MMD_testing_revision(X_Te, Y_Te, z, num_perm = 100):
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
        tau = np.median(Data_xy)

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

