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
import Optimization.MIQP_solver as MIQP_solver
from numpy import linalg as LA

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

def Quad_MMD_training_revision(X, Y, c, Lambda, d, z0, tau, num_epoch=5, sigma=1):
    '''
        Input:
        (X,Y): training data point
            c: coefficient for the quadratic kernel
       Lambda: variance regularization
            d: number of chosen variables
           z0: initial guess of optimal solution 
    '''
    nX, D = np.shape(X)

    
    Kx_hist = []
    Ky_hist = []
    Kxy_hist= []

    for i in range(D):
        X_i = X[:,i]
        Y_i = Y[:,i]
        Data_xx = (X_i.reshape([-1,1]) - X_i.reshape([1,-1]))**2
        Data_yy = (Y_i.reshape([-1,1]) - Y_i.reshape([1,-1]))**2
        Data_xy = (X_i.reshape([-1,1]) - Y_i.reshape([1,-1]))**2
        tau = np.median(Data_xy)*sigma
        Kx = MatConvert(np.exp(-Data_xx/(2*tau)), device, dtype)
        Ky = MatConvert(np.exp(-Data_yy/(2*tau)), device, dtype)
        Kxy = MatConvert(np.exp(-Data_xy/(2*tau)), device, dtype)
        Kx_hist.append(Kx)
        Ky_hist.append(Ky)
        Kxy_hist.append(Kxy)
    
    Kx_hist = torch.stack(Kx_hist, dim=2)
    Ky_hist = torch.stack(Ky_hist, dim=2)
    Kxy_hist = torch.stack(Kxy_hist, dim=2)

    z = torch.tensor(z0, requires_grad=True)
    #tau = 1


    for iter in range(num_epoch):
        Kx_z = (torch.sum(Kx_hist*z.reshape([1,1,-1]), dim=2)+c)**2
        Ky_z = (torch.sum(Ky_hist*z.reshape([1,1,-1]), dim=2)+c)**2
        Kxy_z = (torch.sum(Kxy_hist*z.reshape([1,1,-1]), dim=2)+c)**2

        S_MMD, Var_MMD, _ = h1_mean_var_gram(Kx_z, Ky_z, Kxy_z, is_var_computed=True)
        Obj_MMD = S_MMD - Lambda * Var_MMD
        
        Grad_MMD = torch.autograd.grad(Obj_MMD, z, retain_graph=True, create_graph=True)
        
        Hessian_MMD = torch.zeros([D,D])
        for i in range(D):
            Grad_Grad_i = torch.autograd.grad(Grad_MMD[0][i], z, create_graph=True)[0]

            Hessian_MMD[i,:] = Grad_Grad_i.reshape([-1,])

        
        MIQP_coeff_A = 0.5 * Hessian_MMD.detach().clone().numpy() - 1/(2*tau) * np.eye(D)
        z_numpy = z.detach().numpy()
        MIQP_coeff_t = -2 * MIQP_coeff_A@z_numpy + Grad_MMD[0].detach().numpy()



        MIQP_coeff_A = (MIQP_coeff_A + MIQP_coeff_A.T)/2
        w, v = LA.eig(MIQP_coeff_A)
        Lambda_min = np.min(w)
        if Lambda_min <= 0:
            MIQP_coeff_A = MIQP_coeff_A - Lambda_min * np.eye(D)


        z_new,obj_z_new = MIQP_solver.MIQP_app_solver(MIQP_coeff_A, MIQP_coeff_t, d)

        obj_z_old = z_numpy.T@(MIQP_coeff_A@z_numpy) + MIQP_coeff_t.T@z_numpy
        # print(z_new)
        # print([obj_z_old[0,0], obj_z_new])
        
        if iter >= 1:
            if obj_z_new > obj_z_old[0,0]:
                z = torch.tensor(z_new, requires_grad=True)
            else:
                break
        else:
            z = torch.tensor(z_new, requires_grad=True)
    return z.detach().numpy(), obj_z_new.item()



def Quadratic_MMD_testing_revision(X_Te, Y_Te, z, c, num_perm = 100, sigma=1):
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

    Kx_hist = []
    Ky_hist = []
    Kxy_hist= []

    for i in range(D):
        X_i = X_Te[:,i]
        Y_i = Y_Te[:,i]
        Data_xx = (X_i.reshape([-1,1]) - X_i.reshape([1,-1]))**2
        Data_yy = (Y_i.reshape([-1,1]) - Y_i.reshape([1,-1]))**2
        Data_xy = (X_i.reshape([-1,1]) - Y_i.reshape([1,-1]))**2
        tau = np.median(Data_xy)*sigma
        Kx = MatConvert(np.exp(-Data_xx/(2*tau)), device, dtype)
        Ky = MatConvert(np.exp(-Data_yy/(2*tau)), device, dtype)
        Kxy = MatConvert(np.exp(-Data_xy/(2*tau)), device, dtype)
        Kx_hist.append(Kx)
        Ky_hist.append(Ky)
        Kxy_hist.append(Kxy)
    
    Kx_hist = torch.stack(Kx_hist, dim=2)
    Ky_hist = torch.stack(Ky_hist, dim=2)
    Kxy_hist = torch.stack(Kxy_hist, dim=2)

    Kx_z = (torch.sum(Kx_hist*z.reshape([1,1,-1]), dim=2)+c)**2
    Ky_z = (torch.sum(Ky_hist*z.reshape([1,1,-1]), dim=2)+c)**2
    Kxy_z = (torch.sum(Kxy_hist*z.reshape([1,1,-1]), dim=2)+c)**2


    Kxxy = torch.cat((Kx_z,Kxy_z),1)
    Kyxy = torch.cat((Kxy_z.transpose(0,1),Ky_z),1)
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
    return est.item(), p_val.item(), rest

def h1_mean_var_gram(Kx, Ky, Kxy, is_var_computed, use_1sample_U=True):
    """compute value of MMD and std of MMD using kernel matrix."""
    Kxxy = torch.cat((Kx,Kxy),1)
    Kyxy = torch.cat((Kxy.transpose(0,1),Ky),1)
    Kxyxy = torch.cat((Kxxy,Kyxy),0)
    nx = Kx.shape[0]
    ny = Ky.shape[0]
    is_unbiased = True
    if is_unbiased:
        xx = torch.div((torch.sum(Kx) - torch.sum(torch.diag(Kx))), (nx * (nx - 1)))
        yy = torch.div((torch.sum(Ky) - torch.sum(torch.diag(Ky))), (ny * (ny - 1)))
        # one-sample U-statistic.
        if use_1sample_U:
            xy = torch.div((torch.sum(Kxy) - torch.sum(torch.diag(Kxy))), (nx * (ny - 1)))
        else:
            xy = torch.div(torch.sum(Kxy), (nx * ny))
        mmd2 = xx - 2 * xy + yy
    else:
        xx = torch.div((torch.sum(Kx)), (nx * nx))
        yy = torch.div((torch.sum(Ky)), (ny * ny))
        # one-sample U-statistic.
        if use_1sample_U:
            xy = torch.div((torch.sum(Kxy)), (nx * ny))
        else:
            xy = torch.div(torch.sum(Kxy), (nx * ny))
        mmd2 = xx - 2 * xy + yy
    if not is_var_computed:
        return mmd2, None

    hh = Kx+Ky-Kxy-Kxy.transpose(0,1)
    V1 = torch.dot(hh.sum(1)/ny,hh.sum(1)/ny) / ny
    V2 = (hh).sum() / (nx) / nx
    varEst = 4*(V1 - V2**2)
    if  varEst == 0.0:
        print('error!!'+str(V1))
    return mmd2, varEst, Kxyxy