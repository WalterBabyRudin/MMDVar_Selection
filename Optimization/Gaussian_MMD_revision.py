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

def is_psd(mat):
    return bool((mat == mat.T).all() and (torch.linalg.eigvals(mat).real>=0).all())

def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)

def MatConvert(x, device, dtype):
    """convert the numpy to a torch tensor."""
    x = torch.from_numpy(x).to(device, dtype)
    return x

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

def GMMD_training_revision(X, Y, sigma, Lambda, d, z0, tau0, num_epoch):
    '''
        Input:
        (X,Y): training data point
        sigma: bandwidth for the Gaussian kernel
       Lambda: variance regularization
            d: number of chosen variables
           z0: initial guess of optimal solution 
    '''
    nX, D = np.shape(X)
    z = torch.tensor(z0, requires_grad=True)
    X = torch.tensor(X, requires_grad=False)
    Y = torch.tensor(Y, requires_grad=False)

    for iter in range(num_epoch):

        X_z = X * z.T
        Y_z = Y * z.T
        Data_xx = Pdist2(X_z, X_z)#(X_z.reshape([-1,1]) - X_z.reshape([1,-1]))**2
        Data_yy = Pdist2(Y_z, Y_z)
        Data_xy = Pdist2(X_z, Y_z)
        tau = torch.median(Pdist2(X, Y)) * sigma
        Kx = torch.exp(-Data_xx/(2*tau))
        Ky = torch.exp(-Data_yy/(2*tau))
        Kxy = torch.exp(-Data_xy/(2*tau))

    

        S_MMD, Var_MMD, _ = h1_mean_var_gram(Kx, Ky, Kxy, is_var_computed=True)
        Obj_MMD = S_MMD - Lambda * Var_MMD

        Grad_MMD = torch.autograd.grad(Obj_MMD, z, retain_graph=True, create_graph=True)
        Hessian_MMD = torch.zeros([D,D])
        for i in range(D):
            Grad_Grad_i = torch.autograd.grad(Grad_MMD[0][i], z, create_graph=True)[0]
            Hessian_MMD[i,:] = Grad_Grad_i.reshape([-1,])

        MIQP_coeff_A = 0.5 * Hessian_MMD.detach().numpy() - 1/(2*tau0) * np.eye(D)
        z_numpy = z.detach().numpy()
        MIQP_coeff_t = -2 * MIQP_coeff_A@z_numpy + Grad_MMD[0].detach().numpy()
        MIQP_coeff_A = (MIQP_coeff_A + MIQP_coeff_A.T)/2
        w, v = LA.eig(MIQP_coeff_A)
        Lambda_min = np.min(w)
        if Lambda_min <= 0:
            MIQP_coeff_A = MIQP_coeff_A - Lambda_min * np.eye(D)


        z_new,obj_z_new = MIQP_solver.MIQP_app_solver(MIQP_coeff_A, MIQP_coeff_t, d)
        obj_z_old = z_numpy.T@(MIQP_coeff_A@z_numpy) + MIQP_coeff_t.T@z_numpy
        if iter >= 1:
            if obj_z_new > obj_z_old[0,0]:
                z = torch.tensor(z_new, requires_grad=True)
            else:
                break
        else:
            z = torch.tensor(z_new, requires_grad=True)
    return z.detach().numpy(), obj_z_new.item()


def GMMD_testing_revision(X_Te, Y_Te, z, sigma, num_perm = 100):
    #print(z)
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

    z = torch.tensor(z, requires_grad=False)
    X_Te = torch.tensor(X_Te, requires_grad=False)
    Y_Te = torch.tensor(Y_Te, requires_grad=False)


    #support_z = z.nonzero().squeeze()[:,0]
    #print(support_z)
    X_z = X_Te * z.T
    #X_z = X_z[:, support_z]
    Y_z = Y_Te * z.T
    #Y_z = Y_z[:, support_z]
    Data_xx = Pdist2(X_z, X_z)#(X_z.reshape([-1,1]) - X_z.reshape([1,-1]))**2
    Data_yy = Pdist2(Y_z, Y_z)
    Data_xy = Pdist2(X_z, Y_z)
    tau = torch.median(Pdist2(X_Te, Y_Te)) * sigma
    Kx = torch.exp(-Data_xx/(2*tau))
    Ky = torch.exp(-Data_yy/(2*tau))
    Kxy = torch.exp(-Data_xy/(2*tau))



    Kxxy = torch.cat((Kx,Kxy),1)
    Kyxy = torch.cat((Kxy.transpose(0,1),Ky),1)
    Kxyxy = torch.cat((Kxxy,Kyxy),0)

    _, p_val, _ = mmd2_permutations(Kxyxy, nX_Te, permutations=num_perm)

    if p_val <= 0.05:
        decision = 1
    else:
        decision = 0
    return p_val, decision





