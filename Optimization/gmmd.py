import numpy as np
import cvxpy as cp
from numpy.lib import utils
import torch
import scipy.spatial
import time
from scipy.stats import wishart
from sklearn import metrics
from utils import proj_vector, MatConvert, mmd2_permutations
d = 3

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

def Gaussian_MMD_testing(X_Te, Y_Te, z, num_perm = 500):
    #  perform permutation testing on testing samples
    #    Input:
    #     X_Te: data from mu, dim: n*D
    #     Y_Te: data from nu, dim: n*D
    #        z: projection vector
    #Output:
    # decision: reject H0 or not
    nX_Te, D = np.shape(X_Te)
    nY_Te, _ = np.shape(Y_Te)

    X_Te_proj, Y_Te_proj = proj_vector(z, X_Te), proj_vector(z, Y_Te)
    X_Te_proj_torch = torch.tensor(X_Te_proj)
    Y_Te_proj_torch = torch.tensor(Y_Te_proj)
    #print(X_Te_proj)
    #X_Te_proj_torch, Y_Te_proj_torch = MatConvert(np.float32(X_Te_proj), dtype, device), MatConvert(np.float32(Y_Te_proj), dtype, device)
    Dxx = Pdist2(X_Te_proj_torch,X_Te_proj_torch)
    Dyy = Pdist2(Y_Te_proj_torch,Y_Te_proj_torch)
    Dxy = Pdist2(X_Te_proj_torch,Y_Te_proj_torch)

    sigma = torch.median(Dxy)
    Kx = torch.exp(-Dxx / (2*sigma))
    Ky = torch.exp(-Dyy / (2*sigma))
    Kxy = torch.exp(-Dxy / (2*sigma))
    Kxxy = torch.cat((Kx,Kxy),1)
    Kyxy = torch.cat((Kxy.transpose(0,1),Ky),1)
    Kxyxy = torch.cat((Kxxy,Kyxy),0)

    _, p_val, _ = mmd2_permutations(Kxyxy, nX_Te, permutations=200)

    if p_val <= 0.05:
        decision = 1
    else:
        decision = 0
    return p_val, decision
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

def Gaussian_MMD_training(X_Tr, Y_Tr, Z0, d0, gamma = 5e-3, sigma = 3, Lambda = 1e-1, 
                    max_inner_iter = 50, max_outer_iter = 100):
    # obtain optimal projection vector from Gaussian MMD
    #   Input:
    #    X_Tr: input data, dim: n*D
    #    Y_Tr: input data, dim: m*D
    #      Z0: initial guess of projection
    #  Output:
    #       z: proejction vector s.t. ||z||_2 = 1, ||z||_0 = d, dim: D

    nX_Tr, D = X_Tr.shape
    nY_Tr, _ = Y_Tr.shape
    Z = Z0.copy()

    for outer_iter in range(max_outer_iter):
        # formulate objective estimator of MMD [For only xx and yy part]
        Z_torch = torch.tensor(Z, dtype=torch.float, requires_grad=True)
        D_xx = torch.matmul(torch.matmul(X_Tr, Z_torch), X_Tr.T)
        D_yy = torch.matmul(torch.matmul(Y_Tr, Z_torch), Y_Tr.T)
        D_xy = torch.matmul(torch.matmul(X_Tr, Z_torch), Y_Tr.T)

        # sigma = np.sqrt(torch.median(D_xy**2).item())
        # print(sigma)

        # MMD_xx = torch.exp(-D_xx/(2*sigma))
        # MMD_yy = torch.exp(-D_yy/(2*sigma))
        S_MMD, Var_MMD, _ = h1_mean_var_gram(D_xx, D_yy, D_xy, True)
        #Obj_MMD = S_MMD - Var_MMD
        #Obj_MMD.backward()
        mmd_value_temp = -1 * (S_MMD + 10 ** (-8))
        mmd_std_temp = torch.sqrt(Var_MMD+10**(-8))
        #print(mmd_std_temp)
        STAT_u = torch.div(mmd_value_temp, mmd_std_temp)
        STAT_u.backward(retain_graph=True)

        
        grad_hatF_Z = np.array(Z_torch.grad.cpu().numpy())

        Y = Z * np.exp(- gamma * grad_hatF_Z)
        Z = Y / np.trace(Y)
        Z = (Z + Z.T)/2
        
        
    w, v = np.linalg.eig(Z)
    z = np.real(v[:,0])# + 1e-8
    index_z = np.argsort(-z**2)
    z[index_z > d0] = 0
    z = z / np.linalg.norm(z)

    return z









