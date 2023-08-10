import numpy as np
import cvxpy as cp
from numpy.lib import utils
import torch
import scipy.spatial
import time
from scipy.stats import wishart
from sklearn import metrics

d = 3

dtype = torch.float
device = torch.device("cpu")

def MatConvert(x, device, dtype):
    """convert the numpy to a torch tensor."""
    x = torch.from_numpy(x).to(device, dtype)
    return x

def Pdist2(x, y):
    """compute the paired distance between x and y."""
    Pdist = scipy.spatial.distance.cdist(x,y,'sqeuclidean')

    # # x_norm = (x ** 2).sum(1).view(-1, 1)
    # # if y is not None:
    # #     y_norm = (y ** 2).sum(1).view(1, -1)
    # # else:
    # #     y = x
    # #     y_norm = x_norm.view(1, -1)
    # # Pdist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
    # Pdist[Pdist<0]=0
    return Pdist

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

def sampling_sphere(d, n=1):
    x_hist = []
    for i in range(n):
        x = np.random.normal(size = (d,))
        x = x*0.5 / np.linalg.norm(x)
        x_hist.append(x)
    return x_hist

def sampling_wishart(d, n=1):
    cov_hist = []
    for i in range(n):
        
        cov = np.zeros([3,3])
        for j in range(d):
            x = np.random.normal(size = (d,1))
            cov = cov + x @ x.T
        
        cov_hist.append(cov)
    return cov_hist

def parameter_generation(d, L):
    # generate target mean vectors and covariance matrices
    #   Input:
    #       d: size within a single block
    #       L: number of blocks
    #  Output:
    #       mean_mu_hist: list of mean vectors from mu
    #       mean_nu_hist: list of mean vectors from nu
    #        cov_mu_hist: list of covariance matrics from mu
    #        cov_nu_hist: list of covariance matrics from nu

    mean_mu_hist = sampling_sphere(d, L)
    cov_mu_hist = sampling_wishart(d, L)

    mean_nu_hist = mean_mu_hist.copy()
    x_nu = np.random.normal(size = (d,))
    x_nu = x_nu*0.5 / np.linalg.norm(x_nu)
    mean_nu_hist[0] = x_nu #= mean_nu_hist[0] * 0


    cov_nu_hist = cov_mu_hist.copy()
    cov_nu = np.zeros([3,3])
    for j in range(d):
        x = np.random.normal(size = (d,1))
        cov_nu = cov_nu + x @ x.T
    cov_nu_hist[0] = cov_nu
    
    return mean_mu_hist, mean_nu_hist, cov_mu_hist, cov_nu_hist

def data_generation(mean_mu_hist, mean_nu_hist, cov_mu_hist, cov_nu_hist, n):
    # generate data points from mu and nu with sample size n
    L = len(mean_mu_hist)
    d = len(mean_mu_hist[0])
    D = d*L

    X = np.zeros([n,0])
    Y = np.zeros([n,0])
    for ell in range(L):
        X_ell = np.random.multivariate_normal(mean_mu_hist[ell], cov_mu_hist[ell], size=n)
        X = np.concatenate((X, X_ell), axis=1)
        
        Y_ell = np.random.multivariate_normal(mean_nu_hist[ell], cov_nu_hist[ell], size=n)
        Y = np.concatenate((Y, Y_ell), axis=1)
    
    return X, Y
