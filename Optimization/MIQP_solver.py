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

d = 3

dtype = torch.float
device = torch.device("cpu")



def trust_region_solver(A,t):
    #   Solve the formulation
    #   Maximize    zT*A*z + zT*t
    #   Subject to  norm(z,2)<=1, norm(z,0)<= d
    #   Input:
    #       A: data matrix, dim: D*D
    #       t: vector,      dim: D*1
    #  Output:
    #       z: estimated optimal solution
    
    #z0 = np.linalg.solve(A, -0.5*t)
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

    # obj0 = t.T@z0 + z0.T @ (A@z0)
    # obj1 = t.T@z1 + z1.T @ (A@z1)
    # if obj1 >= obj0:
    #     return z1
    # else:
    #     return z0
    return z1

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
    #   obj_z: objective value of estimated solution

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
    
