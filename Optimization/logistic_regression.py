import numpy as np
import cvxpy as cp
from numpy.lib import utils
import torch
import scipy.spatial
import time
from scipy.stats import wishart
from sklearn import metrics

def get_cuts_logistic_training(q, X_Tr, Y_Tr, gamma):
    q = q.reshape([-1,1])
    n, D = np.shape(X_Tr)
    m, _ = np.shape(Y_Tr)
    Data_Tr = np.concatenate((X_Tr, Y_Tr), axis=0)
    Data_Tr_2 = Data_Tr @ Data_Tr.T

    N_Tr = n+m

    Labels = np.concatenate((np.ones([n,1]), -np.ones([m,1])), axis=0)
    alpha = cp.Variable((N_Tr, 1))

    #print(Labels.shape)

    obj = cp.sum(cp.entr(-cp.multiply(Labels, alpha)) + cp.entr(1 + cp.multiply(Labels, alpha)))
    obj = obj - gamma/2 * cp.quad_form(alpha, Data_Tr_2)
    constraints = [cp.sum(alpha) == 0, cp.multiply(Labels,alpha)<=0, cp.multiply(Labels,alpha)>=-1]
    prob = cp.Problem(cp.Maximize(obj),
                        constraints)
    prob.solve(solver=cp.MOSEK, verbose=False)

    
    fq = prob.value
    alpha_val = alpha.value

    Data_Tr_alpha = Data_Tr.T @ alpha_val
    gq = -gamma/2 * Data_Tr_alpha**2
    return fq, gq

def logistic_classifier_support(X_Tr, Y_Tr, gamma=1, d_0=3, maxiter=10):
    """
        Input:
         X_Tr: n*D data matrix
         Y_Tr: m*D data matrix
       Output:
        (w,b): classification weights
    """

    n, D = np.shape(X_Tr)
    m, _ = np.shape(Y_Tr)

    q = np.ones((D,1)) # initial guess
    eta = 0

    fq_hist = []
    gq_hist = []
    eta_val = 0
    residual_hist = []
    for iter in range(maxiter):
        fq, gq = get_cuts_logistic_training(q, X_Tr, Y_Tr, gamma)
        residual = np.abs(fq - eta_val).reshape([-1,])

        if residual <= .1:
            break
        residual_hist.append(residual)


        fq_hist.append(fq)
        gq_hist.append(gq)

        # --------------------------------------------------------------------------------
        # ------------------------ Solve Cutting Plane Problem   -------------------------
        # --------------------------------------------------------------------------------
        q_new = cp.Variable((D,1), integer=True)
        eta = cp.Variable((1,))
        constraints = [cp.sum(q_new) <= d_0, q_new >= 0, q_new <= 1]
        for j in range(len(fq_hist)):
            constraints += [eta >= fq_hist[j] + cp.matmul(gq_hist[j].T, q_new - q)]
        
        prob = cp.Problem(cp.Minimize(eta),
                        constraints)
        prob.solve(solver=cp.GUROBI, verbose=False)
        
        q = q_new.value
        eta_val = eta.value
        #print("Iter: ", iter, "q: ", q[:3], "residual: ", residual)

    # print(residual_hist)
    # print(q[:3])
    return q

def logistic_classifier(X_Tr, Y_Tr, q, gamma=1):
    """
        Get classifer based on support q
        Input:
         X_Tr: n*D data matrix
         Y_Tr: m*D data matrix
        gamma: regularization value
            q: support vector
       Output:
        (w,b): classification weights
    """
    q = q.reshape([-1,1])
    n, D = np.shape(X_Tr)
    m, _ = np.shape(Y_Tr)
    N_Tr = n+m

    Data_Tr = np.concatenate((X_Tr, Y_Tr), axis=0)
    Labels = np.concatenate((np.ones([n,1]), -np.ones([m,1])), axis=0)

    w = cp.Variable((D,1))
    b = cp.Variable((1,))

    obj = cp.sum(cp.logistic(- cp.multiply(Labels, Data_Tr @ w + b))) + 1/(2*gamma) * cp.sum_squares(w)

    prob = cp.Problem(cp.Minimize(obj))
    prob.solve(solver = cp.MOSEK, verbose = False)
    return w.value, b.value
    
def logistic_classifier_testing(X_Te, Y_Te, w, b, num_permute=500, alpha=0.05):
    #  perform permutation testing on testing samples
    #    Input:
    #     X_Te: data from mu, dim: n*D
    #     Y_Te: data from nu, dim: n*D
    #    (w,b): classification weights

    f_X_Te = X_Te @ w + b
    f_Y_Te = Y_Te @ w + b


    nX_Te, _ = np.shape(f_X_Te)
    f_XY_Te = np.concatenate((f_X_Te, f_Y_Te), axis=0)
    n_Te, _ = np.shape(f_XY_Te)


    eta = np.mean(f_X_Te) - np.mean(f_Y_Te)


    eta_hist = np.float32(np.zeros(num_permute))

    for iboot in range(num_permute):
        tmp = np.random.permutation(n_Te)
        idx1_perm, idx2_perm = tmp[0:nX_Te], tmp[nX_Te:n_Te]
        f_X_perm, f_Y_perm = f_XY_Te[idx1_perm], f_XY_Te[idx2_perm]

        eta_hist[iboot] = np.mean(f_X_perm) - np.mean(f_Y_perm)
    
    t_alpha = np.quantile(eta_hist, 1-alpha)

    if eta > t_alpha:
        decision = 1
    else:
        decision = 0
    return decision

