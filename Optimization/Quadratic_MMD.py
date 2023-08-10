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

def trust_region_solver(A,t):
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

def MatConvert(x, device, dtype):
    """convert the numpy to a torch tensor."""
    x = torch.from_numpy(x).to(device, dtype)
    return x

def quad_product_compute(X, Y, c):
    """
    Compute sum_{i,j} (X_i[:]Y_j[:]) @ (X_i[:]Y_j[:]).T
        Input:
            X: n*D data matrix
            Y: m*D data matrix
            c: kernel bandwidth
         simi: if data X == Y
       Output:
         A_XY: D*D matrix
         t_XY: D-dim array
    """
    n, D = np.shape(X)
    m, _ = np.shape(Y)
    A_XY = np.zeros([D,D])
    t_XY = np.zeros([D,1])
    
    YTY = Y.T @ Y
    for i in range(n):
        X_i = X[i, :].reshape([-1,1])
        A_XY = A_XY + YTY * (X_i@X_i.T)
        # if simi == True: 
        #     X_i2 = X_i**2
        #     A_XY = A_XY - X_i2@X_i2.T

        X_iY = np.sum(X_i.T * Y, 0).reshape([-1, 1])
        t_XY = t_XY + X_iY

    # if simi == True:
    #     t_XY = t_XY - np.sum(X**2,0).reshape([-1,1])

    #     A_XY = A_XY / (n * (n-1))
    #     t_XY = t_XY / (n * (n-1))
    # else:
    
    A_XY = A_XY / (n * m)
    t_XY = t_XY / (n * m)
    
    return A_XY, t_XY*2*c

def quad_MMD_training(A, t, d_0=3):
    # obtain optimal projection vector from quadratic MMD (approximation algorithm)
    #   Input:
    #       A: input matrix, dim: D*D
    #       t: input vector, dim: D
    #      d0: proejcted dimension
    #  Output:
    #       z: proejction vector s.t. ||z||_2 = 1, ||z||_0 = d, dim: D

    D = len(t)
    tilde_A = np.block([
    [np.zeros((1,1)),   t.reshape((1,-1))],
    [t.reshape((-1,1)), A               ]
    ])

    # --------------------------------------------------------------------------------
    # ------------------------- Solve SDP Relaxation ---------------------------------
    # --------------------------------------------------------------------------------
    Z = cp.Variable((D+1,D+1), symmetric=True)
    q = cp.Variable((D, 1))
    constraints = [Z >> 0, Z[0,0] == 1, cp.trace(Z) == 2, q <= 1, q >= 0, cp.sum(q) <= d_0]#, cp.norm(Z,1) <= d]
    constraints += [Z[i+1,i+1] <= q[i] for i in range(D)]
    constraints += [Z[i+1,i+1] >= -q[i] for i in range(D)]
    for i in range(D):
        constraints += [cp.norm(Z[1:, i+1], 1) <= d_0]
        constraints += [cp.quad_over_lin(Z[i+1, 1:], Z[i+1,i+1]) <= q[i]]

        for j in range(i+1, D):
            constraints += [Z[i+1,j+1] >= -0.5 * q[i]]
            constraints += [Z[i+1,j+1] <=  0.5 * q[i]]

    prob = cp.Problem(cp.Maximize(cp.trace(Z @ tilde_A)),
                    constraints)

    prob.solve(solver=cp.MOSEK)
    q_val = q.value.reshape([-1,])
    index_q = np.argsort(-q_val)

    q = np.zeros(D)
    q[index_q <= d_0 - 1] = 1

    # --------------------------------------------------------------------------------
    # ------------------------- Get Projection Vector --------------------------------
    # --------------------------------------------------------------------------------
    # print(q)
    # supp_q_zero = np.where(q == 0)
    # supp_q_zero = supp_q_zero[0]
    # Z = cp.Variable((D+1,D+1), symmetric=True)
    # constraints_1 = [Z >> 0, Z[0,0] == 1, cp.trace(Z) == 2, Z[supp_q_zero+1, supp_q_zero+1] == 0]
    # prob_1 = cp.Problem(cp.Maximize(cp.trace(Z @ tilde_A)),
    #                 constraints_1)
    # prob_1.solve(solver=cp.SCS, verbose=False)
    # w, v = np.linalg.eig(Z.value[1:, 1:])
    # z = v[:,0]
    
    
    # z[supp_q_zero] = 0
    # z = z / np.linalg.norm(z)
    z = quad_MMD_projection_vec_compute(q, tilde_A)
    return z

def quad_MMD_projection_vec_compute(q, tilde_A):
    # get projection vector based on q
    #   Input: 
    #       q: support vector
    # tilde_A: coefficient matrix
    #  Output:
    #       z: proejction vector s.t. ||z||_2 = 1, ||z||_0 = d, dim: D
    q = q.reshape([-1,1])
    D = len(q)
    supp_q_zero = np.where(q == 0)
    supp_q_zero = supp_q_zero[0]
    Z = cp.Variable((D+1,D+1), symmetric=True)
    constraints_1 = [Z >> 0, Z[0,0] == 1, cp.trace(Z) == 2, Z[supp_q_zero+1, supp_q_zero+1] == 0]
    prob_1 = cp.Problem(cp.Maximize(cp.trace(Z @ tilde_A)),
                    constraints_1)
    prob_1.solve(solver=cp.SCS, verbose=False)
    _, v = np.linalg.eig(Z.value[1:, 1:])
    z = v[:,0]
    
    
    z[supp_q_zero] = 0
    z = z / np.linalg.norm(z)
    return z

def quadratic_MMD_testing(X_Te, Y_Te, z, c, num_perm = 500):
    #  perform permutation testing on testing samples
    #    Input:
    #     X_Te: data from mu, dim: n*D
    #     Y_Te: data from nu, dim: n*D
    #        z: projection vector
    #        c: kernel bandwidth
    #Output:
    # decision: reject H0 or not
    nX_Te, D = np.shape(X_Te)
    nY_Te, _ = np.shape(Y_Te)

    z_X_Te = z.reshape([1,-1]) * X_Te
    z_Y_Te = z.reshape([1,-1]) * Y_Te


    Kx =  MatConvert(metrics.pairwise.polynomial_kernel(z_X_Te, X_Te, 2, 1, c), device, dtype)
    Ky =  MatConvert(metrics.pairwise.polynomial_kernel(z_Y_Te, Y_Te, 2, 1, c), device, dtype)
    Kxy = MatConvert(metrics.pairwise.polynomial_kernel(z_X_Te, Y_Te, 2, 1, c), device, dtype)

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
    return est.item(), p_val.item(), rest

def get_cuts_MMD_training_exact(q, tilde_A, d_0):
    # get objective and subgradient estimator of SDP
    #   Input:
    #       q: input parameter
    # tilde_A: coefficient matrix
    #     d_0: projected dimension
    #  Output:
    #      fq: objective estimator
    #      gq: subgradient estimator
    D = len(q)
    # --------------------------------------------------------------------------------
    # ------------------------ Solve Outer Approximation SDP -------------------------
    # --------------------------------------------------------------------------------
    Lambda = cp.Variable((2,))
    alpha_u = cp.Variable((D, D))
    alpha_l = cp.Variable((D, D))
    beta = cp.Variable((D, 1))

    constraints = [alpha_u>=0, alpha_l>=0, beta >=0]
    constraints += [cp.bmat(
                    [[cp.reshape(Lambda[0], [1,1]),       np.zeros((1,D))], 
                    [np.zeros((D,1)),                     Lambda[1] * np.eye(D) + alpha_u - alpha_l]]
                    ) - tilde_A >> 0]
    obj = Lambda[0] + 2 * Lambda[1] + np.sqrt(d_0) * cp.matmul(q.T, beta)
    
    for i in range(D):
        obj += q[i,0] * cp.pos(alpha_u[i,i] + alpha_l[i,i] - beta[i])
        for j in range(D):
            if j != i:
                obj += q[i,0] * 0.5 * cp.pos(alpha_u[i,j] + alpha_u[i,j] - beta[i])


    prob = cp.Problem(cp.Minimize(obj),
                    constraints)
    prob.solve(solver=cp.MOSEK, verbose=False)

    fq = obj.value
    alpha_u_val = alpha_u.value
    alpha_l_val = alpha_l.value
    beta_val = beta.value
    gq = np.zeros([D,1])
    for i in range(D):
        
        gq_i = np.sqrt(d_0) * beta_val[i] + np.clip(alpha_u_val[i,i] + alpha_l_val[i,i] - beta_val[i], 0, None)
        for j in range(D):
            if j != i:
                gq_i += 0.5*np.clip(alpha_u_val[i,j] + alpha_l_val[i,j] - beta_val[i], 0, None)
        
        gq[i] = gq_i
    return fq.reshape([-1,]), gq

def quad_MMD_training_exact(A, t, d_0=3, maxiter = 10):
    # obtain optimal projection vector from quadratic MMD by solving exact QP
    #   Input:
    #       A: input matrix, dim: D*D
    #       t: input vector, dim: D
    #      d0: proejcted dimension
    #  Output:
    #       z: proejction vector s.t. ||z||_2 = 1, ||z||_0 = d, dim: D
    D = len(t)
    tilde_A = np.block([
    [np.zeros((1,1)),   t.reshape((1,-1))],
    [t.reshape((-1,1)), A               ]
    ])

    q = np.ones((D,1)) # initial guess
    # q[0] = 1
    # q[2] = 1

    fq_hist = []
    gq_hist = []
    theta_val = 0
    residual_hist = []
    for iter in range(maxiter):
        fq, gq = get_cuts_MMD_training_exact(q, tilde_A, d_0)
        residual = np.abs(fq - theta_val)
        residual_hist.append(residual)
        if residual <= 0.1:
            break
        fq_hist.append(fq)
        gq_hist.append(gq)

        # --------------------------------------------------------------------------------
        # ------------------------ Solve Cutting Plane Problem   -------------------------
        # --------------------------------------------------------------------------------
        q_new = cp.Variable((D,1), integer=True)
        theta = cp.Variable((1,))
        constraints = [cp.sum(q_new) <= d_0, q_new >= 0, q_new <= 1]
        for j in range(len(fq_hist)):
            constraints += [theta <= fq_hist[j] + cp.matmul(gq_hist[j].T, q_new - q)]
        
        prob = cp.Problem(cp.Maximize(theta),
                        constraints)
        prob.solve(solver=cp.GUROBI, verbose=True)
        
        q = q_new.value
        theta_val = theta.value
        print("Iter: ", iter, "q: ", q[:3], "residual: ", residual)
    #print(residual_hist)
    #print(q[:3])

    return q

def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)

def my_trust_region_solver(A,t, check = False):
    """
    Solve the trust region subproblem
        max_{\|z\|_2=1}   z.T @ A @ z + z.T  @ t
    """
    D = len(t)

    # formulate matrix penceil
    M0 = np.block([
        [-np.eye(D),   -A],
        [-A,            -t @ t.T]
                 ])
    M1 = np.block([
    [np.zeros([D,D]),   np.eye(D)],
    [np.eye(D),         np.zeros([D,D])]
    ])

    w, v = eigs(M0, M=-M1, k=2*D, which="LM")
    Lambda = np.real(w[0]).reshape([-1,])                # optimal dual variable
    x = np.real(v[:D,0]).reshape([-1,1])                 # optimal primal variable


    # check optimality condition
    if check == True:
        residual_1 = np.abs(np.linalg.norm(x)-1)
        residual_2 = np.linalg.norm(- (A @ x) + Lambda * x - t)
        residual_3 = is_pos_def(-A + Lambda * np.eye(D))
        print([residual_1, residual_2, residual_3])
    
    Az_t = A @ x + t
    Lambda_z = x.T @ Az_t
    return x, Lambda_z[0]

def solution_recover(A, t, S_chosen):
    """
    Find greedy solution to
        max_{z\in\cZ} z.T @ A @ z + z.T @ t
    provided that non-zero set is given
    """
    D = len(t)
    indexN = np.flatnonzero(S_chosen)
    A_temp = A[np.ix_(indexN, indexN)]
    t_temp = t[indexN]

    z_temp, Lambda_temp = my_trust_region_solver(A_temp, t_temp)
    z = np.zeros([D,1])
    z[indexN, :] = z_temp[:]
    
    return z, Lambda_temp

def my_greedy(A, t, d, silence = True):
    """
    Find greedy solution to
        max_{z\in\cZ} z.T @ A @ z + z.T @ t
    """

    c = 1
    D = len(t)
    S_chosen = [0] * D
    S_unchosen = [1] * D

    indexN = np.flatnonzero(S_unchosen)
    Lambda_val = 1

    sel = []

    while c < d+1:

        Lambda_val = []
        for i in indexN:
            sel.append(i)
            A_temp = A[np.ix_(sel, sel)]
            t_temp = t[sel].reshape([-1,1])

            _, Lambda_temp = my_trust_region_solver(A_temp, t_temp)
            Lambda_val.append(Lambda_temp)
            sel.remove(i)
        
        temp_i = np.argmax(np.array(Lambda_val),0).astype(int)
        opt_i = indexN[temp_i].item()
        S_chosen[opt_i] = 1
        S_unchosen[opt_i] = 0
        sel = list(np.flatnonzero(S_chosen))
        indexN = np.flatnonzero(S_unchosen)
        c = c + 1

        if silence == False:
            print("Iter: ", c, "optimal S: ", S_chosen)
    
    z_opt, Lambda_opt = solution_recover(A, t, S_chosen)
    return S_chosen, z_opt, Lambda_opt[0]

def my_greedy_once_for_all(A, t, silence = True):
    """
    Find greedy solution to
        max_{z\in\cZ} z.T @ A @ z + z.T @ t
    """

    c = 1
    D = len(t)
    S_chosen = [0] * D
    S_unchosen = [1] * D

    indexN = np.flatnonzero(S_unchosen)
    Lambda_val = 1

    sel = []

    opt_i_hist = []
    while c < D+1:

        Lambda_val = []
        for i in indexN:
            sel.append(i)
            A_temp = A[np.ix_(sel, sel)]
            t_temp = t[sel].reshape([-1,1])

            _, Lambda_temp = my_trust_region_solver(A_temp, t_temp)
            Lambda_val.append(Lambda_temp)
            sel.remove(i)
        
        temp_i = np.argmax(np.array(Lambda_val),0).astype(int)
        opt_i = indexN[temp_i].item()
        opt_i_hist.append(opt_i)
        S_chosen[opt_i] = 1
        S_unchosen[opt_i] = 0
        sel = list(np.flatnonzero(S_chosen))
        indexN = np.flatnonzero(S_unchosen)
        c = c + 1

        if silence == False:
            print("Iter: ", c, "chosen element: ", opt_i)
    
    #z_opt, Lambda_opt = solution_recover(A, t, S_chosen)
    return opt_i_hist#S_chosen, z_opt, Lambda_opt[0]

def my_localsearch(A, t, d, S_chosen, z_opt, Lambda_opt, silence=True, maxiter=1000):
    """
    Find local search solution to
        max_{z\in\cZ} z.T @ A @ z + z.T @ t
        Input:
        (A,t): problem parameters
            d: size of selected features
     S_chosen: initial guess of set
        z_opt: initial guess of optimal projection
   Lambda_opt: initial guess of optimal value
    """

    sel = np.flatnonzero(S_chosen)
    D = len(t)

    t_unchosen = [i for i in range(D) if S_chosen[i] == 0]

    best_S = S_chosen
    best_Lambda = Lambda_opt

    optimal = False
    iter = 0
    while (optimal == False) or (iter >= maxiter):
        optimal = True
        for i in sel:
            for j in t_unchosen:
                temp_S = [0] * D
                for g in range(D):
                    temp_S[g] = best_S[g]
                
                temp_S[i] = 0
                temp_S[j] = 1
                temp_sel = np.flatnonzero(temp_S)

                
                A_temp = A[np.ix_(temp_sel, temp_sel)]
                t_temp = t[temp_sel].reshape([-1,1])
                _, Lambda_temp = my_trust_region_solver(A_temp, t_temp)

                iter += 1

                if Lambda_temp > best_Lambda:
                    if silence == False:
                        print("Iter: ", iter, "Best lambda: ", Lambda_temp[0])
                    optimal = False
                    best_S = temp_S
                    best_Lambda = Lambda_temp

                    sel = list(np.flatnonzero(best_S))
                    t_unchosen = [i for i in range(D) if best_S[i] == 0]
                    break

    z_opt, Lambda_opt = solution_recover(A, t, best_S)

    return best_S, z_opt, Lambda_opt[0]

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

def Quad_MMD_training_revision(X, Y, c, Lambda, d, z0, num_epoch=5):
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
        tau = np.median(Data_xy)
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

        MIQP_coeff_A = 0.5 * Hessian_MMD.detach().numpy()
        z_numpy = z.detach().numpy()
        MIQP_coeff_t = -2 * MIQP_coeff_A@z_numpy + Grad_MMD[0].detach().numpy()

        z_new,obj_z_new = MIQP_app_solver(MIQP_coeff_A, MIQP_coeff_t, d)

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



def Quadratic_MMD_testing_revision(X_Te, Y_Te, z, c, num_perm = 100):
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
        tau = np.median(Data_xy)
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








