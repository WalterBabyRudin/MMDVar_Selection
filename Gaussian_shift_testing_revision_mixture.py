import numpy as np
import cvxpy as cp
from numpy.lib import utils
import torch
import scipy.spatial
import utils_revision as utils
import time
from scipy.stats import wishart
from sklearn import metrics
from numpy import linalg as LA
from scipy.sparse.linalg import eigs
import Optimization.Linear_MMD_revision as Linear_MMD
import Optimization.MIQP_solver as MIQP_solver
import Optimization.Quadratic_MMD_revision as Quadratic_MMD
import Optimization.Gaussian_MMD_revision as Gaussian_MMD
from scipy.stats import multivariate_normal
import Optimization.logistic_regression as Logistic_regression

def generate_Gaussian_mixture(N,D,seedNum):
    # input:
    #     N: sample size
    #     D: dimension of target distributions
    mu_1_hist = np.zeros([2,D])
    mu_1_hist[1] = mu_1_hist[1] + 1
    
    mu_2_hist = np.zeros([2,D])
    mu_2_hist[1] = mu_2_hist[1] + 1 + 0.8/np.sqrt(D)
    
    sigma_1 = np.identity(D)
    sigma_2_hist = [np.identity(D), np.identity(D)]
    sigma_2_hist[0][0,0] = 4
    sigma_2_hist[0][1,1] = 4
    sigma_2_hist[0][0,1] = -0.9
    sigma_2_hist[0][1,0] = -0.9
    sigma_2_hist[1][0,1] = 0.9
    sigma_2_hist[1][1,0] = 0.9
    
    X = np.zeros([N,D])
    Y = np.zeros([N,D])
    
    rv_X1 = multivariate_normal(mu_1_hist[0], sigma_1)
    rv_Y1 = multivariate_normal(mu_2_hist[0], sigma_2_hist[0])
    rv_X2 = multivariate_normal(mu_1_hist[1], sigma_1)
    rv_Y2 = multivariate_normal(mu_2_hist[1], sigma_2_hist[1])
    for i in range(N):
        uni_rand = np.random.rand()
        if uni_rand < 0.5:
            X[i, :] = rv_X1.rvs(1)#, random_state=seedNum)
            Y[i, :] = rv_Y1.rvs(1)#, random_state=seedNum+1)
        else:
            X[i, :] = rv_X2.rvs(1)#, random_state=seedNum)
            Y[i, :] = rv_Y2.rvs(1)#, random_state=seedNum+1)
    return X, Y
    




np.random.seed(1)
# Parameter Setting
d = 3
#L_hist = [4,6,8,10,12,14,16,18,20]
D = int(input("Data dimension: "))
#mean_mu_hist_large, mean_nu_hist_large, cov_mu_hist_large, cov_nu_hist_large = utils.parameter_generation(d, 20)

nX_Tr = 50
nY_Tr = 50
nX_Te = 50
nY_Te = 50
n_run = 10
n_test = 200

print('---------------------------')
input_method = input("Enter Testing Method: ")

Power_nrun_hist = []
for trial in range(n_run):
    np.random.seed(5 + 1*11 + trial * 1111 + n_run * 1741)
    torch.manual_seed(5 + 1*11 + trial * 1111 + n_run * 1741)
    torch.cuda.manual_seed(5 + 1*11 + trial * 1111 + n_run * 1741)

    X_Tr, Y_Tr = generate_Gaussian_mixture(nX_Tr, D, seedNum = 5 + 1*11 + trial * 1111 + n_run * 1741)


    if input_method == "LMMD":
        Lambda = 0.1 # variance regularization value for LMMD
        d_proj = 6   # number of variables to be chosen by LMMD
        A, t = Linear_MMD.Linear_MMD_coeff_revision(X_Tr, Y_Tr, Lambda)
        z_linear_MMD, obj_z_linear_MMD = MIQP_solver.MIQP_app_solver(A,t,d=d_proj)
        #print(z_linear_MMD)
    if input_method == "QMMD":

        z0 = np.ones([D,1])
        c = 1
        Lambda = 0.01
        tau = 0.1
        d_proj = 6   # number of variables to be chosen by LMMD

        # A, t = Linear_MMD.Linear_MMD_coeff_revision(X_Tr, Y_Tr, Lambda)
        # z_linear_MMD, obj_z_linear_MMD = MIQP_solver.MIQP_app_solver(A,t,d=d_proj*2)


        # if L >= 8 and L <= 14:
        #     d_proj = 10
        #     Lambda = 0.05
        #     sigma = 0.5
            

        z_quadratic_MMD, obj_z_quadratic_MMD = Quadratic_MMD.Quad_MMD_training_revision(X_Tr, Y_Tr, c, Lambda, d=d_proj, z0=z0, tau=tau, num_epoch=30, sigma=2)
        
    if input_method == "GMMD1":
        z0 = np.ones([D,1])
        sigma = 1.5
        Lambda = 0.01
        tau = 0.05

        d_proj = 10

        A, t = Linear_MMD.Linear_MMD_coeff_revision(X_Tr, Y_Tr, Lambda)
        z_linear_MMD, obj_z_linear_MMD = MIQP_solver.MIQP_app_solver(A,t,d=d_proj)

        z_Gaussian_MMD1, obj_z_Gaussian_MMD1 = Gaussian_MMD.GMMD_training_revision(X_Tr, Y_Tr, sigma, Lambda, d=d_proj, z0=z0, tau0=tau, num_epoch=30)

    if input_method == "SLR":
        d_proj = 6   # number of variables to be chosen by LMMD
        q = Logistic_regression.logistic_classifier_support(X_Tr, Y_Tr, d_0=d_proj)
        w, b = Logistic_regression.logistic_classifier(X_Tr, Y_Tr, q)


    decision_hist = []
    for test_idx in range(n_test):
        np.random.seed(5 + 1*11 + trial * 1111 + n_run * 1741 + 17467*test_idx)
        torch.manual_seed(5 + 1*11 + trial * 1111 + n_run * 1741 + 17467*test_idx)
        torch.cuda.manual_seed(5 + 1*11 + trial * 1111 + n_run * 1741 + 17467*test_idx)

        X_Te, Y_Te = generate_Gaussian_mixture(nX_Te, D, seedNum = 5 + 1*11 + trial * 1111 + n_run * 1741 + 17467*test_idx)
        if input_method == "LMMD":
            p_val, decision =  Linear_MMD.Linear_MMD_testing_revision(X_Te, Y_Te, z_linear_MMD)
            #print('Idx: ',test_idx, '\t D: ',d*L, '\t pval: ', p_val, '\t Decision: ',decision)
            #Linear_MMD_decision_hist[id_L, test_idx, trial] = decision
        if input_method == "QMMD":
            p_val, decision =  Quadratic_MMD.Quadratic_MMD_testing_revision(X_Te, Y_Te, z_quadratic_MMD, c)
            #print('Idx: ',test_idx, '\t D: ',d*L, '\t pval: ', p_val, '\t Decision: ',decision)
            #Quadratic_MMD_decision_hist[id_L, test_idx, trial] = decision
        if input_method == "GMMD1":
            p_val, decision =  Gaussian_MMD.GMMD_testing_revision(X_Te, Y_Te, z_Gaussian_MMD1, sigma)
            #print('Idx: ',test_idx, '\t D: ',d*L, '\t pval: ', p_val, '\t Decision: ',decision)
        if input_method == "SLR":
            decision = Logistic_regression.logistic_classifier_testing(X_Te, Y_Te, w, b)
        decision_hist.append(decision)

    Power = np.mean(np.array(decision_hist))
    print('Proj d: ',d_proj, '\t Power: ', Power)
    Power_nrun_hist.append(Power)

print(np.array(Power_nrun_hist))
np.save("Gaussian_power_"+input_method+"_"+str(D)+"_Mixture.npy", np.array(Power_nrun_hist))



            
            


        


