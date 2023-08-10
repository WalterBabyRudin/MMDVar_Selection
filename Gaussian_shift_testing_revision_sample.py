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
import Optimization.logistic_regression as Logistic_regression

np.random.seed(1)
# Parameter Setting
d = 3
L = 20

N = int(input("Number of samples: "))
mean_mu_hist_large, mean_nu_hist_large, cov_mu_hist_large, cov_nu_hist_large = utils.parameter_generation(d, 20)

nX_Tr = N
nY_Tr = N
nX_Te = N
nY_Te = N
n_run = 10
n_test = 200

print('---------------------------')
input_method = input("Enter Testing Method: ")

mean_mu_hist, mean_nu_hist, cov_mu_hist, cov_nu_hist = mean_mu_hist_large[:L], mean_nu_hist_large[:L], cov_mu_hist_large[:L], cov_nu_hist_large[:L]

Power_nrun_hist = []
for trial in range(n_run):
    np.random.seed(5 + 1*11 + trial * 1111 + n_run * 1741)
    torch.manual_seed(5 + 1*11 + trial * 1111 + n_run * 1741)
    torch.cuda.manual_seed(5 + 1*11 + trial * 1111 + n_run * 1741)

    X_Tr, Y_Tr = utils.data_generation(mean_mu_hist, mean_nu_hist, cov_mu_hist, cov_nu_hist, nX_Tr)

    if input_method == "LMMD":
        Lambda = 0.1 # variance regularization value for LMMD
        d_proj = 6   # number of variables to be chosen by LMMD
        A, t = Linear_MMD.Linear_MMD_coeff_revision(X_Tr, Y_Tr, Lambda)
        z_linear_MMD, obj_z_linear_MMD = MIQP_solver.MIQP_app_solver(A,t,d=d_proj)
        #print(z_linear_MMD)
    if input_method == "QMMD":

        z0 = np.ones([L*3,1])
        c = 1
        Lambda = 0.01
        tau = 0.1
        d_proj = 6   # number of variables to be chosen by LMMD

        A, t = Linear_MMD.Linear_MMD_coeff_revision(X_Tr, Y_Tr, Lambda)
        z_linear_MMD, obj_z_linear_MMD = MIQP_solver.MIQP_app_solver(A,t,d=d_proj)


        # if L >= 8 and L <= 14:
        #     d_proj = 10
        #     Lambda = 0.05
        #     sigma = 0.5
            

        z_quadratic_MMD, obj_z_quadratic_MMD = Quadratic_MMD.Quad_MMD_training_revision(X_Tr, Y_Tr, c, Lambda, d=d_proj, z0=z_linear_MMD, tau=tau, num_epoch=1, sigma=2)
        
    if input_method == "GMMD1":
        z0 = np.ones([L*3,1])
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

        X_Te, Y_Te = utils.data_generation(mean_mu_hist, mean_nu_hist, cov_mu_hist, cov_nu_hist, nX_Te)
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
np.save("Gaussian_power_sample_"+input_method+"_"+str(N)+".npy", np.array(Power_nrun_hist))



            
            


        


