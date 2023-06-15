# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 15:04:19 2023

@author: PKollepara
"""
import numpy as np

def Psi(x, Nk, deg):
    n = len(deg)
    return np.sum([Nk[i]*x**deg[i] for i in range(n)], 0)
    #return np.sum(Nk*x**(np.array(degrees)))

def PsiPrime(x, Nk, deg):
    n = len(deg)
    return np.sum([Nk[i]*deg[i]*x**(deg[i]-1) for i in range(n)], 0)

def PsiDoublePrime(x, Nk, deg):
    n = len(deg)
    return np.sum([Nk[i]*deg[i]*(deg[i]-1)*x**(deg[i]-2) for i in range(n)], 0)

def dxdt(t, X, rho, beta1, beta2, gamma, Nk, degrees):
    R, pi_R, theta, Chi = X
    #rho, beta1, beta2, gamma = ARGS
    #Nk_values = np.array(list(Nk.values()))
    Sk_0 = Nk*(1-rho)
    S = np.exp(-Chi)*np.sum(Sk_0*theta**degrees)
    #pi_S = (1-rho)*theta*PsiPrime(theta, Nk, degrees)*np.exp(-Chi)/PsiPrime(1, Nk, degrees)
    pi_S = np.exp(-Chi)*np.sum(degrees*Sk_0*theta**degrees)/PsiPrime(1, Nk, degrees)
    pi_I = 1 - pi_S - pi_R
    I = 1 - S - R
    return np.array([gamma*I, gamma*pi_I, -beta1*pi_I*theta, beta2*I])

def FS(x, G, n): 
    return [x[k] - n[k]*(1-np.exp(-np.sum(G[k]*x)/n[k])) for k in range(len(n))]

def beta2_eqn(beta2, beta1, R0, gamma, Nk, degrees):
    M = np.array([[beta2/gamma, beta2/gamma], [beta1/gamma*PsiPrime(1, Nk, degrees), beta1/gamma*(1+PsiDoublePrime(1, Nk, degrees)/PsiPrime(1, Nk, degrees))]])
    return np.amax(np.abs((np.linalg.eigvals(M)))) - R0

def beta1_eqn(beta1, beta2, R0, gamma, Nk, degrees):
    M = np.array([[beta2/gamma, beta2/gamma], [beta1/gamma*PsiPrime(1, Nk, degrees), beta1/gamma*(1+PsiDoublePrime(1, Nk, degrees)/PsiPrime(1, Nk, degrees))]])
    return np.amax(np.abs((np.linalg.eigvals(M)))) - R0

def final_prob_relation(X, beta1, beta2, gamma, Nk, degrees):
    Y = np.zeros(2)
    Y[0] = beta2/gamma*(1-np.exp(-X[0])*Psi(X[1], Nk, degrees))
    Y[1] = np.exp(-beta1/gamma*(1-np.exp(-X[0])*X[1]*PsiPrime(X[1], Nk, degrees)/PsiPrime(1, Nk, degrees)))    
    return Y

def iterative_solver(beta1, beta2, gamma, Nk, degrees, X_init, rel_tol):
    X = X_init
    while True:
        tmp = np.copy(X)    
        X = final_prob_relation(X, beta1, beta2, gamma, Nk, degrees)        
        if np.linalg.norm((tmp-X)/X)<=rel_tol:
            break
    return X