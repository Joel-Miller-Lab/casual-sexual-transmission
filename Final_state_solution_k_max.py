# -*- coding: utf-8 -*-
"""
Created on Wed May 24 14:05:19 2023

In spyder, use -- "runfile('Final_state_solution_k_max.py', args = '%.2f %.2f %.2f'%(beta2, N0, alpha))"

@author: pkollepara
"""

# %% Package import and function definitions
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import sys
import seaborn as sns
import time 
#import scipy.optimize as so


#%%
plt.rcParams['font.family'] = 'Helvetica'
plt.rcParams['mathtext.fontset'] = 'cm'
#%%
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
# %% Process Input
beta2 = float(sys.argv[1])
N0 = float(sys.argv[2])
alpha = float(sys.argv[3]) #Exponent

# %% Network definition

beta1_list = np.linspace(0.1, 0.8, 10)
#beta1_list = [0.1, 0.21, 0.3]
cutoff_list = list(range(10, 101, 10))
#cutoff_list = [40]
#R_ss_list = []
# %% Model parameters
gamma = 1

#%%
X_init = np.array([0.4, 0.4])
rel_tol = 1e-3

#%%
p_sexual = np.zeros((len(beta1_list), len(cutoff_list)))
p_casual = np.zeros((len(beta1_list), len(cutoff_list)))
R_ss = np.zeros((len(beta1_list), len(cutoff_list)))
for i, beta1 in enumerate(beta1_list):
    for j, cutoff in enumerate(cutoff_list):

        #Network Definition
        degrees = np.array(list(range(0, cutoff+1))).astype(float)
        Nk = [N0]
        norm = (1-N0)/np.sum(degrees[1:]**-alpha) #1/0 is not defined
        Nk = np.array(Nk + (norm*degrees[1:]**-alpha).tolist())
        n = len(degrees)

        R_ss[i, j] = beta1*(1 + PsiDoublePrime(1, Nk, degrees)/PsiPrime(1, Nk, degrees))
        
        G_route = np.array([[beta2/gamma, beta2/gamma], [beta1*PsiPrime(1, Nk, degrees)/gamma, beta1*(1+PsiDoublePrime(1, Nk, degrees)/PsiPrime(1, Nk, degrees))/gamma]])
        R0 = np.amax(np.abs((np.linalg.eigvals(G_route))))
        
        if R0>1:
            X = iterative_solver(beta1, beta2, gamma, Nk, degrees, X_init, rel_tol)                
            p_casual[i, j] = 1-np.exp(-X[0])
            p_sexual[i, j] = 1-Psi(X[1], Nk, degrees)
        else:
            p_casual[i, j] = np.nan
            p_sexual[i, j] = np.nan

out = p_casual - p_sexual

out2 = (p_casual - p_sexual)/p_sexual

out3 = p_casual/p_sexual

#%%
fig, axs = plt.subplots(1, 3, figsize = (3*9.5*0.5, 8*0.5), tight_layout = True)
#sns.heatmap(out, annot=True, cbar=True, square = True, cmap = 'RdBu_r', norm = mcolors.CenteredNorm(vcenter=0), 
#            linewidths = 1, linecolor = 'white', ax = ax, annot_kws = {'fontsize': 'x-small'})

sns.heatmap(R_ss, annot=True, cbar=True, square = True, cmap = 'RdBu_r', norm = mcolors.CenteredNorm(vcenter=1), 
            ax = axs[0], annot_kws = {'fontsize': 'x-small'}, fmt = '.2f')

sns.heatmap(p_sexual, annot=True, cbar=True, square = True, cmap = 'Blues', 
            ax = axs[1], annot_kws = {'fontsize': 'x-small'}, fmt = '.2f')

sns.heatmap(out3, annot=True, cbar=True, square = True, cmap = 'RdBu_r', norm = mcolors.CenteredNorm(vcenter=1), 
            ax = axs[2], annot_kws = {'fontsize': 'x-small'}, fmt = '.2f')

#sns.heatmap(out2, annot=True, cbar=True, square = True, cmap = 'RdBu_r', norm = mcolors.CenteredNorm(vcenter=0), 
#            ax = axs[1], annot_kws = {'fontsize': 'x-small'}, fmt = '.2f')
#sns.heatmap(out, annot=True, cbar=True, square = True, cmap = 'RdBu_r', norm = mcolors.CenteredNorm(vcenter=0), 
#            ax = axs[0], annot_kws = {'fontsize': 'x-small'}, fmt = '.2f')



for ax in axs:
    ax.invert_yaxis()
    #ax.set_yticks(list(range(0, 10)), labels = [str(round(beta1, 2)) for beta1 in beta1_list])
    ax.set_yticklabels([str(round(beta1, 2)) for beta1 in beta1_list], rotation = 'horizontal')
    ax.set_xticklabels(cutoff_list)
    ax.set_xlabel(r'Maximum allowed sexual partners ($k_{\rm{max}})$')
    ax.set_ylabel(r'Sexual transmission rate parameter ($\beta_1$)')
axs[0].set_title(r'$\mathcal{R}_{\rm{s/s}}$', fontsize = 'small')
axs[1].set_title('Probability of exposure to sexual transmission', fontsize = 'small')
axs[2].set_title('Ratio of probabilities of exposure (casual/sexual)', fontsize = 'small')
plt.suptitle(r'$\mathcal{R}_{\rm{c}}$ = %.2f, $\alpha$ = %d, $N_0$ = %.2f'%(beta2/gamma, alpha, N0))

fname = str(round(time.time()))
fig.savefig('k_max-'+fname+'.pdf')
fig.savefig('k_max-'+fname+'.png', dpi = 600)

#%%
# ax1 = sns.heatmap(p_casual, annot=True, cbar=True, square = True, cmap = 'Blues', 
#             annot_kws = {'fontsize': 'x-small'}, fmt = '.2f')
# ax1.invert_yaxis()
# #ax1.set_yticks(list(range(0, 10)), labels = [str(round(beta1, 2)) for beta1 in beta1_list])
# ax1.set_yticklabels([str(round(beta1, 2)) for beta1 in beta1_list], rotation = 'horizontal')
# ax1.set_xticklabels(cutoff_list)
# ax1.set_xlabel(r'Maximum allowed sexual partners ($k_{\rm{max}})$')
# ax1.set_ylabel(r'Sexual transmission rate parameter ($\beta_1$)')


#%%
# ax2 = sns.heatmap(p_casual*(1-p_sexual), annot=True, cbar=True, square = True, cmap = 'Blues', 
#             annot_kws = {'fontsize': 'x-small'}, fmt = '.2f')
# ax2.invert_yaxis()
# #ax.set_yticks(list(range(0, 10)), labels = [str(round(beta1, 2)) for beta1 in beta1_list])
# ax2.set_yticklabels([str(round(beta1, 2)) for beta1 in beta1_list], rotation = 'horizontal')
# ax2.set_xticklabels(cutoff_list)
# ax2.set_xlabel(r'Maximum allowed sexual partners ($k_{\rm{max}})$')
# ax2.set_ylabel(r'Sexual transmission rate parameter ($\beta_1$)')
# ax2.set_title('Probability of exposure only to casual transmission', fontsize = 'small')

