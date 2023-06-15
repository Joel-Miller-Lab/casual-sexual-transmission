# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 15:00:08 2023

In spyder, use -- "runfile('Final_state_solution_analysis.py', args = '%.2f %.2f %d'%(N0, alpha, max_degree))"

@author: PKollepara
"""

# %% Package import and function definitions
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import sys
import seaborn as sns
import time 
import casual_sexual_transmission_functions as cst
#import scipy.optimize as so

#%% Plot fonts and styling
plt.style.use('seaborn-colorblind')
plt.rcParams['font.family'] = 'Helvetica'
plt.rcParams['mathtext.fontset'] = 'cm'

#%% Process Input
N0 = float(sys.argv[1])
alpha = float(sys.argv[2]) #Exponent
cutoff = int(sys.argv[3])

#%% Network definition
degrees = np.array(list(range(0, cutoff+1))).astype(float)
Nk = [N0]
norm = (1-N0)/np.sum(degrees[1:]**-alpha) #1/0 is not defined
Nk = np.array(Nk + (norm*degrees[1:]**-alpha).tolist())
n = len(degrees)

#%% Model parameters
gamma = 1
network_structure_parameter = (1 + cst.PsiDoublePrime(1, Nk, degrees)/cst.PsiPrime(1, Nk, degrees))
R_ss_min = 0.01
R_ss_max = 6
beta1_list = np.linspace(gamma*R_ss_min/network_structure_parameter,
                         gamma*R_ss_max/network_structure_parameter, 500)


beta2_list = [0.4, 0.5, 0.6, 0.9]
R_ss = np.zeros(len(beta1_list))
for i, beta1 in enumerate(beta1_list):
    R_ss[i] = beta1*network_structure_parameter
#%% Iterative solver parameters
X_init = np.array([0.4, 0.4])
rel_tol = 1e-3

#%% Main function 
def solver(beta2):
    p_sexual = np.zeros(len(beta1_list))
    p_casual = np.zeros(len(beta1_list))
    
    R0 = np.zeros(len(beta1_list))
    for i, beta1 in enumerate(beta1_list):
        G_route = np.array([[beta2/gamma, beta2/gamma], [beta1*cst.PsiPrime(1, Nk, degrees)/gamma, beta1*(1+cst.PsiDoublePrime(1, Nk, degrees)/cst.PsiPrime(1, Nk, degrees))/gamma]])
        R0[i] = np.amax(np.abs((np.linalg.eigvals(G_route))))
        #print(R0)
        if R0[i]>1:
            X = cst.iterative_solver(beta1, beta2, gamma, Nk, degrees, X_init, rel_tol)                
            p_casual[i] = 1-np.exp(-X[0])
            p_sexual[i] = 1-cst.Psi(X[1], Nk, degrees)
        else:
            p_casual[i] = np.nan
            p_sexual[i] = np.nan
    return p_casual, p_sexual, R0


#%%
fontsize= 5
plt.rcParams['axes.linewidth'] = 0.2
plt.rcParams['xtick.major.width'] = 0.2
plt.rcParams['ytick.major.width'] = 0.2
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rc('font', size=fontsize)          # controls default text sizes
plt.rc('axes', titlesize=fontsize)     # fontsize of the axes title
plt.rc('axes', labelsize=fontsize)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=fontsize)    # fontsize of the tick labels
plt.rc('ytick', labelsize=fontsize)    # fontsize of the tick labels
plt.rc('legend', fontsize=fontsize)    # legend fontsize

#%% Main function calls and figure
fig, axs = plt.subplots(1, 1, figsize = (5/3, 8/3), tight_layout = True)
for beta2 in beta2_list:
    p_casual, p_sexual, R0 = solver(beta2)
    axs.plot(R_ss, p_casual/p_sexual, label = r'%.2f'%beta2, lw = 0.75)    

axs.hlines(1, R_ss[0], R_ss[-1], ls = ':', lw = 0.5, color = 'k')
#axs.set_title('Ratio of exposure probabilities (w.r.t to sexual)')
axs.set_xlabel(r'$\mathcal{R}_{\rm{s/s}}$')
axs.grid('on', lw = 0.2, ls = '-')
axs.set_ylim(0, )
axs.set_xlim(0, )
axs.legend(frameon=False, title = r'$\mathcal{R}_c$')
plt.suptitle(r'$\alpha =$ %.2f, $N_0 =$ %.2f, $k_{\rm{max}} =$ %d'%(alpha, N0, cutoff))
fname = str(round(time.time()))
#fig.savefig('final-state-analysis-'+fname+'.png', dpi = 600)
#fig.savefig('final-state-analysis-'+fname+'.pdf')
fig.savefig('final-state-analysis-'+fname+'.svg')


#%%
# fig, axs = plt.subplots(1, 2, figsize = (8/2, 5/2), tight_layout = True)



# axs[0].plot(R_ss, p_casual, label = 'Casual transmission', lw = 1, ls = '-.', c = 'palevioletred')
# axs[0].plot(R_ss, p_sexual, label = 'Sexual transmission', lw = 1, ls = '--', c = 'steelblue')
# axs[0].set_title('Exposure probability')
# axs[0].set_xlabel(r'$\mathcal{R}_{\rm{s/s}}$')
# axs[0].legend(frameon=False, ncol = 1)
# axs[0].grid('on', lw = 0.2, ls = '-')
# axs[0].set_ylim(0, )
# axs[0].set_xlim(0, )


# axs[1].plot(R_ss, p_casual/p_sexual, label = 'p_casual/p_sexual', lw = 1, ls = '--', c = 'darkseagreen')
# axs[1].hlines(1, R_ss[0], R_ss[-1], ls = ':', lw = 0.5, color = 'k')
# axs[1].set_title('Ratio of exposure probabilities (w.r.t to sexual)')
# axs[1].set_xlabel(r'$\mathcal{R}_{\rm{s/s}}$')
# axs[1].grid('on', lw = 0.2, ls = '-')
# axs[1].set_ylim(0, )
# axs[1].set_xlim(0, )


