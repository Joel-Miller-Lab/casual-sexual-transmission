# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 10:15:35 2022

Includes degree zero. Plotting 1 - S, 1 - Psi(theta) and, 1 - e^-chi, Log scale inset for early times inset
Uses the transmission route matrix from Joel's paper. Give a value of R0, and a casual transmission coefficient 
and the sexual transmission will be calculated automatically

Does not solve IVP. Uses final size transcendental equations to plot R(infty) vs R_ss for different values of Rc

The file takes inout for parameters from command line in the order: N0, alpha, max degree, R_c, R_0 and a boolean variable for log scale of inset
Also plots the distribution 
In spyder, use -- "runfile('Final_size_solution.py', args = '%.2f %.2f %d'%(N0, alpha, max_degree))"

@author: pkollepara
"""

# %% Package import and function definitions
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from itertools import cycle
import sys
#import scipy.optimize as so

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
# %% Process Input
N0 = float(sys.argv[1])
alpha = float(sys.argv[2]) #Exponent
cutoff = int(sys.argv[3]) #Maximum degree in the network that is allowed

# %% Network definition
degrees = np.array(list(range(0, cutoff+1))).astype(float)
Nk = [N0]
norm = (1-N0)/np.sum(degrees[1:]**-alpha) #1/0 is not defined
Nk = np.array(Nk + (norm*degrees[1:]**-alpha).tolist())
n = len(degrees)


R_ss_range = np.linspace(0, 8, 20)
R_c_list = np.array([0, 0.8, 1.0, 1.2])

# %% Model parameters
gamma = 1



# %% Find beta1     X[0]:chi:casual, X[1]:theta:sexual

tol = 1e-3

data = {}
for j, R_c in enumerate(R_c_list):
    R_ss_list = []
    R_final_list = []
    for i, R_ss in enumerate(R_ss_range):
        beta2 = R_c*gamma
        beta1 = R_ss*gamma/(1+PsiDoublePrime(1, Nk, degrees)/PsiPrime(1, Nk, degrees))
        #solution = root_scalar(beta1_eqn, args = (beta2, R0, gamma, Nk, degrees), x0 = R0*0.25, x1 = 1.5*R0)
        #beta1 = solution.root
        
        G_route = np.array([[beta2/gamma, beta2/gamma], [beta1*PsiPrime(1, Nk, degrees)/gamma, beta1*(1+PsiDoublePrime(1, Nk, degrees)/PsiPrime(1, Nk, degrees))/gamma]])
        R0 = np.amax(np.abs((np.linalg.eigvals(G_route))))
        if R0>1:
            R_ss_list.append(G_route[1, 1])
            
            # if solution.converged == False:
            #     print('Could not find beta2')
            X = np.array([0.5, 0.5])
            while True:
                tmp = np.copy(X)    
                X = final_prob_relation(X, beta1, beta2, gamma, Nk, degrees)
                if np.linalg.norm(tmp-X)<=tol:
                    break
            R_final = 1 - np.exp(-X[0])*Psi(X[1], Nk, degrees)
            R_final_list.append(R_final)                            
    data[R_c] = [R_ss_list, R_final_list]
    

# %% Plot heatmap
# plt.rcParams['font.family']='serif'
# plt.rcParams['font.serif']='cmr10'
# plt.rcParams['axes.unicode_minus']=False
# plt.rcParams['mathtext.fontset'] = 'cm'
# plt.rcParams['mathtext.fallback'] = 'stix'

plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{lmodern}'

fontsize = 6
plt.rcParams['axes.linewidth'] = 0.4
plt.rcParams['xtick.major.width'] = 0.4
plt.rcParams['ytick.major.width'] = 0.4
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rc('font', size=fontsize)          # controls default text sizes
plt.rc('axes', titlesize=fontsize)     # fontsize of the axes title
plt.rc('axes', labelsize=fontsize)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=fontsize)    # fontsize of the tick labels
plt.rc('ytick', labelsize=fontsize)    # fontsize of the tick labels
plt.rc('legend', fontsize=fontsize)    # legend fontsize


marker = cycle(('o', 'D', '^', 's')) 

fig, ax = plt.subplots(1, 1, figsize = (5*3.5/8, 5*3.5/8), constrained_layout = True)
#ax.set_title(r'$N(k=0) = %.2f$, $\alpha = %.2f$, $k_{max} = %d$'%(N0, alpha, cutoff))

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)


current_cmap = cm.get_cmap('PRGn').copy()
current_cmap.set_bad(color='yellow')

ax.set_xlabel(r'$\mathcal{R}_{\mathrm{s/s}}$')
ax.set_ylabel(r'$R(\infty)$')
#ax.set_ylim(bottom = 0, top = None)
#ax.set_ylim(top = np.amax(R_final_list)*1.2)
ax.set_xlim(0, R_ss_range[-1]*1.1)

for key in data:
    ax.scatter(data[key][0], data[key][1], facecolors = 'thistle', edgecolors = 'k', marker = next(marker), s = 6,
               linewidth = 0.3, label = r'$%.1f$'%key)
    ax.plot(data[key][0], data[key][1], lw = 0.1, c = 'k', ls = '-', zorder = 0)


ax.legend(frameon = False, title = r'$\mathcal{R}_c$', ncol = 2)#, loc = 'upper left')



fname = 'N0_%.2f_kmax_%d_alpha_%.2f'%(N0, cutoff, alpha)


fig.savefig('final-size-'+fname+'.pdf')
plt.close(fig)



fig, ax = plt.subplots(1, 1, figsize = (5*3.5/8, 5*3.5/8), constrained_layout = True)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

ax.set_xlabel(r'$k$')
ax.set_ylabel(r'$N_k$')
#ax.set_ylim(top = 0.75)

#ax.set_ylim(np.amin(degrees), degrees[-1]*1.1)
ax.text(0.5, 0.002, s = r'$N_0 = %.2f$'%N0 + '\n' + r'$N_k \sim k^{-%d}$'%alpha + '\n' + r'$k_{\mathrm{max}} = %d$'%cutoff)
#plt.text()
ax.plot(degrees, Nk, lw = 0.25, c = 'gray', ls = '--', zorder = 0)
ax.scatter(degrees, Nk, facecolors = 'lightsteelblue', edgecolors = 'k', marker = 's', s = 6, linewidth = 0.3)
ax.set_yscale('symlog', linthresh=Nk[-1]*0.9)
ax.set_xscale('symlog', linthresh=1)
ax.set_xlim(0, degrees[-1]*1.1)
ax.minorticks_off()
fig.savefig('dist-'+fname+'.pdf')
plt.close(fig)


