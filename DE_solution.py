# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 16:20:33 2022

Initial conditions ~ degree 

The file takes inout for parameters from command line in the order: N0, alpha, max degree, R_c, R_0 and a boolean variable for log scale of inset

In spyder, use -- "runfile('DE_solution.py', args = '%.2f %.2f %d %.2f %.2f True'%(N0, alpha, max_degree, R_c, R_0))"
@author: pkollepara
"""

# %% Package import and function definitions
import numpy as np 
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib import rc
import time
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

# %% Process Input
N0 = float(sys.argv[1])
alpha = float(sys.argv[2]) #Exponent
cutoff = int(sys.argv[3]) #Maximum degree in the network that is allowed
R_c = float(sys.argv[4])
R0 = float(sys.argv[5])
log_inset = sys.argv[6]

# %% Network definition
degrees = np.array(list(range(0, cutoff+1))).astype(float)
Nk = [N0]
norm = (1-N0)/np.sum(degrees[1:]**-alpha) #1/0 is not defined
Nk = np.array(Nk + (norm*degrees[1:]**-alpha).tolist())
n = len(degrees)


# %% Initial conditions
rho_total = 0.0001
rho = rho_total/PsiPrime(1, Nk, degrees)*degrees
#rho = rho_total/n*np.ones(n)
Sk_0 = Nk*(1-rho)


# %% Model parameters
gamma = 1
beta2 = R_c*gamma

# %% Find beta1


solution = root_scalar(beta1_eqn, args = (beta2, R0, gamma, Nk, degrees), x0 = R0*0.25, x1 = 1.5*R0)
beta1 = solution.root

if solution.converged == False:
    print('Not converged')
        
# %% Solve ODE - both transmissions AND plotting
dT = 0.01
T = np.arange(0, 40, dT)
print('Step Size: ', dT)
print('Time: ', T[0], T[-1])
X_0 = [0, 0, 1, 0]

print('R_0 from NGM' + '    ' + 'R_0 from route matrix')


    #print(beta1, beta2)
G = np.array([[Nk[k]*(degrees[k]*degrees[l]*beta1/PsiPrime(1, Nk, degrees) + beta2)/gamma for l in range(n)] for k in range(n)])
G_route = np.array([[beta2/gamma, beta2/gamma], [beta1*PsiPrime(1, Nk, degrees)/gamma, beta1*(1+PsiDoublePrime(1, Nk, degrees)/PsiPrime(1, Nk, degrees))/gamma]])
R0 = np.amax(np.abs((np.linalg.eigvals(G))))
R0_route = np.amax(np.abs((np.linalg.eigvals(G_route))))
print('%.3f'%R0, '           ', '%.3f'%R0_route)


plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.latex.preamble'] = r'\usepackage{lmodern}'
#plt.rc('text', usetex=True)
#plt.rcParams['font.size'] = '6'
fontsize= 6
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

fig, ax = plt.subplots(1, 1, figsize = (8*3.5/8, 5*3.5/8), constrained_layout = True)
fig2, ax2 = plt.subplots(1, 1, figsize = (8*3.5/8, 5*3.5/8))

ax.yaxis.tick_right()
ax.spines['left'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.text(.4, 0.8, r'$\mathcal{R}_0 = %.2f$'%R0, transform = ax.transAxes)
ax.text(.4, 0.5, r'$\mathcal{R}_{\mathrm{c}} = %.2f$'%(G_route[0, 0]) + '\n'+
        r'$\mathcal{R}_{\mathrm{s/c}} = %.2f$'%(G_route[1, 0]) + '\n'+
        r'$\mathcal{R}_{\mathrm{s/s}} = %.2f$'%(G_route[1, 1]), transform = ax.transAxes)

ax2.yaxis.tick_right()
ax2.spines['left'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.text(.4, 0.8, r'$\mathcal{R}_0 = %.2f$'%R0, transform = ax2.transAxes)
ax2.text(.4, 0.5, r'$\mathcal{R}_{\mathrm{c}} = %.2f$'%(G_route[0, 0]) + '\n'+
        r'$\mathcal{R}_{\mathrm{s/c}} = %.2f$'%(G_route[1, 0]) + '\n'+
        r'$\mathcal{R}_{\mathrm{s/s}} = %.2f$'%(G_route[1, 1]), transform = ax2.transAxes)

if R0 > 1 and np.abs(R0-R0_route)<1e-2:
    sol = solve_ivp(lambda t, X: dxdt(t, X, rho, beta1, beta2, gamma, Nk, degrees), t_span=(0, T[-1]), y0 = X_0, method = 'RK45', t_eval = T[1:-1])
    R, pi_R, theta, Chi = sol.y[0, :], sol.y[1, :], sol.y[2, :], sol.y[3, :]
    S = np.exp(-Chi)*np.sum(np.array([Sk_0[i]*theta**degrees[i] for i in range(n)]), 0)    
    Psioftheta = Psi(theta, Nk, degrees)
    
    
    ax.plot(sol.t, 1-S, label = r'Proportion infected', ls = ':', lw = 0.75, c = 'k')
    ax.plot(sol.t, 1 - np.exp(-Chi), label = r'Casual transmission', ls = '-.', lw = 0.5, c = 'k')
    ax.plot(sol.t, 1 - Psioftheta, label = r'Sexual transmission', lw = 0.5, c = 'k')
    
    pi_S = np.exp(-Chi)*theta*PsiPrime(theta, Nk, degrees)/PsiPrime(1, Nk, degrees)
    Rate_sexual_exposure = beta1*PsiPrime(theta, Nk, degrees)*theta*(1-pi_S-pi_R)*theta
    Rate_casual_exposure = beta2*np.exp(-Chi)*(1-S-R)
    
    for deg in np.round(np.linspace(0, degrees[-1], 5)):
        ax2.plot(sol.t, np.exp(-Chi)*theta**deg, label = r'$k = %d$'%(deg), lw = 0.6, ls = ':')
    ax2.plot(sol.t, Rate_casual_exposure, ls = '-.', lw = 0.5, c = 'k', label = 'Casual rate')
    ax2.plot(sol.t, Rate_sexual_exposure, ls = '-', lw = 0.5, c = 'k', label = 'Sexual rate')
    
    axins = inset_axes(ax, width="25%", height="30%", loc=2, borderpad=2)
    axins.plot(sol.t, 1-S, label = r'$1-S$', ls = ':', lw = 0.75, c = 'k')
    axins.plot(sol.t, 1 - np.exp(-Chi), label = r'$1-e^{-\chi}$', ls = '-.', lw = 0.5, c = 'k')
    axins.plot(sol.t, 1 - Psioftheta, label = r'$1-\Psi(\theta)$', lw = 0.5, c = 'k')
    axins.set_xlim(0, T[np.argmin(np.abs(1 - S - 0.1))])
    axins.set_ylim(10**-7, 0.05)
    if log_inset == 'True':
        axins.set_yscale('log')
    axins.minorticks_off()
    
    ax.set_xlabel(r'$t$', size = fontsize + 1)
    ax.legend(ncol = 1, loc = 'upper right', frameon = False)
    
    #ax.set_xlim(0, 11)
    ax.set_ylim(0, 1.05)
    
    ax2.set_xlabel(r'$t$', size = fontsize + 1)
    ax2.legend(ncol = 1, loc = 'upper right', frameon = False)
    
    #ax2.set_xlim(0, 11)
    ax2.set_ylim(0, 1.05)

params = {}
params['N0'] = N0
params['alpha'] = round(alpha, 3)
params['cutoff'] = cutoff
params['R_c'] = round(R_c, 3)
params['R_0'] = round(R0, 3)
params['gamma'] = round(gamma, 3)
params['rho total'] = round(rho_total, 3)
params['beta1 (computed)'] = round(beta1, 3)
params['beta2 (computed)'] = round(beta2, 3)

fname = str(round(time.time()))

fig.savefig('DE-'+fname+'.pdf', metadata = {'Subject': str(params)})
fig2.savefig('DE-instant-'+fname+'.pdf', metadata = {'Subject': str(params)})
#fig.savefig('11-'+fname+'.png', metadata = {'Description': str(params)}, dpi = 300)
fig2.savefig('DE-instant-'+fname+'.png', metadata = {'Description': str(params)}, dpi = 300)

plt.close(fig)



