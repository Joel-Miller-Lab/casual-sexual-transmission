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

def dIkdt(t, x, gamma, c):
    return c[t] - gamma*x

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
T = np.arange(0, 11, dT)
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


#plt.rcParams['font.sans-serif'] = "CMU Serif"
plt.rcParams['font.family'] = 'Helvetica'
plt.rcParams['mathtext.fontset'] = 'cm'
#plt.rcParams['axes.unicode_minus'] = False
#plt.rcParams['svg.fonttype'] = 'none'

# plt.rcParams['text.usetex'] = True
# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['text.latex.preamble'] = r'\usepackage{lmodern}'

fontsize= 4
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

#fig, ax = plt.subplots(1, 1, figsize = (8*3.5/8, 5*3.5/8), constrained_layout = True)
#fig2, ax2 = plt.subplots(1, 1, figsize = (8*3.5/8, 5*3.5/8))
#fig3, ax3 = plt.subplots(1, 1, figsize = (8*3.5/8, 5*3.5/8), constrained_layout = True)
FIG, [ax, ax2, ax3] = plt.subplots(3, 1, figsize = ((5*3.5/8, 8*3.5/8)), constrained_layout = True)

ax.yaxis.tick_right()
ax.spines['left'].set_visible(False)
ax.spines['top'].set_visible(False)

# ax.text(.75, 0.8, r'$\mathcal{R}_0 = %.2f$'%R0, transform = ax.transAxes)
# ax.text(.75, 0.6, r'$\mathcal{R}_{\mathrm{c}} = %.2f$'%(G_route[0, 0]) + '\n'+
#         r'$\mathcal{R}_{\mathrm{s/c}} = %.2f$'%(G_route[1, 0]) + '\n'+
#         r'$\mathcal{R}_{\mathrm{s/s}} = %.2f$'%(G_route[1, 1]), transform = ax.transAxes)

Title = r'$\mathcal{R}_0 = %.2f$'%R0 + ', ' + r'$\mathcal{R}_{\mathrm{c}} = %.2f$'%(G_route[0, 0]) + '\n'+ r'$\mathcal{R}_{\mathrm{s/c}} = %.2f$'%(G_route[1, 0]) + ', '+ r'$\mathcal{R}_{\mathrm{s/s}} = %.2f$'%(G_route[1, 1])
ax3.set_title(Title, y=-0.5)

ax3.yaxis.tick_right()
ax3.spines['left'].set_visible(False)
ax3.spines['top'].set_visible(False)
#ax3.text(.4, 0.8, r'$\mathcal{R}_0 = %.2f$'%R0, transform = ax3.transAxes)
# ax3.text(.4, 0.5, r'$\mathcal{R}_{\mathrm{c}} = %.2f$'%(G_route[0, 0]) + '\n'+
#         r'$\mathcal{R}_{\mathrm{s/c}} = %.2f$'%(G_route[1, 0]) + '\n'+
#         r'$\mathcal{R}_{\mathrm{s/s}} = %.2f$'%(G_route[1, 1]), transform = ax3.transAxes)

ax2.yaxis.tick_right()
ax2.spines['left'].set_visible(False)
ax2.spines['top'].set_visible(False)

if R0 > 1 and np.abs(R0-R0_route)<1e-2:
    sol = solve_ivp(lambda t, X: dxdt(t, X, rho, beta1, beta2, gamma, Nk, degrees), t_span=(0, T[-1]), y0 = X_0, method = 'BDF', t_eval = T[1:-1])
    R, pi_R, theta, Chi = sol.y[0, :], sol.y[1, :], sol.y[2, :], sol.y[3, :]
    S = np.exp(-Chi)*np.sum(np.array([Sk_0[i]*theta**degrees[i] for i in range(n)]), 0)        
    Psioftheta = Psi(theta, Nk, degrees)
    #S_approx = Psioftheta*np.exp(-Chi)
    pi_S = np.exp(-Chi)*theta*PsiPrime(theta, Nk, degrees)/PsiPrime(1, Nk, degrees)
    Rate_sexual_exposure = beta1*PsiPrime(theta, Nk, degrees)*(1-pi_S-pi_R)*theta
    Rate_casual_exposure = beta2*np.exp(-Chi)*(1-S-R)
    Rate_sexual_infections = np.exp(-Chi)*Rate_sexual_exposure
    Rate_casual_infections = Psioftheta*Rate_casual_exposure
    
    ax.plot(sol.t, 1-S, label = r'Any', ls = ':', lw = 0.75, c = 'k')
    ax.plot(sol.t, 1 - np.exp(-Chi), label = r'Casual', ls = '-.', lw = 0.5, c = 'k')
    ax.plot(sol.t, 1 - Psioftheta, label = r'Sexual', lw = 0.5, c = 'k')
    
    # plot_arr = np.exp(-Chi)*theta**degrees[0]
    # for deg in degrees[1:]:
    #     #ax3.plot(sol.t, np.exp(-Chi)*theta**deg, label = r'$k = %d$'%(deg), lw = 0.6, ls = ':')
    #     Sk = Sk_0[deg]*np.exp(-Chi)*theta**deg
    #     plot_arr = np.vstack((plot_arr, Sk/Sk_0[deg]))
         #ax3.scatter(sol.t, deg*np.ones(len(sol.t)), c = np.exp(-Chi)*theta**deg, cmap = 'viridis', vmin = 0, vmax = 1, marker = '|', s = 3.2, alpha = 0.5)
    Sk = Sk_0[0]*np.exp(-Chi)*theta**degrees[0]
    plot_arr = -np.cumsum(np.exp(gamma*sol.t[0:-1])*np.diff(Sk))/np.exp(gamma*sol.t[0:-1])/Nk[0]
    #ax2.plot(sol.t[0:-1], plot_arr, label = r'$k = %d$'%(0), lw = 0.6, ls = ':', c = 'gray')
    
    for deg in degrees[1:]:    
        Sk = Sk_0[int(deg)]*np.exp(-Chi)*theta**deg
        Ik = -np.cumsum(np.exp(gamma*sol.t[0:-1])*np.diff(Sk))/np.exp(gamma*sol.t[0:-1])
        plot_arr = np.vstack((plot_arr, Ik/Nk[int(deg)]))
        # if int(deg)%10==0:
        #     ax2.plot(sol.t[0:-1], Ik/Nk[int(deg)], label = r'$k = %d$'%(deg), lw = 0.6, ls = ':')
    #im = ax3.imshow(plot_arr, cmap = 'bone_r', aspect = 'auto', interpolation = 'none', extent = (0, T[-1], degrees[-1]+1, 0))#, vmin=0, vmax=1)
    im = ax3.imshow(plot_arr, cmap = 'hot', aspect = 'auto', interpolation = 'none', extent = (0, T[-1], degrees[-1]+1, 0))#, vmin=0, vmax=1)
    yticks = list(range(0, int(degrees[-1])+1, int(len(degrees)/5)))
    ax3.set_yticks(yticks)
    #cb = fig3.colorbar(im)
    cb = FIG.colorbar(im, ax = ax3, aspect = 15, location = 'left')
    cb.ax.set_title(r'$I_k/N_k$')
        
    axins1 = inset_axes(ax, width="20%", height="30%", loc=2, borderpad=2)
    axins1.plot(sol.t, 1-S, label = r'$1-S$', ls = ':', lw = 0.75, c = 'k')
    axins1.plot(sol.t, 1 - np.exp(-Chi), label = r'$1-e^{-\chi}$', ls = '-.', lw = 0.5, c = 'k')
    axins1.plot(sol.t, 1 - Psioftheta, label = r'$1-\Psi(\theta)$', lw = 0.5, c = 'k')
    axins1.set_xlim(0, T[np.argmin(np.abs(1 - S - 0.1))])
    axins1.set_ylim(10**-7, 0.05)
    #axins1.set_title('Initial dynamics')
    if log_inset == 'True':
        axins1.set_yscale('log')
    axins1.minorticks_off()
    
    ax2.plot(sol.t, Rate_casual_infections + Rate_sexual_infections, ls = ':', lw = 0.75, c = 'k', label = 'Any')
    ax2.plot(sol.t, Rate_casual_infections, ls = '-.', lw = 0.5, c = 'k', label = 'Casual')
    ax2.plot(sol.t, Rate_sexual_infections, ls = '-', lw = 0.5, c = 'k', label = 'Sexual')
    
    
    # axins2 = inset_axes(ax, width="20%", height="30%", borderpad=2, loc=9)
    # axins2.plot(sol.t, Rate_casual_infections, ls = '-.', lw = 0.5, c = 'k')
    # axins2.plot(sol.t, Rate_sexual_infections, ls = '-', lw = 0.5, c = 'k')
    # axins2.plot(sol.t, Rate_casual_infections + Rate_sexual_infections, ls = ':', lw = 0.5, c = 'k')
    #axins2.plot(sol.t, Rate_casual_exposure, ls = '-.', lw = 0.5, c = 'k')
    #axins2.plot(sol.t, Rate_sexual_exposure, ls = '-', lw = 0.5, c = 'k')
    
    #axins2.plot(sol.t[0:-1], -np.diff(S)/dT, ls = ':', lw = 0.75, c = 'k')
    # axins2.set_xlim(0, T[-1])
    # axins2.set_ylim(0, )
    # axins2.set_title('New infections rate')
    
    #ax2.plot(sol.t, Rate_casual_exposure, label = r'Casual', lw = 1, ls = '-', c = 'k')
    #ax2.plot(sol.t, Rate_sexual_exposure, label = r'Sexual', lw = 1, ls = '--', c = 'k')
    # axins3 = inset_axes(ax, width="20%", height="30%", borderpad=2, loc=1)
    # im3 = axins3.imshow(plot_arr, cmap = 'bone_r', aspect = 'auto', interpolation = 'none', extent = (0, T[-1], degrees[-1], 0))#, vmin=0, vmax=1)
    # #cb3 = fig.colorbar(im3, cax=axins3)
    # #cb3.ax.set_title(r'$I_k/N_k$')
    # axins3.set_xlim(0, 11)
    # #axins3.set_xlabel(r'$t$', size = fontsize + 1)
    # axins3.set_ylabel(r'$k$', size = fontsize + 1)
    # for side in axins3.spines.keys():  # 'top', 'bottom', 'left', 'right'
    #     axins3.spines[side].set_linewidth(0.2)
    
    #ax.set_xlabel(r'$t$', size = fontsize + 1)
    ax.set_ylabel('Cumulative probability \n of exposure', size = fontsize + 1)
    ax.yaxis.set_label_position("right")
    ax.legend(ncol = 3, loc = 'upper right', frameon = False)
    ax.set_xlim(0, T[-1])
    ax.set_ylim(0, 1.05)    
    
    ax3.set_xlabel('Time', size = fontsize + 1)
    ax3.set_ylabel('No. of sexual partners', size = fontsize + 1)
    ax3.yaxis.set_label_position("right")
    #ax3.legend(ncol = 1, loc = 'lower left', frameon = False)
    ax3.set_xlim(0, T[-1])
    
    #ax2.set_xlabel(r'$t$', size = fontsize + 1)
    ax2.set_ylabel('Rate of \n new infections', size = fontsize + 1)
    ax2.yaxis.set_label_position("right")
    ax2.legend(ncol = 1, frameon = False)
    ax2.set_xlim(0, T[-1])
    ax2.set_ylim(0, )
    
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

#fig.savefig('DE-'+fname+'.pdf', metadata = {'Subject': str(params)})
#fig3.savefig('DE-instant-'+fname+'.pdf', metadata = {'Subject': str(params)})
FIG.savefig('DE-combined-'+fname+'.pdf', metadata = {'Subject': str(params)})

#fig.savefig('DE-'+fname+'.png', metadata = {'Description': str(params)}, dpi = 300)
#fig3.savefig('DE-k-'+fname+'.png', metadata = {'Description': str(params)}, dpi = 300)
FIG.savefig('DE-combined-'+fname+'.png', metadata = {'Description': str(params)}, dpi = 600)



