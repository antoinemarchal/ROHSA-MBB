import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib import ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy.io import fits
from astropy import constants as const
from astropy import units as u
from scipy import optimize

import mod_optimize as mod_opt
reload(mod_opt)

import turbulence as tb

def planckct():
    colombi1_cmap = matplotlib.colors.ListedColormap(np.loadtxt("/home/amarchal/library/magnetar/Planck_Parchment_RGB.txt")/255.)
    colombi1_cmap.set_bad("white")
    colombi1_cmap.set_under("white")
    return colombi1_cmap

#Open data GNILC_IRIS-SFD_adim
path = "/data/amarchal/GNILC_IRIS-SFD_adim/"
# path = "/data/amarchal/PLANCK_IRIS-SFD_adim/"
fitsname = "GNILC_IRIS-SFD_adim_G86.fits"
# fitsname = "PLANCK_IRIS-SFD_adim_G86.fits"
hdu = fits.open(path+fitsname)
hdr = hdu[0].header
cube = hdu[0].data[:,44-32:44+32,44-32:44+32] *1.e6 #Attention rescale
# cube = hdu[0].data[:,:64,:64] *1.e6 #Attention rescale

nside = np.max([int(np.ceil(np.log(cube.shape[1]) / np.log(2))), int(np.ceil(np.log(cube.shape[2]) / np.log(2)))])

freq = np.array([np.full((cube.shape[1],cube.shape[2]),i) for i in np.array([353,545,857,3000])])*u.GHz
wavelength = (const.c / freq).to(u.micron)
wavelength_full = np.array([np.full((cube.shape[1],cube.shape[2]),i) for i in np.arange(100,900,1)]) * u.micron

#Open HI model CNM WNM with ROHSA
path_HI = "/data/amarchal/G86/model/"
fitsname_HI = "GHIGLS_G86_CNM_WNM.fits"
hdu_HI = fits.open(path_HI+fitsname_HI)
hdr_HI = hdu_HI[0].header
# NHI = hdu_HI[0].data[:,44-32:44+32,44-32:44+32] *u.cm**-2 # 1.e18
# NHI = hdu_HI[0].data[:,:64,:64] *u.cm**-2 # 1.e18

NHI = np.zeros((1,cube.shape[1],cube.shape[2]))
NHI[0,:,:] = np.sum(hdu_HI[0].data[:,44-32:44+32,44-32:44+32],0)
NHI = NHI*u.cm**-2

# #Reshape cubes
# cube = mod_opt.reshape_cube_up(cube,nside)
# NHI =  mod_opt.reshape_cube_up(NHI,nside) *u.cm**-2

#Parameters ROHSA+ MBB
n_mbb = NHI.shape[0]
lambda_sig = 100.                                                                                                                      
lambda_beta = 0.5
lambda_T = 0.5                                                                              
lambda_var_sig = 100.
lambda_var_beta = 0.
lambda_var_T = 0.
lb_sig = 0.                                                                                                       
ub_sig = np.inf                                                                                                       
lb_beta = 0.                                                                                                       
ub_beta = np.inf                                                                                                       
lb_T = 0.                                                                                                       
ub_T = np.inf                       
lb_b = 0.                                                                                                       
ub_b = np.inf                       
lb_c = 0.                                                                                                       
ub_c = np.inf                       
lb_d = 0.                                                                                                       
ub_d = np.inf                       
l0 = 349.81617036*u.micron
sig_init = 1. 
beta_init = 2. 
T_init = 17. 
maxiter_init = 200                                                                                              
maxiter = 800                                                                                   
m = 10                                                                                                
noise = ".false."                                                                                                              
iprint = 1                                                                                                                     
iprint_init = 1                                                 
save_grid = ".true."  

kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]) / 4.

#Start algorithn ROHSA+ MMB
mean_sed = np.mean(cube,(1,2))
mean_NHI = np.mean(NHI,(1,2))
mean_params = np.zeros(3*n_mbb)
mean_params[0::3] = sig_init
mean_params[1::3] = beta_init
mean_params[2::3] = T_init

#Fit mean SED
bounds = mod_opt.init_bounds_mean(n_mbb, lb_sig, ub_sig, lb_beta, ub_beta, lb_T, ub_T)

init = optimize.fmin_l_bfgs_b(mod_opt.f_g_mean, mean_params, args=(n_mbb, wavelength[:,0,0], mean_sed, l0, mean_NHI), bounds=bounds, 
                                    approx_grad=False, disp=iprint_init, maxiter=maxiter_init)

#Init multiresolution process
b = init[0][0::3]
c = init[0][1::3]
d = init[0][2::3]

params = np.zeros((3*n_mbb,2,2))
for i in range(params.shape[1]):
    for j in range(params.shape[2]):
        params[:,i,j] = init[0]

for n in np.arange(nside)+1: 

    dim_params = params.shape[0]*params.shape[1]*params.shape[2]
    theta = np.zeros(dim_params+(3*len(b)))

    SED_mean = mod_opt.mean_cube(cube,2**n)
    NHI_mean = mod_opt.mean_cube(NHI,2**n) * u.cm**-2
    
    freq_mean = np.array([np.full((SED_mean.shape[1],SED_mean.shape[2]),i) for i in np.array([353,545,857,3000])])*u.GHz
    wavelength_mean = (const.c / freq_mean).to(u.micron)

    bounds = mod_opt.init_bounds(SED_mean, params, lb_sig, ub_sig, lb_beta, ub_beta, lb_T, ub_T)

    theta[:dim_params] = params.ravel()    
    for i in np.arange(len(b)): 
        theta[dim_params+(0+(3*i))] = b[i]
        theta[dim_params+(1+(3*i))] = c[i]
        theta[dim_params+(2+(3*i))] = d[i]
        bounds.append((lb_b, ub_b))
        bounds.append((lb_c, ub_c))
        bounds.append((lb_d, ub_d))

    theta = optimize.fmin_l_bfgs_b(mod_opt.f_g, theta, args=(n_mbb, wavelength_mean, SED_mean, l0, NHI_mean, lambda_sig, lambda_beta, lambda_T, 
                                                             lambda_var_sig, lambda_var_beta, lambda_var_T, kernel), 
                                   bounds=bounds, approx_grad=False, disp=iprint, maxiter=maxiter)
    params = np.reshape(theta[0][:dim_params],(params.shape))
    for i in np.arange(len(b)): 
        b[i] = theta[0][dim_params+(0+(3*i))]
        c[i] = theta[0][dim_params+(1+(3*i))]
        d[i] = theta[0][dim_params+(2+(3*i))]

    if n != nside : params = mod_opt.go_up_level(params)
    
wavelength_mean_full = np.array([np.full((SED_mean.shape[1],SED_mean.shape[2]),i) for i in np.arange(100,900,1)]) * u.micron

model_full = 0.
model = 0.
for k in np.arange(n_mbb):
    model_full += mod_opt.MBB_l_adim(wavelength_mean_full,NHI[k],params[0+(k*3)]*u.cm**2,params[1+(k*3)],params[2+(k*3)]*u.K,l0)
    model += mod_opt.MBB_l_adim(wavelength_mean,NHI[k],params[0+(k*3)]*u.cm**2,params[1+(k*3)],params[2+(k*3)]*u.K,l0)

sigfield = params[0::3]
betafield = params[1::3]
Tfield = params[2::3]

stop

#PLOT
lh = 2; lw = 2
fig, axs = plt.subplots(lh, lw, sharex=True, sharey=True, figsize=(10,12.1))
fig.subplots_adjust(top=1.02, bottom=0.03, left=0.01, right=0.99, hspace=0.01, wspace=0.02)
k = 0
for i in np.arange(lh):
    for j in np.arange(lw):
        im = axs[i][j].imshow(NHI.value[k], **tb.imkw_inferno)
        if j == 0: axs[i][j].set_ylabel(r'y')
        axs[i][j].set_xlabel(r'x')
        axs[i][j].axes.xaxis.set_ticklabels([])
        axs[i][j].axis('off')
        divider = make_axes_locatable(axs[i][j])
        cax = divider.append_axes('bottom', size='5%', pad=0.05)
        cbar = fig.colorbar(im, cax=cax, orientation='horizontal', extend="both")
        tick_locator = ticker.MaxNLocator(nbins=5)
        cbar.locator = tick_locator
        cbar.update_ticks()
        cbar.ax.tick_params(labelsize=14.) 
        if i == lh-1 : cbar.set_label(r"N$_{HI}$ / [10$^{18}$ cm$^{-2}$]", fontsize=16.)
        k += 1
plt.savefig('plot/mosaic_field_NHI.pdf', format='pdf')

#PLOT SIGMA
lh = 2; lw = 2
fig, axs = plt.subplots(lh, lw, sharex=True, sharey=True, figsize=(10,12.1))
fig.subplots_adjust(top=1.02, bottom=0.03, left=0.01, right=0.99, hspace=0.01, wspace=0.02)
k = 0
for i in np.arange(lh):
    for j in np.arange(lw):
        im = axs[i][j].imshow(sigfield[k], origin="lower", cmap=planckct())
        if j == 0: axs[i][j].set_ylabel(r'y')
        axs[i][j].set_xlabel(r'x')
        axs[i][j].axes.xaxis.set_ticklabels([])
        axs[i][j].axis('off')
        divider = make_axes_locatable(axs[i][j])
        cax = divider.append_axes('bottom', size='5%', pad=0.05)
        cbar = fig.colorbar(im, cax=cax, orientation='horizontal', extend="both")
        tick_locator = ticker.MaxNLocator(nbins=5)
        cbar.locator = tick_locator
        cbar.update_ticks()
        cbar.ax.tick_params(labelsize=14.) 
        if i == lh-1 : cbar.set_label(r"$\sigma$ / [arb. unit]", fontsize=16.)
        k += 1
plt.savefig('plot/mosaic_sigfield.pdf', format='pdf')

#PLOT BETA
lh = 2; lw = 2
fig, axs = plt.subplots(lh, lw, sharex=True, sharey=True, figsize=(10,12.1))
fig.subplots_adjust(top=1.02, bottom=0.03, left=0.01, right=0.99, hspace=0.01, wspace=0.02)
k = 0
for i in np.arange(lh):
    for j in np.arange(lw):
        im = axs[i][j].imshow(betafield[k], origin="lower", cmap=planckct())
        if j == 0: axs[i][j].set_ylabel(r'y')
        axs[i][j].set_xlabel(r'x')
        axs[i][j].axes.xaxis.set_ticklabels([])
        axs[i][j].axis('off')
        divider = make_axes_locatable(axs[i][j])
        cax = divider.append_axes('bottom', size='5%', pad=0.05)
        cbar = fig.colorbar(im, cax=cax, orientation='horizontal', extend="both")
        tick_locator = ticker.MaxNLocator(nbins=5)
        cbar.locator = tick_locator
        cbar.update_ticks()
        cbar.ax.tick_params(labelsize=14.) 
        if i == lh-1 : cbar.set_label(r"$\beta$ / [arb. unit]", fontsize=16.)
        k += 1
plt.savefig('plot/mosaic_betafield.pdf', format='pdf')

#PLOT T
lh = 2; lw = 2
fig, axs = plt.subplots(lh, lw, sharex=True, sharey=True, figsize=(10,12.1))
fig.subplots_adjust(top=1.02, bottom=0.03, left=0.01, right=0.99, hspace=0.01, wspace=0.02)
k = 0
for i in np.arange(lh):
    for j in np.arange(lw):
        im = axs[i][j].imshow(Tfield[k], origin="lower", cmap=planckct())
        if j == 0: axs[i][j].set_ylabel(r'y')
        axs[i][j].set_xlabel(r'x')
        axs[i][j].axes.xaxis.set_ticklabels([])
        axs[i][j].axis('off')
        divider = make_axes_locatable(axs[i][j])
        cax = divider.append_axes('bottom', size='5%', pad=0.05)
        cbar = fig.colorbar(im, cax=cax, orientation='horizontal', extend="both")
        tick_locator = ticker.MaxNLocator(nbins=5)
        cbar.locator = tick_locator
        cbar.update_ticks()
        cbar.ax.tick_params(labelsize=14.) 
        if i == lh-1 : cbar.set_label(r"T / [arb. unit]", fontsize=16.)
        k += 1
plt.savefig('plot/mosaic_Tfield.pdf', format='pdf')








