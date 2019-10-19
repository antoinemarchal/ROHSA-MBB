import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy import constants as const
from astropy import units as u
from scipy import optimize

import mod_optimize as mod_opt
reload(mod_opt)

import turbulence as tb

#Open data GNILC_IRIS-SFD_adim
path = "/data/amarchal/GNILC_IRIS-SFD_adim/"
# path = "/data/amarchal/PLANCK_IRIS-SFD_adim/"
fitsname = "Planck_GNILC_IRIS-SFD_adim_G86.fits"
# fitsname = "PLANCK_IRIS-SFD_adim_G86.fits"
hdu = fits.open(path+fitsname)
hdr = hdu[0].header
# cube = hdu[0].data[:,44-32:44+32,44-32:44+32] *1.e6 #Attention rescale
cube = hdu[0].data[:,:64,:64] *1.e6 #Attention rescale

freq = np.array([np.full((cube.shape[1],cube.shape[2]),i) for i in np.array([353,545,857,3000])])*u.GHz
wavelength = (const.c / freq).to(u.micron)
wavelength_full = np.array([np.full((cube.shape[1],cube.shape[2]),i) for i in np.arange(100,900,1)]) * u.micron

#Open HI model CNM WNM with ROHSA
path_HI = "/data/amarchal/G86/model/"
fitsname_HI = "GHIGLS_G86_LVC_IVC_CNM_WNM.fits"
hdu_HI = fits.open(path_HI+fitsname_HI)
hdr_HI = hdu_HI[0].header
# NHI = hdu_HI[0].data[:,44-32:44+32,44-32:44+32] *u.cm**-2 # 1.e18
NHI = hdu_HI[0].data[:,:64,:64] *u.cm**-2 # 1.e18

#Parameters ROHSA+ MBB
n_mbb = 4
lambda_sig = 1.                                                                                                                      
lambda_beta = 1.
lambda_T = 1.                                                                              
lambda_var_sig = 1.
lb_sig = 0.                                                                                                       
ub_sig = np.inf                                                                                                       
lb_beta = 0.                                                                                                       
ub_beta = 4.                                                                                                       
lb_T = 0.                                                                                                       
ub_T = 100.                       
lb_b = 0.                                                                                                       
ub_b = 100.                       
l0 = 349.81617036*u.micron
maxiter_init = 200                                                                                              
maxiter = 1200                                                                                   
m = 10                                                                                                
noise = ".false."                                                                                                              
regul = ".true."                                                                      
descent = ".true."                                                                                                                 
lstd = 1                                                                                                                        
ustd = 20                                                                                                                        
iprint = -1                                                                                                                     
iprint_init = -1                                                 
save_grid = ".true."  

kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]) / 4.

#Start algorithn ROHSA+ MMB
mean_sed = np.mean(cube,(1,2))
mean_NHI = np.mean(NHI,(1,2))

mean_params = [1.5,1.8,16,2.,1.8,17.,1.5,1.8,16,2.,1.8,17.]

#Fit mean SED
# bounds = [(lb_sig,ub_sig), (lb_beta,ub_beta), (lb_T,ub_T), (lb_sig,ub_sig), (lb_beta,ub_beta), (lb_T,ub_T)]
bounds = [(lb_sig,ub_sig), (lb_beta,ub_beta), (lb_T,ub_T), (lb_sig,ub_sig), (lb_beta,ub_beta), (lb_T,ub_T),(lb_sig,ub_sig), (lb_beta,ub_beta), (lb_T,ub_T), (lb_sig,ub_sig), (lb_beta,ub_beta), (lb_T,ub_T)]

init = optimize.fmin_l_bfgs_b(mod_opt.f_g_mean, mean_params, args=(n_mbb, wavelength[:,0,0], mean_sed, l0, mean_NHI), bounds=bounds, 
                                    approx_grad=False, disp=1, maxiter=maxiter_init)


MBB_1 = mod_opt.MBB_l_adim(wavelength_full[:,0,0],mean_NHI[0],init[0][0]*u.cm**2, init[0][1], init[0][2]*u.K, l0)
MBB_2 = mod_opt.MBB_l_adim(wavelength_full[:,0,0],mean_NHI[1],init[0][3]*u.cm**2, init[0][4], init[0][5]*u.K, l0)

MBB = MBB_1 + MBB_2

#Init cube
b = [0.5,0.5,0.5,0.5]
# b = [0.5,0.5]

params = np.zeros((3*n_mbb,2,2))
for i in range(params.shape[1]):
    for j in range(params.shape[2]):
        params[:,i,j] = init[0]

nside = np.max([int(np.ceil(np.log(cube.shape[1]) / np.log(2))), int(np.ceil(np.log(cube.shape[2]) / np.log(2)))])

for n in np.arange(nside)+1: 

    dim_params = params.shape[0]*params.shape[1]*params.shape[2]
    theta = np.zeros(dim_params+len(b))

    SED_mean = mod_opt.mean_cube(cube,2**n)
    NHI_mean = mod_opt.mean_cube(NHI,2**n) * u.cm**-2
    
    freq_mean = np.array([np.full((SED_mean.shape[1],SED_mean.shape[2]),i) for i in np.array([353,545,857,3000])])*u.GHz
    wavelength_mean = (const.c / freq_mean).to(u.micron)

    bounds = mod_opt.init_bounds(SED_mean, params, lb_sig, ub_sig, lb_beta, ub_beta, lb_T, ub_T)

    theta[:dim_params] = params.ravel()    
    for i in np.arange(len(b)): 
        theta[dim_params+i] = b[i]
        bounds.append((lb_b, ub_b))

    theta = optimize.fmin_l_bfgs_b(mod_opt.f_g, theta, args=(n_mbb, wavelength_mean, SED_mean, l0, NHI_mean, lambda_sig, lambda_beta, lambda_T, 
                                                             lambda_var_sig, kernel), bounds=bounds, approx_grad=False, disp=1, maxiter=maxiter)
    params = np.reshape(theta[0][:dim_params],(params.shape))
    for i in np.arange(len(b)): 
        b[i] = theta[0][dim_params+i]

    if n != nside : params = mod_opt.go_up_level(params)
    

wavelength_mean_full = np.array([np.full((SED_mean.shape[1],SED_mean.shape[2]),i) for i in np.arange(100,900,1)]) * u.micron

model_1 = mod_opt.MBB_l_adim(wavelength_mean_full,NHI_mean[0],params[0]*u.cm**2,params[1],params[2]*u.K,l0)
model_2 = mod_opt.MBB_l_adim(wavelength_mean_full,NHI_mean[1],params[3]*u.cm**2,params[4],params[5]*u.K,l0)

model = 0.
for k in np.arange(n_mbb):
    model += mod_opt.MBB_l_adim(wavelength_mean_full,NHI[k],params[0+(k*3)]*u.cm**2,params[1+(k*3)],params[2+(k*3)]*u.K,l0)









